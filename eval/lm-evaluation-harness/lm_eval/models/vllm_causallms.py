import copy
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, configure_pad_token, undistribute
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
)


try:
    import ray
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

eval_logger = eval_logger


@register_model("vllm")
class VLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("vllm"):
            raise Exception(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert "cuda" in device or device is None, "vLLM only supports CUDA"
        assert (
            max_length is None or max_model_len is None
        ), "Either max_length or max_model_len may be provided, but not both"

        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
        }
        self.model_args.update(kwargs)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else batch_size
        )
        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["worker_use_ray"] = True
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )
        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )
        self.tokenizer = configure_pad_token(self.tokenizer)
        self.add_bos_token = add_bos_token
        if "gemma" in pretrained.lower():
            self.add_bos_token = True
            eval_logger.info(
                "Found 'gemma' in model name, a BOS token will be used as Gemma series models underperform without it."
            )

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version(
                "0.3.0"
            ), "lora adapters only compatible with vllm > v0.3.0."
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding: Union[List[List[int]], List[int]] = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)

            rejection_sample = kwargs.pop("rejection_sample", None)
            if rejection_sample:
                if (kwargs.get("temperature_thinking", 0) == 0) and (kwargs.get("temperature", 0) == 0):
                    print("Warning: Rejection sampling works best with temperature/temperature_thinking > 0.")
                assert "max_tokens_thinking" in kwargs, "Rejection sampling requires max_tokens_thinking to be set."

            outputs_thinking = None
            if any(["thinking" in k for k in kwargs]) or rejection_sample:
                print("Separating thinking and answering generation.")
                thinking_start = kwargs.pop("thinking_start", "<|im_start|>think")
                thinking_end = kwargs.pop("thinking_end", "<|im_start|>answer")
                thinking_n_ignore = kwargs.pop("thinking_n_ignore", None)
                thinking_n_ignore_str = kwargs.pop("thinking_n_ignore_str", None) # e.g. "Let me double check step-by-step.")
                if thinking_n_ignore_str is not None:
                    print(f"Thinking ignore string: {thinking_n_ignore_str}")
                    thinking_n_ignore_str_tok = self.tok_encode(thinking_n_ignore_str)
                until_thinking = [kwargs.pop("until_thinking", "<|im_start|>")]
                if "until_thinking_2" in kwargs:
                    until_thinking.append(kwargs.pop("until_thinking_2"))
                if stop is not None:
                    until_thinking.extend(stop)
                print(f"Thinking start: {thinking_start}, Thinking end: {thinking_end}, Stop: {until_thinking}")
                thinking_start_tok = self.tok_encode(thinking_start)
                thinking_end_tok = self.tok_encode(thinking_end)
                thinking_end_max = thinking_end + "\nFinal Answer:"
                thinking_end_max_tok = self.tok_encode(thinking_end_max)
                newline_tok = self.tok_encode("\n")
                # Cast to list to avoid `dictionary changed size during iteration`
                sampling_params_thinking = {k.replace("_thinking", ""): kwargs.pop(k) for k, v in list(kwargs.items()) if "thinking" in k}
                # Add all other kwargs but keep sampling_params_thinking version if duplicate key
                sampling_params_thinking = {**kwargs, **sampling_params_thinking}
                if "max_tokens" in sampling_params_thinking:
                    if sampling_params_thinking["max_tokens"] == "auto":
                        # Leave 100 tokens for answer
                        sampling_params_thinking["max_tokens"] = max_tokens - max([len(x) for x in requests]) - len(thinking_start_tok) - len(thinking_end_max_tok) - 100
                        print(f"Auto setting max_tokens_thinking to {sampling_params_thinking['max_tokens']}")
                    else:
                        sampling_params_thinking["max_tokens"] = int(sampling_params_thinking["max_tokens"])
                    if rejection_sample:
                        sampling_params_thinking["max_tokens"] += 1
                else:
                    sampling_params_thinking["max_tokens"] = max_tokens
                until_thinking_tok = self.tok_encode(until_thinking)
                if ("min_tokens" in sampling_params_thinking) or (thinking_n_ignore is not None):
                    if thinking_n_ignore is not None:
                        sampling_params_thinking["min_tokens"] = 1
                    else:
                        sampling_params_thinking["min_tokens"] = int(sampling_params_thinking["min_tokens"])
                    assert all([len(x) == 1 for x in until_thinking_tok]), "min_tokens_thinking only supports until_thinking tokens that are 1 token long"
                    # min_tokens will not ignore `stop`, only `stop_token_ids` are ignored so need to use these
                    sampling_params_thinking["stop_token_ids"] = [x[0] for x in until_thinking_tok]
                else:
                    sampling_params_thinking["stop"] = until_thinking
                requests = [req + thinking_start_tok for req in requests]
                sampling_params = SamplingParams(**sampling_params_thinking)

                if rejection_sample:
                    requests_thinking = copy.deepcopy(requests)
                    outputs_thinking = [None] * len(requests_thinking)
                    i = 0
                    while True:
                        outputs_tmp = self.model.generate(
                            prompt_token_ids=requests_thinking,
                            sampling_params=sampling_params,
                            use_tqdm=True if self.batch_size == "auto" else False,
                        )
                        # Save ones that are already below the limit
                        outputs_tmp2 = copy.deepcopy(outputs_thinking)
                        for j, o in enumerate(outputs_tmp):
                            if len(o.outputs[0].token_ids) <= sampling_params_thinking["max_tokens"] - 1:
                                if outputs_tmp2[j] is None:
                                    outputs_thinking[j] = o
                                else:
                                    for k, t in enumerate(outputs_tmp2[j:] + outputs_tmp2[:j]):
                                        if t is None:
                                            idx = j + k if j + k < len(outputs_thinking) else j + k - len(outputs_thinking)
                                            outputs_thinking[idx] = o
                                            break

                        # Collect requests remaining
                        requests_thinking_new = [None] * len(requests_thinking)
                        for j, o in enumerate(outputs_thinking):
                            if outputs_thinking[j] is None:
                                requests_thinking_new[j] = requests_thinking[j]

                        samples_left = sum([x is not None for x in requests_thinking_new])

                        if not(samples_left): break
                        gen_tokens_all = [len(o.outputs[0].token_ids) for o in outputs_tmp]
                        print(f"Samples left: {samples_left}, gen_tokens_all: {gen_tokens_all}, i: {i}")
                        # Fill up empty request slots with duplicates of other requests that need to be rerun
                        # Fill each None with the first non-None request after it
                        for j, r in enumerate(requests_thinking_new):
                            if r is None:
                                for k, r2 in enumerate(requests_thinking_new[j:] + requests_thinking_new[:j]):
                                    if r2 is not None:
                                        requests_thinking_new[j] = r2
                                        break
                        requests_thinking = requests_thinking_new
                        i += 1
                    print(f'Rejection sampling took {i} iterations to generate {sampling_params_thinking["max_tokens"] - 1} tokens.')
                elif thinking_n_ignore is not None:
                    print("Will ignore end of thinking " + str(thinking_n_ignore) + " times.")
                    # Add 1 to account for first generation w/o ignoring
                    thinking_n_ignore = int(thinking_n_ignore) + 1
                    outputs_thinking = [None] * len(requests)
                    requests_tmp = copy.deepcopy(requests)
                    indices = list(range(len(requests)))
                    for i in range(thinking_n_ignore):
                        outputs_tmp = self.model.generate(
                            prompt_token_ids=requests_tmp,
                            sampling_params=sampling_params,
                            use_tqdm=True if self.batch_size == "auto" else False,
                        )
                        indices_new = []
                        requests_tmp_new = []
                        for j, o in enumerate(outputs_tmp):
                            idx = indices[j]
                            assert len(o.outputs) == 1
                            cont = list(o.outputs[0].token_ids)
                            # Final; do not generate further
                            if (o.outputs[0].finish_reason == "length") or (i == thinking_n_ignore - 1):
                                if outputs_thinking[idx] is not None:
                                    outputs_thinking[idx].outputs[0].text += o.outputs[0].text
                                    outputs_thinking[idx].outputs[0].token_ids += cont
                                    outputs_thinking[idx].outputs[0].finish_reason = o.outputs[0].finish_reason
                                else:
                                    outputs_thinking[idx] = o
                                    outputs_thinking[idx].outputs[0].token_ids = cont
                                    outputs_thinking[idx].outputs[0].finish_reason = o.outputs[0].finish_reason
                            else:
                                # When using `stop`, the stop text will not be in the text, but still in the token_ids so remove it
                                for toks in until_thinking_tok:
                                    if cont[-len(toks):] == toks:
                                        cont = cont[:-len(toks)]
                                
                                if thinking_n_ignore_str is not None:
                                    cont += thinking_n_ignore_str_tok
                                    o.outputs[0].text += thinking_n_ignore_str

                                if outputs_thinking[idx] is not None:
                                    outputs_thinking[idx].outputs[0].text += o.outputs[0].text
                                    outputs_thinking[idx].outputs[0].token_ids += cont
                                else:
                                    outputs_thinking[idx] = o
                                    outputs_thinking[idx].outputs[0].token_ids = cont

                                requests_tmp_new.append(requests_tmp[j] + cont)
                                indices_new.append(idx)
                        requests_tmp = requests_tmp_new
                        indices = indices_new
                    for idx in list(range(len(requests))):
                        if len(outputs_thinking[idx].outputs[0].token_ids) > sampling_params_thinking["max_tokens"]:
                            print(f'Warning: Generated more than {sampling_params_thinking["max_tokens"]} tokens. Cutting.')
                            outputs_thinking[idx].outputs[0].token_ids = outputs_thinking[idx].outputs[0].token_ids[:sampling_params_thinking["max_tokens"]]

                else:
                    outputs_thinking = self.model.generate(
                        prompt_token_ids=requests,
                        sampling_params=sampling_params,
                        use_tqdm=True if self.batch_size == "auto" else False,
                    )

                for i, o in enumerate(outputs_thinking):
                    assert len(o.outputs) == 1
                    cont = list(o.outputs[0].token_ids)
                    # When using `stop`, the stop text will not be in the text, but still in the token_ids so remove it
                    for toks in until_thinking_tok:
                        if cont[-len(toks):] == toks:
                            cont = cont[:-len(toks)]

                    if o.outputs[0].finish_reason == "length":
                        assert not rejection_sample, "Rejection sampling should not reach this point."
                        # \n appears a lot so a decent chance it happend to just be the last token in which case we don't need to add a newline
                        if (o.outputs[0].text[-1] == "\n") or (thinking_start[0] == "\n"):
                            requests[i] += cont + thinking_end_max_tok
                            outputs_thinking[i].outputs[0].text = thinking_start + outputs_thinking[i].outputs[0].text + thinking_end_max
                        else:
                            requests[i] += cont + newline_tok + thinking_end_max_tok
                            outputs_thinking[i].outputs[0].text = thinking_start + outputs_thinking[i].outputs[0].text + "\n" + thinking_end_max
                    else:
                        requests[i] += cont + thinking_end_tok
                        outputs_thinking[i].outputs[0].text = thinking_start + outputs_thinking[i].outputs[0].text + thinking_end

            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            @ray.remote
            def run_inference_one_model(
                model_args: dict,
                sampling_params,
                requests: List[List[int]],
                lora_request: LoRARequest,
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    prompt_token_ids=requests,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = (
                (self.model_args, sampling_params, req, self.lora_request)
                for req in requests
            )
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)

        outputs = self.model.generate(
            prompt_token_ids=requests,
            sampling_params=sampling_params,
            use_tqdm=True if self.batch_size == "auto" else False,
            lora_request=self.lora_request,
        )
        if outputs_thinking is not None:
            for i, o in enumerate(outputs):
                assert len(o.outputs) == 1
                outputs[i].outputs[0].text = outputs_thinking[i].outputs[0].text + outputs[i].outputs[0].text
        return outputs

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context)
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)

        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            # add EOS token to stop sequences
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            # cache generations
            for i, (output, context) in tqdm(enumerate(zip(cont, context)), desc="final processing"):
                generated_text = output.outputs[0].text
                # swj hack
                # from ipdb import set_trace as bp
                # bp()
                # check if "Answer:" in generated_text, if not resample cont = self._model_generate(requests=context_encoding, generate=True, max_tokens=max_gen_toks, stop=until, **kwargs) until it reaches "Answer:"
                # max_attemp = 5
                # while "Answer:" not in generated_text:
                #     if max_attemp == 0:
                #         print(f"max_attemp reached, question: {i}")
                #         break
                #     max_attemp -= 1
                #     cont_new = self._model_generate(requests=[context_encoding[i]], generate=True, max_tokens=max_gen_toks, stop=until, **kwargs)
                #     generated_text = cont_new[0].outputs[0].text
                #     print(f"resample until 'Answer:', question: {i}")
                res.append(generated_text)

                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )

                res.append(answer)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        # sampling_params
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            eval_logger.debug(
                "Got `do_sample=False` and no temperature value, setting VLLM temperature to 0.0 ..."
            )
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        return kwargs