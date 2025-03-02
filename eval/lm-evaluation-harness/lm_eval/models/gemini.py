import os
from functools import cached_property
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = utils.eval_logger


def gemini_completion(
    client,
    model: str,
    prompt: str,
    max_tokens_to_sample: int,
    temperature: float,
    stop: List[str],
    **kwargs: Any,
) -> str:
    """Wrapper function around the Gemini API client

    params:
        client:
            Gemini API client
        prompt: str
            Prompt to feed to the model
        max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        stop: List[str]
            List of stop sequences
        kwargs: Any
            Additional model_args to pass to the API client
    """

    try:
        import google.generativeai as genai
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'gemini' LM type, but package `google-generativeai` is not installed. \
please install google-generativeai via `pip install 'google-generativeai'`",
        )

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        eval_logger.warning(
            f"RateLimitError occurred: {e.__cause__}\n Retrying in {sleep_time} seconds"
        )

    @retry_on_specific_exceptions(
        on_exceptions=Exception,
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def completion():
        retry_txt = 'FinishReason.RECITATION'
        while retry_txt == 'FinishReason.RECITATION':
            response = client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens_to_sample,
                    temperature=temperature,
                    stop_sequences=stop,
                    **kwargs,
                )
            )
            try:
                retry_txt = str(response.candidates[-1].finish_reason)
            except Exception as e:
                eval_logger.warning("Got error " + str(e))
                retry_txt = ''
        if response.parts:
            return response.parts[-1].text
        else:
            eval_logger.warning("No response from Gemini API due to " + str(response.candidates[-1].finish_reason))
            return ""
    return completion()


@register_model("gemini-completions")
class Gemini(LM):
    REQ_CHUNK_SIZE = 20  # TODO: not used

    def __init__(
        self,
        batch_size: int = 1,
        model: str = "gemini-2.0-flash-thinking-exp",
        max_tokens_to_sample: int = None,
        temperature: float = 0,
        **kwargs,  # top_p, top_k, etc.
    ) -> None:
        """Gemini API wrapper.

        :param model: str
            Gemini model e.g. 'gemini-2.0-flash-thinking-exp'
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature
        :param kwargs: Any
            Additional model_args to pass to the API client
        """
        super().__init__()

        try:
            import google.generativeai as genai
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gemini' LM type, but package `google-generativeai` is not installed. \
    please install google-generativeai via `pip install 'google-generativeai'`",
            )

        self.model = model
        # defaults to os.environ.get("GEMINI_API_KEY")
        self.client = genai.GenerativeModel(model)
        self.temperature = temperature
        self.max_tokens_to_sample = max_tokens_to_sample
        self.tokenizer = None
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        raise NotImplementedError("No idea about gemini tokenization.")

    @property
    def max_length(self) -> int:
        return 8192

    @property
    def max_gen_toks(self) -> int:
        return self.max_tokens_to_sample

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    def tok_encode(self, string: str) -> List[int]:
        # return self.tokenizer.encode(string).ids
        raise NotImplementedError("No idea about gemini tokenization.")

    def tok_decode(self, tokens: List[int]) -> str:
        raise NotImplementedError("No idea about gemini tokenization.")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        try:
            import google.generativeai as genai
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gemini' LM type, but package `google-generativeai` is not installed. \
    please install google-generativeai via `pip install 'google-generativeai'`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm):
            try:
                inp = request[0]
                request_args = request[1]
                # generation_kwargs
                until = request_args.get("until")
                max_gen_toks = request_args.get("max_gen_toks", self.max_length)
                temperature = request_args.get("temperature", self.temperature)
                response = gemini_completion(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens_to_sample=max_gen_toks,
                    temperature=temperature,
                    stop=until,
                    **self.kwargs,
                )
                res.append(response)
                #import pdb; pdb.set_trace()

                self.cache_hook.add_partial("generate_until", request, response)
            except Exception as e:  # type: ignore # noqa: F821
                eval_logger.critical(f"API error {e}")
                break

        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

