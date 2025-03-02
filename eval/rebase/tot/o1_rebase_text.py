import torch
import os
import json

from tqdm import tqdm
from pathlib import Path
from transformers import LlamaForCausalLM, AutoTokenizer
from sglang import function, gen, RuntimeEndpoint
from rebase_utils import *
from metric_utils import *
from metric_utils import _majority_vote
import sys
from data.utils.inference_utils import apiqa, calc_price
from data.utils.io_utils import tload
from data.utils.string_utils import extract_content, remove_special_tokens
from ipdb import set_trace as bp
# swj change later
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MODEL_NAME = "gpt-4o-mini"

class o1:
    def __init__(
        self,
        config=None,
        tokenizer=None,
        num_parallel_steps=4,
        parallel_step_interval=1,
        step_eos="<|reserved_special_token_2|>",
        think_eos="<|reserved_special_token_1|>",
        answer_eos="<|eot_id|>",
        policy_host="http://localhost:30006", # swj: 30003 our models' policy host
        reward_host="http://localhost:30010", 
        reward_model_type="llemma",
        policy_model_type="llemma",
        use_rebase=True,
        **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("qfq/Llama-3.1-8B-Instruct-EI1-2ep-sft")
        self.num_parallel_steps = num_parallel_steps
        self.parallel_step_interval = parallel_step_interval
        self.step_eos = step_eos
        self.think_eos = think_eos
        self.answer_eos = answer_eos
        
        self.eos_token_id = [
            self.tokenizer.encode(eos_token)[1]
            for eos_token in [step_eos, think_eos, answer_eos]
        ]

        self.system_prompt = tload(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)).rsplit("/")[0]), "data/dataset/verifier_index.txt"))
        self.policy_host = policy_host
        self.reward_host = reward_host
        self.use_rebase = use_rebase
        self.reward_model_type = reward_model_type
        self.policy_model_type = policy_model_type
        if self.use_rebase:
            # hard code their default values in rebase
            self.num_threads = 1
            self.softmax_temperature = 0.2

    '''
    generate
    '''
    def generate(self, input_ids, dir_name="/home/weijias/o1/output_swj/openai_math/rebase/qwen_20241130_093851", temperature=0.5, max_length=2048, do_sample=True, **kwargs):
        if self.use_rebase:
            dir_name = f"{dir_name}/T{temperature}_N{self.num_parallel_steps}"
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            print("dir_name: ", dir_name)
            file_name = os.path.join(dir_name, "intermediate_steps.jsonl")
            # it should be a list of strings 
            prompts = input_ids 
            # params is a dict: {'temperature': 1.0, 'max_step_tokens': 256, 'max_tokens': 1024, 'select_method': 'softmax', 'num_threads': 8, 'softmax_temperature': 0.2, 'reward_model_type': 'llemma', 'policy_model_type': 'llemma', 'store_path': './exp_results/rebase_16/', 'width': 16}
            rebase_config = {
                "temperature": temperature,
                "max_step_tokens": max_length,
                "max_tokens": max_length,
                "select_method": "softmax",
                "num_threads": self.num_threads,
                "softmax_temperature": self.softmax_temperature,
                "reward_model_type": self.reward_model_type,
                "policy_model_type": self.policy_model_type,
                "width": self.num_parallel_steps
            }
            # Note: rebase will take raw texts as input: input_ids is raw text
            return self._generate_rebase(prompts, rebase_config, file_name)
        else:
            dir_name = f"{dir_name}/{MODEL_NAME}/T{temperature}_N{self.num_parallel_steps}"
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            file_name = os.path.join(dir_name, "intermediate_steps.txt")
            question = self._extract_question(input_ids)
            thinking_trajectory_so_far = []
            dump_steps = {}
            all_price = []
            return self._generate_base(
                input_ids,
                question,
                thinking_trajectory_so_far,
                dump_steps,
                all_price,
                file_name,
                temperature,
                max_length,
                do_sample,
                **kwargs
            )

    '''
    rebase
    '''
    def _generate_rebase(self, prompts, params, file_name):
            """
            Generate text using the rebase algorithm for multiple prompts.
            """
            # params is a dict: {'temperature': 1.0, 'max_step_tokens': 256, 'max_tokens': 1024, 'select_method': 'softmax', 'num_threads': 8, 'softmax_temperature': 0.2, 'reward_model_type': 'llemma', 'policy_model_type': 'llemma', 'store_path': './exp_results/rebase_16/', 'width': 16}

            input_list_dict = []
            for i, prompt in enumerate(prompts):
                input_list_dict.append({
                    "id": i,
                    "question": prompt,
                    "paras": params,
                    "reward_host": RuntimeEndpoint(self.reward_host)
                })
                # bp()

            states = reward_guided_search.run_batch(
                input_list_dict,
                backend=RuntimeEndpoint(self.policy_host),
                num_threads=self.num_threads,
                progress_bar=True,
            )

            results = []
            total_gen_tokens = 0
            for s in states:
                answer = s.ret_value
                total_gen_tokens += answer["total_tokens"]
                results.append(answer)


            final_answer = []
            # import pdb; pdb.set_trace()
            for prompt, model_answer in zip(prompts, results):
                model_answer = model_answer["model_answer"]
       
                # print(model_answer)
                # bp()
                '''
                aggregate answers
                
                best_answers = {}
                # Majority voting (unweighted)
                majority_answer = _majority_vote(model_answer, weighted=False)
                best_answers["majority_vote"] = majority_answer
                '''
                # bp()
                # Majority voting (weighted using min aggregation)
                # weighted_majority = _majority_vote(model_answer, weighted=True, weight_func="last")
                answers = [cand["text"] for cand in model_answer]
                final_answer.append(answers)
                # dump the model answer and question and answer. make a dict
                ex = {"question": prompt, "model_answer": model_answer} # "answer": weighted_majority
                with open(file_name, 'a') as f:
                    json.dump(ex, f)
                    f.write("\n")

            # bp()
            # print(final_answer)
            return final_answer


    def _generate_base(self, input_ids, question, thinking_trajectory_so_far, dump_steps, all_price, file_name, temperature, max_length, do_sample, **kwargs):
        """Implementation of original base generation algorithm"""
        with open(file_name, 'a') as f, tqdm() as pbar:
            chosen_step = ""
            for i in range(max_length):
                if self._should_generate_multiple_outputs(chosen_step):
                    outputs, decoded_output = self._generate_multiple_outputs(input_ids, temperature, max_length, do_sample, **kwargs)
                    ranked_output, ranked_index, price = self.reward_fn_unified(decoded_output, question, thinking_trajectory_so_far)
                    all_price.append(price)
                    chosen_step = self._process_ranked_output(ranked_output, thinking_trajectory_so_far, dump_steps, i, f, decoded_output, ranked_index)
                else:
                    chosen_step = self._generate_single_output(input_ids, temperature, max_length, do_sample, **kwargs)
                    f.write(f"Iteration {i} - Final output:\n {chosen_step}\n")

                if self._should_break(chosen_step, input_ids, max_length):
                    break

                input_ids = self._update_input_ids(input_ids, chosen_step)
                pbar.update(1)

            self._write_final_output(input_ids, f)

        self._write_problem_dict(file_name, question, thinking_trajectory_so_far, dump_steps)
        self._write_price(file_name, question, all_price)
        return input_ids

    def _generate_step(self, state, temperature, max_length, do_sample, **kwargs):
        """Generate next step using HF model"""
        input_ids = self.tokenizer(state.text(), return_tensors="pt").input_ids.cuda()
        outputs = super().generate(
            input_ids,
            eos_token_id=self.eos_token_id,
            temperature=temperature,
            max_length=max_length,
            do_sample=do_sample,
            **kwargs
        )
        return self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=False)[0]

    def _calculate_scores(self, outputs, question, thinking_trajectory_so_far):
        """Calculate scores for generated steps"""
        ranked_output, _, price = self.reward_fn_unified([outputs], question, thinking_trajectory_so_far)
        return ranked_output[0] if ranked_output else 0.0

    def reward_fn_unified(self, outputs, question, thinking_trajectory_so_far):
        next_steps = outputs

        if len(next_steps) == 1:
            return next_steps, ['A'], 0  # No ranking needed, no price calculated

        next_steps_multiple_choice = {f"{chr(65+i)}": step for i, step in enumerate(next_steps)}
        thinking_trajectory_so_far_text = "\n".join(f"step{i+1}: {step}" for i, step in enumerate(thinking_trajectory_so_far))
        next_steps_multiple_choice_text = "\n".join(f"{chr(65+i)}: {remove_special_tokens(step)}" for i, step in enumerate(next_steps))

        prompt = (f"{self.system_prompt}\n\n\n===========\n\n\n"
                f"## Question\n{question}\n\n"
                f"## Chain of Thoughts so far\n{thinking_trajectory_so_far_text}\n\n"
                f"## Next Thinking Steps\n{next_steps_multiple_choice_text}\n\n"
                f"## Output")

        ranked_output, usage = apiqa(prompt, MODEL_NAME, "", json_format=True)
        price = calc_price(MODEL_NAME, usage)

        try:
            ranked_output = json.loads(ranked_output)
            ranked_indices = ranked_output.get("rank", [])
        except json.JSONDecodeError:
            print("Warning: Failed to parse ranked output JSON")
            ranked_indices = []

        if not ranked_indices:
            print("Warning: No ranked output, returning the first step")
            ranked_indices = ['A']

        try:
            ranked_think_step = [next_steps_multiple_choice[i] for i in ranked_indices]
        except KeyError:
            print("Warning: Invalid ranking, using original order")
            ranked_think_step = next_steps
            ranked_indices = [chr(65+i) for i in range(len(next_steps))]

        return ranked_think_step, ranked_indices, price

    def _extract_question(self, input_ids):
        question = self.tokenizer.decode(input_ids[0])
        return extract_content(question, "user<|end_header_id|>\n\n", "Put your answer on its own line after")

    def _should_generate_multiple_outputs(self, chosen_step):
        return self.num_parallel_steps > 1 and not chosen_step.endswith(self.think_eos)

    def _generate_multiple_outputs(self, input_ids, temperature, max_length, do_sample, **kwargs):
        kwargs.pop("attention_mask", None)
        outputs = super().generate(input_ids, eos_token_id=self.eos_token_id, num_beams=1, num_return_sequences=self.num_parallel_steps, temperature=temperature, max_length=max_length, do_sample=do_sample, **kwargs)
        decoded_output = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=False)
        return outputs, [self._remove_extra_eos(o) for o in decoded_output]

    def _generate_single_output(self, input_ids, temperature, max_length, do_sample, **kwargs):
        kwargs.pop("attention_mask", None)
        outputs = super().generate(input_ids, eos_token_id=self.eos_token_id, temperature=temperature, max_length=max_length, do_sample=do_sample, **kwargs)
        return self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=False)[0]

    def _remove_extra_eos(self, o):
        if self.think_eos in o:
            o = o.split(self.think_eos)[0]
            return o + self.think_eos
        elif self.step_eos in o:
            return o.split(self.step_eos)[0] + self.step_eos
        return o

    def _process_ranked_output(self, ranked_output, thinking_trajectory_so_far, dump_steps, i, f, decoded_output, ranked_index):
        chosen_step = ranked_output[0] if ranked_output else None
        if chosen_step:
            thinking_trajectory_so_far.append(remove_special_tokens(chosen_step))
            dump_steps[i] = ranked_output
            f.write(f"Iteration {i} - Multiple outputs:\n {ranked_index}\n")
            for j, step in enumerate(ranked_output):
                f.write(f"  Rank {j+1}: {step}\n")
            f.write(f"Decoded output: {decoded_output}\n")
        return chosen_step

    def _should_break(self, chosen_step, input_ids, max_length):
        return chosen_step is None or len(input_ids) > max_length

    def _update_input_ids(self, input_ids, chosen_step):
        chosen_step_ids = self.tokenizer(chosen_step, return_tensors="pt").input_ids.cuda()[0][1:]
        if chosen_step.endswith(self.answer_eos) or chosen_step.endswith(self.think_eos) or chosen_step.endswith(self.step_eos):
            return torch.cat([input_ids, chosen_step_ids.unsqueeze(0)], dim=1)
        else:
            print(f"Warning: Chosen step does not end with a valid eos token: {chosen_step}")
            return input_ids

    def _write_final_output(self, input_ids, f):
        text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        f.write(f"Final output: {text[0]}\n")
        f.write("\n==============================================\n")

    def _write_problem_dict(self, file_name, question, thinking_trajectory_so_far, dump_steps):
        problem_dict = {"question": question, "thinking_trajectory_so_far": thinking_trajectory_so_far, "dump_steps": dump_steps}
        with open(file_name.replace(".txt", ".jsonl"), "a") as f:
            f.write(json.dumps(problem_dict, indent=4))
            f.write("\n")

    def _write_price(self, file_name, question, all_price):
        with open(file_name.replace(".txt", "_price.jsonl"), "a") as f:
            f.write(json.dumps({"question": question, "price": sum(all_price)}, indent=4))
            f.write("\n")

def prepare_input_ids(prompts, tokenizer):
    dialog = [{"role" : "user", "content": f"{prompts[0]}\n\nPut your answer on its own line after \"Answer:\""}]
    text = tokenizer.apply_chat_template(dialog, tokenize=False)
    text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    text = [text]
    return tokenizer(text, return_tensors="pt").input_ids.cuda()


if __name__ == "__main__":
    # prompts = ["Find the number of $x$-intercepts on the graph of $y = \sin \frac{1}{x}$ (evaluated in terms of radians) in the interval $(0.0001, 0.001).$"]
    # prompts = ["What is the domain of the function $$u(x) = \frac{1}{\sqrt x}~?$$ Express your answer in interval notation."]
    # prompts = ["Each of the nine dots in this figure is to be colored red, white or blue. No two dots connected by a segment (with no other dots between) may be the same color. How many ways are there to color the dots of this figure? [asy] draw((-75,0)--(-45,0)--(-60,26)--cycle); draw((0,0)--(30,0)--(15,26)--cycle); draw((75,0)--(105,0)--(90,26)--cycle); draw((-60,26)--(90,26)); draw((-45,0)--(75,0)); dot((-75,0)); dot((-45,0)); dot((-60,26)); dot((15,26)); dot((0,0)); dot((30,0)); dot((90,26)); dot((75,0)); dot((105,0)); [/asy]"]
    # prompts = ["Three of the edges of a cube are $\overline{AB}, \overline{BC},$ and $\overline{CD},$ and $\overline{AD}$ is an interior diagonal. Points $P, Q,$ and $R$ are on $\overline{AB}, \overline{BC},$ and $\overline{CD},$ respectively, so that $AP = 5, PB = 15, BQ = 15,$ and $CR = 10.$ What is the area of the polygon that is the intersection of plane $PQR$ and the cube?"]
    prompts = ["How many positive whole-number divisors does 196 have?"]
    step_eos = "<|reserved_special_token_2|>"
    think_eos = "<|reserved_special_token_1|>"
    answer_eos = "<|eot_id|>"
    framework = "sglang"
    model_name = "qfq/Llama-3.1-8B-Instruct-EI1-2ep-sft"

    if framework == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        #model = AutoModelForCausalLM.from_pretrained("qfq/Llama-3.1-8B-Instruct-EI1-2ep-sft").cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = o1.from_pretrained(model_name, tokenizer=None, num_parallel_steps=4, step_eos=step_eos, think_eos=think_eos, answer_eos=answer_eos)
        model = model.cuda()
        input_ids = prepare_input_ids(prompts, tokenizer)
        print(model.generate(input_ids))
    elif framework == "vllm":
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=0.3, top_p=0.95, stop_token_ids=[step_eos, think_eos, answer_eos])
        model = LLM("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer = model.tokenizer
        reward_fn = lambda outputs: model([o + "\n\nIs the above a good step? Yes or No?" for o in outputs], sampling_params)["logits"]
        # vLLM only allows requesting 20 logprobs so need to use generate instead
        reward_fn = lambda outputs: model("Prompt: " + outputs[0].prompt + "\n\nPossible next steps:\n" + "\n".join([chr(65+i) + ": " + o for i, o in enumerate(outputs)]) + "\n\nGiven the prompt, which is the best next step?", sampling_params)["logits"]
        ### WIP ###
    elif framework == "sglang":
        policy_host = "http://localhost:30000"
        reward_host = "http://localhost:30010"
        use_base = True
        reward_model_type = "llemma"
        policy_model_type = "llemma"
        temperature = 1
        model = o1(model_name, tokenizer=None, temperature=temperature, num_parallel_steps=4, step_eos=step_eos, think_eos=think_eos, answer_eos=answer_eos, policy_host=policy_host, reward_host=reward_host, reward_model_type=reward_model_type, policy_model_type=policy_model_type, use_rebase=use_base)
        print(model.generate(prompts))