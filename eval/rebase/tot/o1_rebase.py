import torch
import os
import json
from tqdm import tqdm
from pathlib import Path
from transformers import LlamaForCausalLM, AutoTokenizer
from rebase_utils import *
from data.utils.inference_utils import apiqa, calc_price
from data.utils.io_utils import tload
from data.utils.string_utils import extract_content, remove_special_tokens

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

        self.system_prompt = tload(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/dataset/verifier_index.txt"))


    def generate(self, input_ids, dir_name="tot/outputs/math_o1", temperature=0.5, max_length=2048, do_sample=True, use_rebase=False, **kwargs):
        """
        Generate outputs using either base algorithm or rebase search.
        
        Args:
            use_rebase (bool): Whether to use rebase search algorithm
            rebase_config (dict): Configuration for rebase search, if None uses defaults:
                {
                    "width": self.num_parallel_steps,
                    "max_tokens": max_length,
                    "temperature": temperature,
                    "softmax_temperature": 0.5,
                    "truncate_ratio": 0.8,
                    "select_method": "softmax",
                    "max_step_tokens": 256
                }
        """
        dir_name = f"{dir_name}/{MODEL_NAME}/T{temperature}_N{self.num_parallel_steps}"
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(dir_name, "intermediate_steps.txt")
        
        question = self._extract_question(input_ids)
        thinking_trajectory_so_far = []
        dump_steps = {}
        all_price = []

        if use_rebase:
            # Use default rebase config if none provided
            rebase_config = {
                "width": self.num_parallel_steps,
                "max_tokens": max_length,
                "temperature": temperature,
                "softmax_temperature": 0.2,
                "truncate_ratio": 0.8, # swj: what is default truncate_ratio? 
                "select_method": "softmax",
                # "store_path": dir_name,
                "max_step_tokens": 256
            }
            # Note: rebase will take raw texts as input: input_ids is raw text
            return self._generate_rebase(
                input_ids, 
                question,
                thinking_trajectory_so_far,
                file_name,
                rebase_config,
                **kwargs
            )
        else:
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

    def _generate_rebase(self, input_ids, question, thinking_trajectory_so_far, file_name, config, **kwargs):
        # Initialize tree for rebase search
        tree = Tree(self, config, self)

        depth = 0
        while True:
            tree.reset_running_list()
            continue_search = tree.select_and_expand(depth)
            if not continue_search:
                break
                
            running_list = tree.get_running_list()
            for state, parent in running_list:
                # Generate step using HF model
                outputs = self._generate_step(
                    state, 
                    config["temperature"], 
                    config["max_tokens"], 
                    True,  # do_sample
                    **kwargs
                )
                scores = self._calculate_scores(outputs, question, thinking_trajectory_so_far)
                tree.insert(outputs, scores, parent)
                
            depth += 1
            if depth >= 25:  # Max depth limit
                break

        # Process results
        history_list = tree.get_history_list() 
        all_nodes = tree.get_nodes()
        final_outputs = []
        
        for node in all_nodes:
            if node.is_leaf():
                step_scores = []
                last_node = node 
                while last_node.get_depth() > 0:
                    step_scores.append(last_node.get_score())
                    last_node = last_node.get_parent()
                final_outputs.append({
                    "text": node.get_text(),
                    "scores": step_scores
                })

        # Write outputs
        with open(file_name, 'a') as f:
            f.write(f"Tree search results:\n")
            for output in final_outputs:
                f.write(f"Path: {output['text']}\n")
                f.write(f"Scores: {output['scores']}\n")
            f.write("\n==============================================\n")

        # Return best path
        best_output = max(final_outputs, key=lambda x: sum(x['scores']))
        return self._update_input_ids(input_ids, best_output['text'])

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
            f.write(json.dumps(problem_dict))
            f.write("\n")

    def _write_price(self, file_name, question, all_price):
        with open(file_name.replace(".txt", "_price.jsonl"), "a") as f:
            f.write(json.dumps({"question": question, "price": sum(all_price)}))
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
    framework = "transformers"
    model_name = "qfq/Llama-3.1-8B-Instruct-EI1-2ep-sft"
    #
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
        pass
