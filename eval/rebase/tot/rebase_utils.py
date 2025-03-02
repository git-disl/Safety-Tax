from sglang import function, gen, RuntimeEndpoint
import torch
from sglang import function, gen, RuntimeEndpoint
import fcntl
import os
import math
import yaml
import torch.nn.functional as F
import threading
import argparse
import json
import re
import time
import sys
from ipdb import set_trace as bp

step_eos="<|reserved_special_token_2|>"
think_eos="<|reserved_special_token_1|>"

def convert_text_to_reward_format_llama(text):
    question_start = text.find("user<|end_header_id|>\n\n") + len("user<|end_header_id|>\n\n")
    question_end = text.find("Put your answer on its own line after")
    question = text[question_start:question_end].strip() 
    steps = re.split(r'\n<\|reserved_special_token_[012]\|>\n', text)[1:]
    # answer = text.split("<|reserved_special_token_1|>")[-1]
    # Merge to output format
    
    if steps != []:
        steps[-1] = steps[-1].replace("<|eot_id|>", "").replace("Answer:", f"Step {len(steps)}: The answer is: ")
        separator = "Step {i}: "
    output = (
        question.strip() +
        " ".join([separator.format(i=i+1) + step.strip() + " ки\n" for i, step in enumerate(steps)])) # Remove the last "\n"
    if not output.endswith(" ки\n"):
        output += " ки\n"
    return output

def convert_text_to_reward_format_qwen(text):
    # Find question between user and "Put your answer"
    question_start = text.find("<|im_start|>user\n") + len("<|im_start|>user\n")
    question_end = text.find("Put your answer on its own line after")
    question = text[question_start:question_end].strip()

    # Split steps by <|im_start|>step marker
    steps = re.split(r'<\|im_start\|>step\d+\n', text)[1:]
    
    # Format output
    if steps:
        # Remove answer section from last step if present
        # <|im_start|>answer\nAnswer: \\[ f(-2) + f(-1) + f(0) = \\dfrac{14}{3} \\]
        # bp()
        # if "<|im_start|>answer\nAnswer:" in steps[-1]:
            # bp()
        # swj change
        steps[-1] = steps[-1].replace("<|im_start|>answer\nAnswer: ", f"Step {len(steps)}: The answer is: ")
        # steps[-1] = re.split(r'<\|im_start\|>answer\n', steps[-1])[0].strip()
        separator = "Step {i}: "
        
    output = (
        question.strip() + 
        " ".join([separator.format(i=i+1) + step.strip() + " ки\n" for i, step in enumerate(steps)])
    )
    
    if not output.endswith(" ки\n"):
        output += " ��и\n"
        
    return output

def convert_text_to_reward_format_qwen_steps_left(text):
    # Find question between user and "Put your answer"
    question_start = text.find("<|im_start|>user\n") + len("<|im_start|>user\n")
    question_end = text.find("\n\nThink for up to")
    # bp()
    question = text[question_start:question_end].strip()

    # Split steps by <|im_start|>step marker
    steps = re.split(r'<\|im_start\|>\d+ steps left\n', text)[1:]
    steps = [step.replace("\n<|im_start|>", "") for step in steps]
    # bp()
    # Format output
    if steps:
        # Remove answer section from last step if present
        # <|im_start|>answer\nAnswer: \\[ f(-2) + f(-1) + f(0) = \\dfrac{14}{3} \\]
        # bp()
        # if "<|im_start|>answer\nAnswer:" in steps[-1]:
            # bp()
        # swj change
        # steps[-1] = steps[-1].replace("<|im_start|>answer\nAnswer: ", f"Step {len(steps)}: The answer is: ")
        # steps[-1] = re.split(r'<\|im_start\|>answer\n', steps[-1])[0].strip()
        separator = "Step {i}: "
        
    output = (
        question.strip() + 
        " ".join([separator.format(i=i+1) + step.strip() + " ки\n" for i, step in enumerate(steps)])
    )
    
    if not output.endswith(" ки\n"):
        output += " ки\n"
        
    return output

class TreeNode:
    def __init__(self, id, state, score, num_step_tokens=0, parent=None):
        self.id = id
        self.state = state
        self.text_ = state.text()
        self.text_processed_ = state.text_processed()
        # print("self.text_: ", self.text_)
        self.score_ = score
        self.parent = parent
        self.leaf_ = False
        self.cum_tokens = 0
        self.num_step_tokens = 0
        # bp()
        # if parent is not None and ("The answer is" in self.text_processed_):
        if parent is not None and ("The final answer is" in self.text_processed_):
            self.leaf_ = True
        if parent is not None:
            self.depth = parent.get_depth() + 1
            self.cum_tokens += num_step_tokens
        else:
            self.depth = 0
            self.cum_tokens = num_step_tokens
     
    def get_id(self):
        return self.id
    
    def get_parent(self):
        return self.parent
    
    def get_text(self):
        return self.text_

    def get_state(self):
        return self.state
    
    def get_depth(self):
        return self.depth
    
    def get_score(self):
        return self.score_
    
    def is_leaf(self):
        return self.leaf_
    
    def get_cum_tokens(self):
        return self.cum_tokens

class Tree:
    def __init__(self, root_state, paras, reward_backend):
        self.size_ = 1
        self.nodes = []
        self.paras = paras
        self.reward_backend = reward_backend
        self.root_ = TreeNode(0,root_state, 1.0)
        self.remaining_width = paras["width"]
        self.history_list = []
        self.running_list = []
        self.depth_nodes = [[] for i in range(100)]
        self.nodes.append(self.root_)
        self.depth_nodes[0].append(self.root_)
    
    def reset_running_list(self):
        self.running_list = []
    
    def get_running_list(self):
        return self.running_list
    
    def get_history_list(self):
        return self.history_list
    
    def get_nodes(self):
        return self.nodes
    
    def expand(self, node, wid):
        state = node.get_state()
        forks = state.fork(wid)
        depth = node.get_depth()
        for fork in forks:
            # get number of total steps
            total_steps = int(os.getenv("TIMELIMITNOANSWER"))
            # extract the last step left 
            # cur_step = int(raw_text.rsplit(" steps left", 1)[0].rsplit("<|im_start|>", 1)[1])
            left_steps = total_steps - depth

            # bp()
            # print("fork: ", fork.text())
            print("fork.text(): ", f"{fork.text()}".replace('\n', '\\n'))
            fork.set_score_backend(self.reward_backend)
            if self.paras["policy_model_type"] == "mistral" or self.paras["policy_model_type"] == "llemma":
                # 
                fork += gen("step", self.paras["max_step_tokens"], stop="Step "+str(depth+2), temperature=self.paras["temperature"])
                # bp()
                # import pdb; pdb.set_trace()
                fork += gen("score", max_tokens=0, forward_only=True, logits_require_id=8094)
                # print("fork.text(): ", f"{fork.text()}")
                # print("fork.scores(): ", fork.scores())
                # bp()
    
            elif "our" in self.paras["policy_model_type"]:
                if "our_qwen" in self.paras["policy_model_type"]:
                    # stop_token = f"<|im_start|>step{depth+2}"
                    stop_token = f"{left_steps-1} steps left"
                elif "our_llama" in self.paras["policy_model_type"]:
                    stop_token = "<|reserved_special_token_2|>"
                # bp()
                fork += gen("step", self.paras["max_step_tokens"], stop=stop_token, temperature=self.paras["temperature"])

                # process the text to make it comparable to the format of the reward model
                ''' LLaMA
                input: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nDefine\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$\n\nPut your answer on its own line after "Answer:"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|reserved_special_token_0|>\nWe have the definitions of the sums $p = \\sum_{k=1}^\\infty \\frac{1}{k^2}$ and $q = \\sum_{k=1}^\\infty \\frac{1}{k^3}$.\n
                output: 'Define\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$Step 1: We can rewrite\n\\begin{align*}\n\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3} &= \\sum_{k = 1}^\\infty \\sum_{j = 1}^\\infty \\frac{1}{(j + k)^3} \\\\ &= \\sum_{k = 1}^\\infty \\sum_{j = 1}^\\infty \\frac{1}{j^3} \\cdot \\frac{1}{(1 + \\frac{k}{j})^3} \\\\ &= \\sum_{j = 1}^\\infty \\frac{1}{j^3} \\sum_{k = 1}^\\infty \\frac{1}{(1 + \\frac{k}{j})^3}. ки\nStep 2: \\end{align*}We want to find an expression for this double sum. ки\n'
                '''
                '''
                <|begin_of_text|><|start_header_id|>system<|end_header_id|> Cutting Knowledge Date: December 2023 Today Date: 26 Jul 2024 <|eot_id|><|start_header_id|>user<|end_header_id|> How many distinct triangles can be constructed by connecting three different vertices of a cube? (Two triangles are distinct if they have different locations in space.) Put your answer on its own line after "Answer:"
                <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
                <|reserved_special_token_0|> First, I need to determine how many vertices there are in a cube. 
                <|reserved_special_token_2|> A cube has 8 vertices, so I have a total of 8 points to choose from. 
                <|reserved_special_token_2|> Next, I want to find out how many ways I can select 3 vertices from these 8 to form a triangle. 
                <|reserved_special_token_2|> It seems this is a combination problem since the order of vertices does not matter for forming a triangle. 
                <|reserved_special_token_2|> To calculate this, I can use the binomial coefficient, which gives the number of ways to choose k elements from a set of n elements. 
                <|reserved_special_token_2|> In our case, n = 8 (the vertices of the cube) and k = 3 (the vertices needed to form a triangle). 
                <|reserved_special_token_2|> The formula for the binomial coefficient is \binom{n}{k} = \frac{n!}{k!(n-k)!}. 
                <|reserved_special_token_2|> Using this formula, I can calculate \binom{8}{3}. <|reserved_special_token_2|> Calculating, I have \binom{8}{3} = \frac{8!}{3!(8-3)!} = \frac{8\cdot7\cdot6}{3\cdot2\cdot1}. <|reserved_special_token_2|> Now I can simplify this further. <|reserved_special_token_2|> This results in \frac{8\cdot7\cdot6}{6} = 8\cdot7 = 56. <|reserved_special_token_2|> So, the number of distinct triangles that can be formed by the vertices of a cube is 56. <|reserved_special_token_2|> Since two triangles are distinct if they have different locations in space, this calculation accurately reflects that. 
                <|reserved_special_token_1|> There are eight vertices of a cube, and we choose three of these to form a triangle. Thus, the number of distinct triangles that can be formed is $\binom{8}{3} = \frac{8\cdot7\cdot6}{3\cdot2} = \boxed{56}$. Answer:56<|eot_id|>
                '''

                ''' Qwen
                input: 
                    <|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIf $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.\n\nSolve the question within 32 steps. Put your answer on its own line after \"Answer:\"<|im_end|>\n<|im_start|>assistant\n 
                    <|im_start|>step1\nWe are given the function \\( f(x) = \\dfrac{3x - 2}{x - 2} \\).\n
                    <|im_start|>step2\nOur goal is to compute \\( f(-2) + f(-1) + f(0) \\) and express the answer as a common fraction.\n
                    <|im_start|>step3\nFirst, let's compute \\( f(-2) \\).\n
                    <|im_start|>step4\nSubstitute \\( x = -2 \\) into the function:\n\\[ f(-2) = \\dfrac{3(-2) - 2}{-2 - 2} \\]\n<|im_start|>step5\nCalculate the numerator:\n\\[ 3(-2) - 2 = -6 - 2 = -8 \\]\n<|im_start|>step6\nCalculate the denominator:\n\\[ -2 - 2 = -4 \\]\n<|im_start|>step7\nSo, \\( f(-2) = \\dfrac{-8}{-4} \\).\n<|im_start|>step8\nSimplify the fraction:\n\\[ \\dfrac{-8}{-4} = \\dfrac{8}{4} = 2 \\]\n<|im_start|>step9\nTherefore, \\( f(-2) = 2 \\).\n<|im_start|>step10\nNext, let's compute \\( f(-1) \\).\n<|im_start|>step11\nSubstitute \\( x = -1 \\) into the function:\n\\[ f(-1) = \\dfrac{3(-1) - 2}{-1 - 2} \\]\n<|im_start|>step12\nCalculate the numerator:\n\\[ 3(-1) - 2 = -3 - 2 = -5 \\]\n<|im_start|>step13\nCalculate the denominator:\n\\[ -1 - 2 = -3 \\]\n<|im_start|>step14\nSo, \\( f(-1) = \\dfrac{-5}{-3} \\).\n<|im_start|>step15\nSimplify the fraction:\n\\[ \\dfrac{-5}{-3} = \\dfrac{5}{3} \\]\n<|im_start|>step16\nTherefore, \\( f(-1) = \\dfrac{5}{3} \\).\n<|im_start|>step17\nNow, let's compute \\( f(0) \\).\n<|im_start|>step18\nSubstitute \\( x = 0 \\) into the function:\n\\[ f(0) = \\dfrac{3(0) - 2}{0 - 2} \\]\n<|im_start|>step19\nCalculate the numerator:\n\\[ 3(0) - 2 = 0 - 2 = -2 \\]\n<|im_start|>step20\nCalculate the denominator:\n\\[ 0 - 2 = -2 \\]\n<|im_start|>step21\nSo, \\( f(0) = \\dfrac{-2}{-2} \\).\n<|im_start|>step22\nSimplify the fraction:\n\\[ \\dfrac{-2}{-2} = 1 \\]\n<|im_start|>step23\nTherefore, \\( f(0) = 1 \\).\n<|im_start|>step24\nNow, sum the computed values:\n\\[ f(-2) + f(-1) + f(0) = 2 + \\dfrac{5}{3} + 1 \\]\n<|im_start|>step25\nConvert all numbers to fractions with a common denominator of 3.\n<|im_start|>step26\nExpress \\( 2 \\) and \\( 1 \\) as fractions over 3:\n\\[ 2 = \\dfrac{6}{3}, \\quad 1 = \\dfrac{3}{3} \\]\n<|im_start|>step27\nAdd the fractions:\n\\[ \\dfrac{6}{3} + \\dfrac{5}{3} + \\dfrac{3}{3} = \\dfrac{6 + 5 + 3}{3} \\]\n<|im_start|>step28\nCompute the numerator:\n\\[ 6 + 5 + 3 = 14 \\]\n<|im_start|>step29\nSo, the sum is:\n\\[ \\dfrac{14}{3} \\]\n<|im_start|>step30\nTherefore, \\( f(-2) + f(-1) + f(0) = \\dfrac{14}{3} \\).\n<|im_start|>step31\nAs a final check, ensure all calculations are correct and the answer is in simplest form.\n<|im_start|>answer\nAnswer: \\[ f(-2) + f(-1) + f(0) = \\dfrac{14}{3} \\]"]], "filtered_resps": ["<|im_start|>step1\nWe are given the function \\( f(x) = \\dfrac{3x - 2}{x - 2} \\).\n<|im_start|>step2\nOur goal is to compute \\( f(-2) + f(-1) + f(0) \\) and express the answer as a common fraction.\n<|im_start|>step3\nFirst, let's compute \\( f(-2) \\).\n<|im_start|>step4\nSubstitute \\( x = -2 \\) into the function:\n\\[ f(-2) = \\dfrac{3(-2) - 2}{-2 - 2} \\]\n<|im_start|>step5\nCalculate the numerator:\n\\[ 3(-2) - 2 = -6 - 2 = -8 \\]\n<|im_start|>step6\nCalculate the denominator:\n\\[ -2 - 2 = -4 \\]\n<|im_start|>step7\nSo, \\( f(-2) = \\dfrac{-8}{-4} \\).\n<|im_start|>step8\nSimplify the fraction:\n\\[ \\dfrac{-8}{-4} = \\dfrac{8}{4} = 2 \\]\n<|im_start|>step9\nTherefore, \\( f(-2) = 2 \\).\n<|im_start|>step10\nNext, let's compute \\( f(-1) \\).\n<|im_start|>step11\nSubstitute \\( x = -1 \\) into the function:\n\\[ f(-1) = \\dfrac{3(-1) - 2}{-1 - 2} \\]\n<|im_start|>step12\nCalculate the numerator:\n\\[ 3(-1) - 2 = -3 - 2 = -5 \\]\n<|im_start|>step13\nCalculate the denominator:\n\\[ -1 - 2 = -3 \\]\n<|im_start|>step14\nSo, \\( f(-1) = \\dfrac{-5}{-3} \\).\n<|im_start|>step15\nSimplify the fraction:\n\\[ \\dfrac{-5}{-3} = \\dfrac{5}{3} \\]\n<|im_start|>step16\nTherefore, \\( f(-1) = \\dfrac{5}{3} \\).\n<|im_start|>step17\nNow, let's compute \\( f(0) \\).\n<|im_start|>step18\nSubstitute \\( x = 0 \\) into the function:\n\\[ f(0) = \\dfrac{3(0) - 2}{0 - 2} \\]\n<|im_start|>step19\nCalculate the numerator:\n\\[ 3(0) - 2 = 0 - 2 = -2 \\]\n<|im_start|>step20\nCalculate the denominator:\n\\[ 0 - 2 = -2 \\]\n<|im_start|>step21\nSo, \\( f(0) = \\dfrac{-2}{-2} \\).\n<|im_start|>step22\nSimplify the fraction:\n\\[ \\dfrac{-2}{-2} = 1 \\]\n<|im_start|>step23\nTherefore, \\( f(0) = 1 \\).\n<|im_start|>step24\nNow, sum the computed values:\n\\[ f(-2) + f(-1) + f(0) = 2 + \\dfrac{5}{3} + 1 \\]\n<|im_start|>step25\nConvert all numbers to fractions with a common denominator of 3.\n<|im_start|>step26\nExpress \\( 2 \\) and \\( 1 \\) as fractions over 3:\n\\[ 2 = \\dfrac{6}{3}, \\quad 1 = \\dfrac{3}{3} \\]\n<|im_start|>step27\nAdd the fractions:\n\\[ \\dfrac{6}{3} + \\dfrac{5}{3} + \\dfrac{3}{3} = \\dfrac{6 + 5 + 3}{3} \\]\n<|im_start|>step28\nCompute the numerator:\n\\[ 6 + 5 + 3 = 14 \\]\n<|im_start|>step29\nSo, the sum is:\n\\[ \\dfrac{14}{3} \\]\n<|im_start|>step30\nTherefore, \\( f(-2) + f(-1) + f(0) = \\dfrac{14}{3} \\).\n<|im_start|>step31\nAs a final check, ensure all calculations are correct and the answer is in simplest form.\n
                    <|im_start|>answer\nAnswer: \\[ f(-2) + f(-1) + f(0) = \\dfrac{14}{3} \\]
                output: 'Define\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$Step 1: We can rewrite\n\\begin{align*}\n\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3} &= \\sum_{k = 1}^\\infty \\sum_{j = 1}^\\infty \\frac{1}{(j + k)^3} \\\\ &= \\sum_{k = 1}^\\infty \\sum_{j = 1}^\\infty \\frac{1}{j^3} \\cdot \\frac{1}{(1 + \\frac{k}{j})^3} \\\\ &= \\sum_{j = 1}^\\infty \\frac{1}{j^3} \\sum_{k = 1}^\\infty \\frac{1}{(1 + \\frac{k}{j})^3}. ки\nStep 2: \\end{align*}We want to find an expression for this double sum. ки\n'
                '''

                # write code to process input into output
                raw_text = fork.text()
                fork.stream_executor.text_ = raw_text
                if "our_llama" in self.paras["policy_model_type"]:
                    processed_text = convert_text_to_reward_format_llama(raw_text)
                elif "our_qwen_steps_left" == self.paras["policy_model_type"]:
                    processed_text = convert_text_to_reward_format_qwen_steps_left(raw_text)
          
                elif "our_qwen" in self.paras["policy_model_type"]:
                    processed_text = convert_text_to_reward_format_qwen(raw_text)
           
                # bp()
                if left_steps-1 <= 0:
                    # bp()
                    raw_text = raw_text + "answer"
                # check if it goes to negative step:
                # elif ">0 steps left" in raw_text:
                #     # bp()
                #     raw_text = raw_text.split("0 steps left")[0]
                #     raw_text += "answer"
                #     processed_text = processed_text.split("0 steps left")[0]
                else:
                    raw_text += stop_token
                fork.stream_executor.text_ = processed_text
                fork.stream_executor.text_raw_ = raw_text
                fork.stream_executor.text_processed_ = processed_text

                
                fork += gen("score", max_tokens=0, forward_only=True, logits_require_id=8094)
                # bp()
                # print("fork.text(): ", fork.text().replace('\n', '\\n'))
                # print score
                print("fork.scores(): ", fork.scores())
                # put the raw text back to the stream executor
                fork.stream_executor.text_ = raw_text
            
                # print("fork.scores(): ", fork.scores())
            # TODO: swj change later to support our models
            self.running_list.append((fork, node))
            self.history_list.append(fork)
        # bp()
            
    
    def insert(self, state, parent):
        # print("state.scores() : ", state.scores())
        if state.scores() == [] or state.scores == None:
            return
        score = state.scores()[-1]
        num_step_tokens = state.get_meta_info("step")["completion_tokens"]
        new_node = TreeNode(self.size_, state, score, num_step_tokens, parent)
        self.size_ += 1
        depth = new_node.get_depth()
        self.depth_nodes[depth].append(new_node)
        self.nodes.append(new_node)
        return
    
    def select_softmax(self, node_list, node_weights, width):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        weights = []
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            weights.append(pair[1])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        for weight in exp_weights:
            num = int(math.ceil(width * weight / sum_exp_weights))
            select_num.append(num)
            width -= num
            sum_exp_weights -= weight
        return nodes, select_num
    
    def select_softmax_with_truncation(self, node_list, node_weights, width):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        weights = []
        truncate_ratio = self.paras["truncate_ratio"]
        keep_num = int(math.ceil(len(sorted_node_weight_pair_list) * truncate_ratio))
        sorted_node_weight_pair_list = sorted_node_weight_pair_list[:keep_num]
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            weights.append(pair[1])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        for weight in exp_weights:
            num = int(math.ceil(width * weight / sum_exp_weights))
            select_num.append(num)
            width -= num
            sum_exp_weights -= weight
        return nodes, select_num



    def select_and_expand(self, depth):
        cand_node_list = []
        cand_node_weights = []
        for node in self.depth_nodes[depth]:
            # bp()
            if node.is_leaf() == True or node.get_cum_tokens() >= self.paras["max_tokens"]:
                #print("node: ", node.get_text())
                # bp()
                #print("node.get_cum_tokens(): ", node.get_cum_tokens())
                #print("node.is_leaf(): ", node.is_leaf())
                self.remaining_width -= 1
            else:
                #print("else node: ", node.get_text())
                cand_node_list.append(node)
                cand_node_weights.append(node.get_score())
        # print("self.depth_nodes: ", self.depth_nodes)
        # print("cand_node_list: ", cand_node_list)
        # print("remaining_width: ", self.remaining_width)
        # bp()
        if self.remaining_width <= 0 or cand_node_list == []:
            return False
        if self.paras["select_method"] ==  "softmax":
            nodes, widths = self.select_softmax(cand_node_list, cand_node_weights, self.remaining_width)
        if self.paras["select_method"] == "softmax_with_truncate":
            nodes, widths = self.select_softmax_with_truncation(cand_node_list, cand_node_weights, self.remaining_width)
        # bp()
        #print("cand_node_list: ", cand_node_list)
        for expand_node, width in zip(nodes, widths):
            # print("expand_node: ", expand_node.get_text())
            # print("width: ", width)
            if width >= 1:
                self.expand(expand_node, width)
        return True 
             
@function
def reward_guided_search(s, id, question, paras, reward_host):
    answers = []
    i = 0
    while len(answers) == 0:
        s += question
        tree = Tree(s, paras, reward_host)
        depth = 0
        # print(f"loop: {i} step")
        # print("answers: ", answers)
        i += 1
        while True:
            tree.reset_running_list()
            continue_search = tree.select_and_expand(depth)
            # print("continue_search: ", continue_search)
            if not continue_search:
                break
            
            running_list = tree.get_running_list()
            # print("running_list: ", running_list)
            for state, parent in running_list:
                tree.insert(state, parent)
            
            depth += 1
            # print("depth: ", depth)
            if depth >= 50:
                break

        history_list = tree.get_history_list()
        total_tokens = sum(state.get_meta_info("step")["completion_tokens"] for state in history_list)

        all_nodes = tree.get_nodes()
        answers = []
        nodes_info = []
        
        for node in all_nodes:
            # print("id: ", id)
            parent_id = node.get_parent().get_id() if node.get_parent() else None
            nodes_info.append({
                "id": node.get_id(),
                "text": node.get_text(),
                "score": node.get_score(),
                "parent_id": parent_id,
                "depth": node.get_depth(),
                "leaf": node.is_leaf()
            })
            # bp()
            if node.is_leaf():
                # print(f"leaf node: ", node.get_text())
                step_scores = []
                last_node = node
                while last_node.get_depth() > 0:
                    step_scores.append(last_node.get_score())
                    last_node = last_node.get_parent()
                # bp()
                step_scores.reverse()
                answers.append({"text": node.get_text(), "step_scores": step_scores})
            # else:
                # print(f"non-leaf node: ", node.get_text())
    return {
        "id": id,
        "question": question,
        "model_answer": answers,
        "total_tokens": total_tokens
    }