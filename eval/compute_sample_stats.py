"""Compute avg thinking steps"""

import sys
import json

path = sys.argv[1] # e.g. qwen_20241129-012159-32steps_32kctxt/ckpts__qwen_20241129-012159/samples_openai_math_2024-12-01T01-56-47.156277.jsonl
try:
    # allowed_steps = int(path.split('steps')[0].split('-')[-1].split('_')[-1])
    allowed_steps = int(path.split('step')[-1].split('/')[0].split('forcing')[0])
    allowed_tokens = None
except:
    try:
        allowed_tokens = int(path.split('tokens')[0].split('-')[-1].split('_')[-1])
        allowed_steps = None
    except:
        allowed_steps, allowed_tokens = None, None
        print("No steps/tokens in path; Assuming it was run without steps/tokens limit")

tokens = True
total_steps = 0
step_lens = []
total_lens = []
total_lens_with_sep = []
total_lens_answer = []
samples = 0
samples_too_many_steps = 0
samples_too_many_tokens = 0
samples_too_many_thinking_tokens = 0
samples_without_answer = 0
#samples_without_answer_but_steps = 0
samples_without_thinking = 0
correct = []

if tokens:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

with open(path) as f: # JSONL file
    for line in f:
        data = json.loads(line)

        if isinstance(data['filtered_resps'][0], list):
            data['filtered_resps'][0] = data['filtered_resps'][0][0]

        ### <|im_start|>step format ###
        # # total_steps += data['filtered_resps'][0].count('<|reserved_special_token_2|>')
        # total_steps += data['filtered_resps'][0].count('<|im_start|>step')
        # samples_without_answer += int('<|im_start|>answer' not in data['filtered_resps'][0])
        # #samples_without_answer_but_steps += int(('<|im_start|>answer' not in data['filtered_resps'][0]) and ('<|im_start|>step' in data['filtered_resps'][0]))
        # samples_without_thinking += int('<|im_start|>step' not in data['filtered_resps'][0])
        # samples_too_many_steps += int(data['filtered_resps'][0].count('<|im_start|>step') > allowed_steps)
        # step_lens.extend([len("\n".join(step.split("\n")[1:])) for step in data['filtered_resps'][0].split('<|im_start|>step')[1:-1]])
        # samples += 1

        if tokens and ("qwq" in path):
            total_lens_with_sep.append(len(tok.tokenize(data['filtered_resps'][0])))

            too_long = int(len(tok.tokenize(data['arguments']['gen_args_0']['arg_0'] + data['filtered_resps'][0])) > 32760)
            samples_too_many_tokens += too_long

            no_answer = int(('Answer:' not in data['filtered_resps'][0]) and ('\\boxed' not in data['filtered_resps'][0]))
            samples_without_answer += no_answer
        elif tokens and allowed_steps:
            total_steps += data['filtered_resps'][0].count(' steps left\n')
            samples_without_thinking += int(' steps left\n' not in data['filtered_resps'][0])
            samples_too_many_steps += int(data['filtered_resps'][0].count(' steps left\n') > allowed_steps)
            # if int(data['filtered_resps'][0].count(' steps left\n') > allowed_steps):
            #     import pdb; pdb.set_trace()
            step_lens_tmp = [len(tok.tokenize("\n".join(step.split("\n")[1:]))) for step in data['filtered_resps'][0].split('<|im_start|>')[1:-1]]
            total_lens.extend([sum(step_lens_tmp)])
            total_lens_with_sep.extend([len(tok.tokenize(data['filtered_resps'][0]))])
            if "<|im_start|>answer" in data['filtered_resps'][0]:
                total_lens_answer.append(len(tok.tokenize(data['filtered_resps'][0].split('<|im_start|>answer')[1])))
            else:
                total_lens_answer.append(0)
            step_lens.extend(step_lens_tmp)

            #import pdb; pdb.set_trace()
            too_long = int(len(tok.tokenize(data['arguments']['gen_args_0']['arg_0'] + data['filtered_resps'][0])) > 32760) # 32768
            samples_too_many_tokens += too_long

            no_answer = int(('Answer:' not in data['filtered_resps'][0]) and ('\\boxed' not in data['filtered_resps'][0]))
            samples_without_answer += no_answer

            # Worth checking
            #if no_answer and not(too_long):
            #    import pdb; pdb.set_trace()
        elif tokens:
            steps = data['filtered_resps'][0].split('<|im_start|>answer')[0].split("\n")[1:-1]
            thinking = "\n".join(steps)
            if '<|im_start|>answer' in data['filtered_resps'][0]:
                answer = data['filtered_resps'][0].split('<|im_start|>answer')[-1]
                if "\n" in answer:
                    answer = "\n".join(answer.split("\n")[1:])
            else:
                answer = ""
            
            total_steps += data['filtered_resps'][0].split('<|im_start|>answer')[0].count('\n') - 1
            samples_without_thinking += int('<|im_start|>think' not in data['filtered_resps'][0])
            
            step_lens_tmp = [len(tok.tokenize(step)) for step in steps]
            step_lens.extend(step_lens_tmp)

            thinking_tokens = len(tok.tokenize(thinking))
            total_lens.append(thinking_tokens)

            #if sum(step_lens_tmp) == 30119:
            #   import pdb; pdb.set_trace()

            if allowed_tokens:
                samples_too_many_thinking_tokens += int(thinking_tokens > allowed_tokens)
        
            total_lens_with_sep.append(len(tok.tokenize(data['filtered_resps'][0])))
            total_lens_answer.append(len(tok.tokenize(answer)))

            too_long = int(len(tok.tokenize(data['arguments']['gen_args_0']['arg_0'] + data['filtered_resps'][0])) > 32760)
            if too_long:
                import pdb; pdb.set_trace()
            samples_too_many_tokens += too_long

            no_answer = int(('Answer:' not in data['filtered_resps'][0]) and ('\\boxed' not in data['filtered_resps'][0]))
            samples_without_answer += no_answer

            # Worth checking
            #if not(no_answer) and too_long:
            #    import pdb; pdb.set_trace()
            #if no_answer and not(too_long):
            #    import pdb; pdb.set_trace()
            # if no_answer:
            #    import pdb; pdb.set_trace()

        ### <|im_start|>13 steps left format ###
        else:
            if allowed_steps:
                total_steps += data['filtered_resps'][0].count(' steps left\n')
                samples_without_thinking += int(' steps left\n' not in data['filtered_resps'][0])
                samples_too_many_steps += int(data['filtered_resps'][0].count(' steps left\n') > allowed_steps)
                step_lens_tmp = [len("\n".join(step.split("\n")[1:])) for step in data['filtered_resps'][0].split('<|im_start|>')[1:-1]]
                total_lens.extend([sum(step_lens_tmp)])
                step_lens.extend(step_lens_tmp)
            else:
                # Approximation
                total_steps += data['filtered_resps'][0].split('<|im_start|>answering\n')[0].count('\n') - 1
                samples_without_thinking += int('<|im_start|>thinking\n' not in data['filtered_resps'][0])
                step_lens.extend([len("\n".join(step.split("\n")[1:])) for step in data['filtered_resps'][0].split('<|im_start|>answering\n')[0].split('\n')[1:-1]])
                total_lens.extend([len(data['filtered_resps'][0].split('<|im_start|>answering\n')[0])])

            samples_without_answer += int('<|im_start|>answer' not in data['filtered_resps'][0])
            #samples_without_answer_but_steps += int(('<|im_start|>answer' not in data['filtered_resps'][0]) and ('<|im_start|>step' in data['filtered_resps'][0]))

        samples += 1
        correct.append(data['exact_match'])

print("# samples:", samples)
if tokens:
    print("avg steps:", round(total_steps / samples, 1))
    if step_lens:
        print("avg step tokens:", round(sum(step_lens) / len(step_lens), 1))
    if total_lens:
        print("min total thinking tokens:", min(total_lens))
        print("max total thinking tokens:", max(total_lens))
        print("sorted total thinking tokens:", sorted(total_lens))
        correct_sorted = [correct[i] for i in sorted(range(len(total_lens)), key=lambda k: total_lens[k])]
        print("sorted correct:", correct_sorted)
        print("avg total thinking tokens:", round(sum(total_lens) / len(total_lens), 1))
    if total_lens_answer:
        print("avg total answer tokens:", round(sum(total_lens_answer) / len(total_lens_answer), 1))
    print("avg total tokens:", round(sum(total_lens_with_sep) / len(total_lens_with_sep), 1))
    if allowed_steps:
        print("samples w/ too many steps:", samples_too_many_steps)
    else:
        print("samples w/ too many thinking tokens:", samples_too_many_thinking_tokens)
    print("samples w/ too many tokens:", samples_too_many_tokens)
    print("samples w/o answer:", samples_without_answer)
    print("samples w/o thinking:", samples_without_thinking)
elif allowed_steps:
    print("avg steps:", round(total_steps / samples, 1))
    print("avg step len:", round(sum(step_lens) / len(step_lens), 1))
    print("avg total thinking len:", round(sum(total_lens) / len(total_lens), 1))
    print("samples w/ too many steps:", samples_too_many_steps)
    print("samples w/o 'Answer:':", samples_without_answer)
    #print("samples w/o answer but steps:", samples_without_answer_but_steps)
    print("samples w/o thinking:", samples_without_thinking)
else:
    print("avg steps (approx via \\n):", round(total_steps / samples, 1))
    print("avg step len (approx via \\n):", round(sum(step_lens) / len(step_lens), 1))
    print("avg total thinking len:", round(sum(total_lens) / len(total_lens), 1))
    print("samples w/o answer:", samples_without_answer)
    print("samples w/o thinking:", samples_without_thinking)
