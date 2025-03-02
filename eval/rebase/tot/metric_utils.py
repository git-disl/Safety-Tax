import re
from ipdb import set_trace as bp
import os

def _aggregate_scores(step_scores, method="min"):
    """Aggregate step scores using different methods."""
    if not step_scores:
        return 0
    
    if method == "min":
        return min(score for score in step_scores if score is not None)
    elif method == "mean":
        valid_scores = [score for score in step_scores if score is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0
    elif method == "prod":
        prod = 1.0
        for score in step_scores:
            if score is not None:
                prod *= score
        return prod
    elif method == "last":
        return step_scores[-1] if step_scores and step_scores[-1] is not None else 0
    
    return 0

def _extract_shepherd_answer(completion):
    """Extract answer from completion text."""
    if completion is None:
        return "[invalid]"
    if os.getenv("BACKEND") == "llemma":
        match = re.search(r"The answer is:(.+) \u043a\u0438", completion)
    else:
        completion = completion.strip('<|reserved_special_token_2|>')
        match = re.search(r"Answer:(.+)$", completion)

    if match:
        match_str = match.group(1).strip()
        # match_str = match_str.replace(",", "")
        # for qwen
        if "<|im_end|>" in match_str:
            match_str = match_str.split("<|im_end|>")[0]
        return match_str
    return "[invalid]"

def _majority_vote(model_answers, weighted=False, weight_func="min"):
    """Implement majority voting among answers."""
    equiv_classes = []  # Unique answers
    equiv_weights = []  # Weights for each unique answer
    
    max_vote = 0
    max_rep = None
    
    for cand in model_answers:
        # bp()
        answer = _extract_shepherd_answer(cand["text"])
        # print("answer: ", answer)
        weight = 1
        if weighted:
            weight = _aggregate_scores(cand["step_scores"], method=weight_func)
            
        # Check if answer matches any existing equivalent class
        found = False
        for i, rep in enumerate(equiv_classes):
            if answer == rep:  # You might want to use grade_answer() here instead
                equiv_weights[i] += weight
                if equiv_weights[i] > max_vote:
                    max_vote = equiv_weights[i]
                    max_rep = answer
                found = True
                break
                
        if not found:
            equiv_classes.append(answer)
            equiv_weights.append(weight)
            if max_vote == 0:
                max_vote = weight
                max_rep = answer
                
    return max_rep

