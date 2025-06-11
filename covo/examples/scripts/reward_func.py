import torch
import json
from math_verify import parse, verify

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def reward_func(queries, prompts, labels):
    rewards = []
    sc_rewards = []
    expression_count = {}

    for query, prompt, label in zip(queries, prompts, labels):
        # query = query.replace("\\boxed{<ANSWER>}", "")
        wrap_label = "\\boxed{" + label + "}" if "\\boxed" not in label else label
        model_boxed_answer = last_boxed_only_string(query)
        if prompt not in expression_count:
            expression_count[prompt] = {}

        # 统计答案出现次数, key 是 "\\boxed{xxx}"
        if model_boxed_answer:
            found = False
            for expr in expression_count[prompt].keys():
                if is_equiv(remove_boxed(model_boxed_answer), remove_boxed(expr)):
                    expression_count[prompt][expr] += 1
                    found = True
                    break
                else:
                    if verify(parse(expr), parse(model_boxed_answer)):
                        expression_count[prompt][expr] += 1
                        found = True
                        break
            if not found:
                expression_count[prompt][model_boxed_answer] = 1

        # true rewards
        if model_boxed_answer:
            if is_equiv(remove_boxed(model_boxed_answer), remove_boxed(last_boxed_only_string(wrap_label))):
                rewards.append(1.0)
            else:
                _gold = parse(wrap_label)
                _answer = parse(model_boxed_answer)
                if verify(_gold, _answer):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
        else:
            rewards.append(0.0)
    

    sc_answer = {}
    different_prompt_number = 0
    for p in expression_count.keys():
        max_count = 0
        sc_answer[p] = "None of"
        different_prompt_number += 1
        for expr, count in expression_count[p].items():
            if count > max_count:
                max_count = count
                sc_answer[p] = expr
    
    assert different_prompt_number == 1
    print(sc_answer)
    for query, prompt, label in zip(queries, prompts, labels):
        # query = query.replace("\\boxed{<ANSWER>}", "")
        wrap_label = sc_answer[prompt]
        model_boxed_answer = last_boxed_only_string(query)
        # self-consistency 答案
        if model_boxed_answer:
            if is_equiv(remove_boxed(model_boxed_answer), remove_boxed(wrap_label)):
                sc_rewards.append(1.0)
            else:
                _gold = parse(wrap_label)
                _answer = parse(model_boxed_answer)
                if verify(_gold, _answer):
                    sc_rewards.append(1.0)
                else:
                    sc_rewards.append(0.0)
        else:
            sc_rewards.append(0.0)

    return {"real_rewards": torch.tensor(rewards), "sc_rewards": torch.tensor(sc_rewards)}


# if __name__ == "__main__":
#     prompts = ["Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pairings are possible?  One such pairing is a blue bowl and a yellow glass."] * 8 + ["The distance between two cities on a map is 15 inches. If the scale is 0.25 inches = 3 miles, how many miles apart are the actual cities?"] * 8
#     answer = ["xxx there are $4 \\times 4 = \\boxed{16}$ possible pairings."] * 2 + ["xxx there are $4 \\times 4 = \\boxed{2}$ possible pairings."] * 6 + ["xxx there are $\\boxed{2}$ prime numbers between 20 and 30: 23 and 29."] *1 + ["xxx there are $\\boxed{16}$ prime numbers between 20 and 30: 23 and 29."] *7
#     label = ["16"] * 8 + ["2"] * 8
#     r = reward_func(answer, prompts, label)

#     print(r)

if __name__ == "__main__":
    prompts = ["Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pairings are possible?  One such pairing is a blue bowl and a yellow glass."] * 16
    answer = ['system\n\nCutting Knowledge Date: December 2023\nToday Date: 07 Apr 2025\n\nuser\n\nYou are a mathematics expert that solve problems step by step.\n\nNote that you should output your final answer with the format of "The answer is: \\boxed{<ANSWER>}.", where <ANSWER> should be a numeric result or a math expression.\n\nQuestion: Find the sum of the first 10 elements that appear both in the arithmetic progression $\\{4, 7, 10, 13, \\ldots\\}$ and the geometric progression $\\{10, 20, 40, 80, \\ldots\\}$. (10 points)\nLet\'s think step by step.assistant\n\nTo find the first 10 elements that appear in both the arithmetic progression and the geometric progression, we need to find the common terms.\n\nThe first term of the arithmetic progression (AP) is 4 and the common difference is 3 (since 7 - 4 = 3). The first term of the geometric progression (GP) is 10 with a common ratio of 2 (since 20 / 10 = 2).\n\nNow, we can write the nth term of AP and GP.\n\nArithmetic progression (AP): the nth term an = a1 + (n - 1)d, where a1 is the first term and d is the common difference.\nan = 4 + (n - 1)*3\nan = 3n + 1\n\nGeometric progression (GP): the nth term an = a1 * r^(n - 1), where a1 is the first term and r is the common ratio.\nan = 10 * 2^(n - 1)\n\nNow, we need to find the first 10 terms of both sequences to identify the common terms. We can solve for n where we set the AP and GP formulas equal:\n\n3n + 1 = 10 * 2^(n - 1)\n\nLet\'s first list the first few terms and see if that helps identify the pattern.\n\nArithmetic: 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,...\nGeometric: 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240,...\n\nFrom these sequences, we can identify common numbers:\n- 10\n- 40\n- 160\n- 640\n- 2560\n- 10240\n\nThese are the six numbers that appear in both sequences in the limit as n tends to infinity (n = 0 has no answer values in these sequences).\n\nWe now calculate the sum of the first 6 common values.\n\nSum = 10 + 40 + 160 + 640 + 2560 + 10240 = \n\nSum = 10 + 40 + 160 = 210\n210 + 640 = 850\n850 + 2560 = 3410\n3410 + 10240 = 13450 \n\nTherefore, \n\nThe answer is: 13450'] * 16
    label = ["16"] * 16
    r = reward_func(answer, prompts, label)

    print(r)