
import numpy as np
import math
import scipy.stats
def compute_RiL(consistency, volatility):
    return np.mean(consistency) - np.mean(volatility)

def compute_RiV(consistency, volatility):
    x_list = np.array([consistency[i] * math.cos(volatility[i]) for i in range(len(volatility))])
    y_list = np.array([consistency[i] * math.sin(volatility[i]) for i in range(len(volatility))])
    al_combdiff_x_ave = np.mean(x_list)
    al_combdiff_y_ave = np.mean(y_list)

    return math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)

def kl_divergence(current_sentence):
    """
        Uniform KL penalty for curiosity rewards calculation
    """
    step_len = len(current_sentence)
    # print(step_len)
    # kl_uniform = np.random.uniform(0,1, size=(step_len))
    kl_uniform = []
    for _ in range(step_len):
        kl_uniform.append(float(1 / step_len))
    # print(kl_uniform)
    step_kl = np.log(np.power(scipy.stats.entropy(current_sentence, kl_uniform), 1/3)+1)
    # step_kl = scipy.stats.entropy(current_sentence, kl_uniform)

    return step_kl


def merge_colon_ended_elements(lst):
    result1 = []
    i = 0
    while i < len(lst):
        # 如果当前元素以冒号结尾
        if lst[i].endswith(':'):
            # 将当前元素与下一个元素合并
            if i + 1 < len(lst):  # 确保有下一个元素
                merged_element = lst[i] + "\n" + lst[i + 1]
                result1.append(merged_element)
                i += 2  # 跳过当前元素和下一个元素
            else:
                # 如果没有下一个元素，直接添加当前元素
                result1.append(lst[i])
                i += 1
        else:
            # 如果当前元素不以冒号结尾，直接添加到结果列表
            result1.append(lst[i])
            i += 1
    return result1

def merge_steps(lst):
    thred = len(lst) // 15
    result = []
    x = (thred + 1) * 15 - len(lst)
    y = len(lst) - 15 * thred
    split_list = [thred] * x + [thred+1] * y

    merge_idx = 0
    for item in split_list:
        result.append("\n".join(lst[merge_idx: merge_idx + item]))
        merge_idx += item
    return result

def process_thoughts(resp):
    thoughts = [line.strip() for line in resp.split('\n') if line.strip()]
    result = []
    merge_mode = False
    temp_merge = []

    for item in thoughts:
        if "\\[" in item and "\\]" not in item:
            merge_mode = True
            temp_merge.append(item)
        elif "\\]" in item and "\\[" not in item:
            merge_mode = False
            temp_merge.append(item)
            if temp_merge:
                result.append("\n".join(temp_merge))
                temp_merge = []
        elif merge_mode:
            temp_merge.append(item)
        else:
            result.append(item)
    
    if len(temp_merge) > 0:
        print(temp_merge)
        result += temp_merge
    
    if len(result) >= 15:
        result = merge_colon_ended_elements(result)
    
    if len(result) >= 15:
        result = merge_steps(result)

    return result

def detect_duplicate(text):
    duplicate_pattern = text[-50:]
    if len(text.split(duplicate_pattern)) > 10:
        return True
    return False