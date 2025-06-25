import json
from utils import check
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

prompt_template = {
    "math": "Question: {}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "mmlu": "Question: {}\nAnswer the multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the choices. Think step by step before answering.\n",
    "gpqa": "Question: {}\nAnswer the multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n",
    "commonsenseqa": "Question: {}\nAnswer the multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step before answering.\n"
}
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip out whitespace and check if the line is not empty
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_choice(resp):
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt in opts:
        if f"Answer: {opt}" in resp or f"Answer:{opt}" in resp or f"Answer: **{opt}" in resp or f"Answer: ({opt}" in resp:
            return opt
    return None

def eval_acc(responses, labels, is_math):
    scores = []
    if is_math:
        for response, label in zip(responses, labels):
            scores.append(check(label, response))
    else:
        for response, label in zip(responses, labels):
            scores.append(label==extract_choice(response))
    acc = {sum(scores) / len(scores)}
    print(f"Average performance: {acc:.4f}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Evaluation model path")
    parser.add_argument("--dataset", type=str, default=None, help="Evaluation dataset")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max generation tokens")
    args = parser.parse_args()

    model_path = args.model
    eval_dataset = args.dataset
    is_math = False
    if "math" in eval_dataset or "gsm8k" in eval_dataset or "amc" in eval_dataset or "olympiad" in eval_dataset:
        is_math = True
    template = prompt_template["math"] if is_math else prompt_template["mmlu"]

    # Load the model
    llm = LLM(model=model_path, enable_chunked_prefill=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Sampling parameters
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    dataset = read_jsonl(eval_dataset)

    # prompts = [prompt_template.format(item["problem"]) for item in dataset]
    prompts = [template.format(item["problem"]) for item in dataset]
    labels = [item["answer"] for item in dataset]

    # Batch inference
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [item.outputs[0].text for item in outputs]
    print(f"Response [1/{len(dataset)}]: {responses[0]}\n")

    # Accuracy
    eval_acc(responses, labels, is_math)
