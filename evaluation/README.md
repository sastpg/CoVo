# Evaluation
This is the code for evaluating the performance of the models.

## Datasets
We provide the formated evaluation datasets in the `eval_dataset` directory:

| Dataset        | Domain            | Description                                                  |
| -------------- | ----------------- | ------------------------------------------------------------ |
| MATH-500       | Mathematics       | High school–level math problems covering 7 major areas such as precalculus, algebra, and number theory. |
| GSM8K          | Mathematics       | Elementary school math problems.                             |
| AMC-23         | Mathematics       | High school–level math competition problems.                 |
| Olympiad Bench | Mathematics       | Challenging math problems sourced from international Olympiads. |
| MMLU-Pro       | General knowledge | Natural sciences, social sciences, humanities, and some interdisciplinary content, etc. |
| GPQA           | Science           | Hard questions including biology, physics, and chemistry.    |
| CommonsenseQA  | Commonsense       | Commonsense questions involving various aspects such as daily life, social culture, and natural phenomena. |


## Requirements
- Python>=3.10
- math-verify[antlr4_13_2]
- vllm>=0.7.2
- torch>=2.5.1
- transformers>=4.48.3

You can install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

## Usage
To evaluate the performance of the models, you can run the following command:
```bash
python eval.py --model <local_model_path> --dataset <dataset_name>
# e.g. python eval.py --model /path/to/local/model --dataset ./eval_data/math500.jsonl
```

