<h1 align="center"> Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning</h1>

<div align="center">
<a href='https://huggingface.co/sastpg/Qwen2.5-3B-Instruct-CoVo'><img src='https://img.shields.io/badge/Hugging_Face-Models-%23FFD21E?style=flat&logo=huggingface&logoColor=%23FFD21E'></a>
<a href='https://huggingface.co/datasets/sastpg/CoVo_Dataset'><img src='https://img.shields.io/badge/Hugging_Face-Datasets-blue?style=flat&logo=huggingface&logoColor=%23FFD21E'></a>
</div>

> [!NOTE]
> Official codebase for the paper "Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning". The training code is based on the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) framework, and the evaluation code is based on the project [Math-Verify](https://github.com/huggingface/Math-Verify).

<div align="center">
<img src="./images/framework.png">
</div>


## Overview
We propose a novel self-rewarding Reinforcement Learning (RL) framework to enhance Large Language Model (LLM) reasoning by leveraging the consistency of intermediate reasoning states across different response trajectories. Our key insight is that correct responses often exhibit consistent trajectory patterns in terms of model likelihood: their intermediate reasoning states tend to converge toward their own final answers *high consistency* with minimal deviation toward other candidates *low volatility*. Inspired by this observation, we introduce **CoVo**, an intrinsic reward mechanism that integrates *<u>**Co**</u>nsistency* and *<u>**Vo**</u>latility* via a robust vector-space aggregation strategy, complemented by a curiosity bonus to promote diverse exploration. CoVo enables LLMs to perform reinforcement learning in a self-rewarding manner, offering a scalable pathway for learning to reason without external supervision. Extensive experiments on diverse reasoning benchmarks show that CoVo achieves performance comparable to or even surpassing supervised RL.

![](./images/vol.png)
> Normalized distance curve of correct and incorrect trajectories with varying state numbers.


## Acknowledgement
We thank the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for providing the awesome open-source RL infrastructure. We also thank the developers of [Qwen](https://github.com/QwenLM), [Llama](https://github.com/meta-llama) and [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) for their innovation and contribution to the open-source community.