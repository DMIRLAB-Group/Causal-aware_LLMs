This repository is the official implementation of the IJCAI 2025 paper:  
**Causal-aware Large Language Models: Enhancing Decision-Making through Learning, Adapting and Acting**


##  Overview

**Causal-aware LLMs** is a novel framework that integrates the structural causal model (SCM) into the decision-making process to model, update, and utilize structured knowledge of the environment in a “learning-adapting-acting” paradigm.

![causal-aware_LLMs](./img/framework.png)


##  Quick Start

### 1. Requirements

- **GPU Resource**: At least one NVIDIA GPU with **24GB** memory is recommended for running LLaMA3-8B and PPO training stably.
- **Python Environment**:
  - Python 3.10+
  - Install dependencies via:

```bash
pip install -r requirements.txt
```

### 2. LLMs

This project supports various Large Language Models (LLMs) for causal knowledge extraction and policy guidance.

- If you prefer to use an API-based LLM (e.g., OpenAI, DeepSeek), set use_api to true and provide your API key:

```yaml
env_spec:
  use_local_llm: false
  api_key: your_api_key_here
```
- If you are using a local LLM, set the `local_lm_path` in `config.yaml` to the path of your local model:

```yaml
llm:
  use_local_llm: True
  local_lm_path: /path/to/your/local/llm
```

### 3. Sentence-BERT

In order to transform environment information into textual embeddings for policy conditioning, we use **Sentence-BERT (sBERT)**.

- Download a pre-trained sBERT model from Hugging Face.

We recommend using `paraphrase-MiniLM-L3-v2` for a good balance between speed and performance.


### 4. Hyperparameters

You can modify the training-related hyperparameters in the `ppo.yaml` file.

### 5. Weights & Biases (WandB)

We use **Weights & Biases (WandB)** to log and visualize training metrics.

- First, make sure you have a **WandB account**. You can sign up [here](https://wandb.ai/signup) if you don't have one.
- Install WandB using pip:
```bash
pip install wandb
```
- Log in to your WandB account and follow the instructions in the terminal to authenticate your account by running:
```bash
wandb login  
```

### 6. Running

To start training, simply run the following command:

```bash
python train.py
```
### 7. Result


| Method Type              | Method                        | Score (%)           |
|--------------------------|-------------------------------|---------------------|
| **Ours**                 | Causal-aware LLMs (@1M)       | **18.9 ± 0.53**     |
|                          | Causal-aware LLMs (@5M)       | **33.6 ± 0.02**     |
| **RL-based methods**     | Rainbow (@1M)                 | 4.3 ± 0.2           |
|                          | DreamerV2 (@1M)               | 10.0 ± 1.2          |
|                          | DreamerV3 (@1M)               | 14.77 ± 1.42        |
|                          | PPO (ResNet) (@1M)            | 15.6 ± 1.66         |
| **LLM-based methods**    | ReAct (GPT-4) (@1M)           | 8.3 ± 1.2           |
|                          | Reflexion (GPT-4) (@1M)       | 11.7 ± 1.4          |
|                          | AdaRefiner (@1M)              | 15.8 ± 1.4          |
|                          | AdaRefiner (@5M)              | 28.2 ± 1.8          |
| **Additional references**| Human Experts                 | 50.5 ± 6.8          |
|                          | SPRING (+prior) (@1M)         | 27.3 ± 1.2          |
|                          | Random (@1M)                  | 1.6 ± 0.0           |


### 8. 📜 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{chen2025causalllm,
  title={Causal-aware Large Language Models: Enhancing Decision-Making through Learning, Adapting and Acting},
  author={Wei Chen, Jiahao Zhang, Haipeng Zhu, Boyan Xu, Zhifeng Hao, Keli Zhang, Junjian Ye, Ruichu Cai},
  booktitle={Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```
