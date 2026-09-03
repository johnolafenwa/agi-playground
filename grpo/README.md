# GRPO

GRPO training of `Qwen/Qwen3.5-2B` on small verifiable tasks, written from scratch on top of Hugging Face Transformers. No TRL, the whole algorithm is in one notebook so every step is readable.

- `train_gpu.ipynb`, the training loop: sample a group of completions per prompt, score them with a string-match reward, and run a clipped PPO update against the group-mean baseline. Logs to wandb, keeps the best-so-far checkpoint in `checkpoints/grpo`.
- `model_test_hf.ipynb`, compares the base and trained models, then plots accuracy against generation budget on GSM8K and on the held out code test set.
- `pyproject.toml`, pinned to the CUDA 12.9 torch build (the pod driver tops out at CUDA 12.8).

## Setup

```bash
uv sync --group dev
uv run python -m ipykernel install --user --name grpo --display-name "Python (grpo)"
```

The notebooks expect a `data/` directory beside them with the `*_train.jsonl` / `*_test.jsonl` task files (one JSON row per problem: chat `context`, `label.answer`, `metadata`). Generate them with the scripts in the `rl_presentation` repo.
