## Setup

```
conda create -p ./penv python=3.11 -y
conda activate ./penv
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install vllm==0.8.5  # required for Mistral support; overrides 0.7.2 from requirements.txt
```

## Sample command

```
python src/inference.py \
    --solver_model_name Qwen/Qwen3-0.6B \
    --verifier_model_name Qwen/Qwen3-0.6B \
    --dataset_name gsm \
    --dataset_subset_ratio 0.01 \
    --output_dir result
```

## Custom Problem Generators

2/3SAT, 4x4 or 9x9 Sudoku, and variable sized square matrix multiplication datasets can be generated on the fly. In `--data_generation_kwargs`, you should always provide every CLI argument in the data generation script (except `output_path`) in the order they appear in the data generation script for the best caching behavior. Examples:
```
# SAT
python src/inference.py \
    --solver_model_name Qwen/Qwen3-0.6B \
    --verifier_model_name Qwen/Qwen3-0.6B \
    --dataset_name sat \
    --dataset_subset_ratio 0.1 \
    --data_generation_kwargs "sat_type=2,num_samples=100,min_vars=2,max_vars=8,min_clauses=2,max_clauses=8,seed=42" \
    --output_dir result

# Sudoku
python src/inference.py \
    --solver_model_name Qwen/Qwen3-0.6B \
    --verifier_model_name Qwen/Qwen3-0.6B \
    --dataset_name sudoku \
    --dataset_subset_ratio 0.1 \
    --data_generation_kwargs "size=4,num_samples=100,min_empty=4,max_empty=8,seed=42" \
    --output_dir result

# Matmul
python src/inference.py \
    --solver_model_name Qwen/Qwen3-0.6B \
    --verifier_model_name Qwen/Qwen3-0.6B \
    --dataset_name matmul \
    --dataset_subset_ratio 0.1 \
    --data_generation_kwargs "size=5,num_samples=100,min_val=-5,max_val=5,seed=42" \
    --output_dir result
```

## Cache System

Solver outputs are SHA256-keyed and cached automatically; no extra CLI arguments are needed. Loading and saving can each be disabled when desired via `--no_load_solver_cache` and `--no_save_solver_cache`. Whenever you add a dataset or change any argument that affects solver outputs, run a `--no_verify` pass first to populate the cache, then all subsequent verification runs execute quickly against the cached solver outputs.