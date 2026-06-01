from pprint import pprint
from typing import List, Dict, Tuple
import math
import argparse
import json
import gc
import os
from functools import partial

import torch
torch._inductor.config.autotune_local_cache = False
from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams

from solver_cache import SolverCacheManager
from dataset_processors import *
from answer_extractors import *
from oracle_verifiers import *
from dataset_validators import *


# model num params taken from HF, in billions
MODEL_SIZES = {
    # Qwen3 pretrained + post-trained
    'Qwen/Qwen3-0.6B': 0.6,
    'Qwen/Qwen3-1.7B': 1.7,
    'Qwen/Qwen3-4B': 4.0,
    'Qwen/Qwen3-8B': 8.2,
    'Qwen/Qwen3-14B': 14.8,
    'Qwen/Qwen3-32B': 32.8,
    'Qwen/Qwen3-30B-A3B': 3.3, # MoE
    # Qwen3 pretrained
    'Qwen/Qwen3-0.6B-Base': 0.6,
    'Qwen/Qwen3-1.7B-Base': 1.7,
    'Qwen/Qwen3-4B-Base': 4.0,
    'Qwen/Qwen3-8B-Base': 8.2,
    'Qwen/Qwen3-14B-Base': 14.8,
    'Qwen/Qwen3-30B-A3B': 3.3, # MoE
    # Qwen2.5 pretrained + post-trained
    'Qwen/Qwen2.5-0.5B-Instruct': 0.5,
    'Qwen/Qwen2.5-1.5B-Instruct': 1.5,
    'Qwen/Qwen2.5-3B-Instruct': 3.1,
    'Qwen/Qwen2.5-7B-Instruct': 7.6,
    'Qwen/Qwen2.5-14B-Instruct': 14.8,
    'Qwen/Qwen2.5-32B-Instruct': 32.8,
    'Qwen/Qwen2.5-72B-Instruct': 72.7,
    # Qwen2.5 pretrained
    'Qwen/Qwen2.5-0.5B': 0.5,
    'Qwen/Qwen2.5-1.5B': 1.5,
    'Qwen/Qwen2.5-3B': 3.1,
    'Qwen/Qwen2.5-7B': 7.6,
    'Qwen/Qwen2.5-14B': 14.8,
    'Qwen/Qwen2.5-32B': 32.8,
    'Qwen/Qwen2.5-72B': 72.7,
    # llama3 pretrained + instruction-tuned
    "meta-llama/Llama-3.2-1B-Instruct": 1.23,
    "meta-llama/Llama-3.2-3B-Instruct": 3.21,
    "meta-llama/Llama-3.1-8B-Instruct": 8.03,
    "meta-llama/Llama-3.1-70B-Instruct": 70.6,
    # llama3 pretrained
    "meta-llama/Llama-3.2-1B": 1.23,
    "meta-llama/Llama-3.2-3B": 3.21,
    "meta-llama/Llama-3.1-8B": 8.03,
    "meta-llama/Llama-3.1-70B": 70.6,
    # mistral pretrained + instruction-tuned + post-trained
    "mistralai/Magistral-Small-2507": 24.0,
    # mistral pretrained + instruction-tuned
    "mistralai/Mistral-7B-Instruct-v0.3": 7.25,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 24.0,
    # mistral pretrained
    "mistralai/Mistral-7B-v0.3": 7.25,
    "mistralai/Mistral-Small-3.1-24B-Base-2503": 24.0,
    # deepseekR1 distilled reasoning models
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 1.54,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 7.62,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 14.8,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 32.8,
}

DATA_SPECIFIC_FUNCTIONS = {
    "gsm": {
        'process_data': process_gsm,
        'extract_answer': extract_float_answer,
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "aime": {
        'process_data': process_aime,
        'extract_answer': extract_float_answer,
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "mmlu_all": {
        'process_data': partial(process_mmlu, "all"),
        'extract_answer': extract_float_answer, # choices are just 0, 1, 2, 3
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "mmlu_stem": {
        'process_data': partial(process_mmlu, "stem"),
        'extract_answer': extract_float_answer, # choices are just 0, 1, 2, 3
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "mmlu_social_sciences": {
        'process_data': partial(process_mmlu, "social_sciences"),
        'extract_answer': extract_float_answer, # choices are just 0, 1, 2, 3
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "mmlu_humanities": {
        'process_data': partial(process_mmlu, "humanities"),
        'extract_answer': extract_float_answer, # choices are just 0, 1, 2, 3
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "mmlu_other": {
        'process_data': partial(process_mmlu, "other"),
        'extract_answer': extract_float_answer, # choices are just 0, 1, 2, 3
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "csqa": {
        'process_data': process_csqa,
        'extract_answer': extract_float_answer, # choices are just 0, 1, 2, 3
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "gpqa": {
        'process_data': process_gpqa,
        'extract_answer': extract_float_answer, # choices are just 0, 1, 2, 3
        'oracle_verifier': exact_match,
        'dataset_validator': trivial_validator,
    },
    "sat": {
        'process_data': lambda output_dir, data_generation_kwargs: process_generated_data('sat', output_dir, data_generation_kwargs),
        'extract_answer': extract_sat_answer,
        'oracle_verifier': sat_is_correct, # sat can have multiple correct answers
        'dataset_validator': validate_sat_dataset,
    },
    "sudoku": {
        'process_data': lambda output_dir, data_generation_kwargs: process_generated_data('sudoku', output_dir, data_generation_kwargs),
        'extract_answer': extract_sudoku_answer,
        'oracle_verifier': sudoku_is_correct,
        'dataset_validator': validate_sudoku_dataset,
    },
    "matmul": {
        'process_data': lambda output_dir, data_generation_kwargs: process_generated_data('matmul', output_dir, data_generation_kwargs),
        'extract_answer': extract_matmul_answer,
        'oracle_verifier': matmul_is_correct,
        'dataset_validator': validate_matmul_dataset,
    },
}


def process_prompts(
    tokenizer,
    messages_batch: List[List[Dict[str, str]]]
) -> List[str]:
    """Apply chat template"""

    processed = []
    for messages in messages_batch:
        assert len(messages) == 1 # single turn convo

        # Check if tokenizer has a chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # qwen3 has an argument enable_thinking that defaults to true
            )
        else:
            # Fallback for models without chat templates (like base llama1b)
            text = messages[0]['content']
        processed.append(text)
    return processed


def inference(
    model,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    top_k: int,
    top_p: int,
    n_samples: int,
    seed: int,
) -> Tuple[List[str], int]:
    """Burn GPUs"""

    # sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        n=n_samples,
        seed=seed,
    )
    # generate
    outputs = model.generate(prompts, sampling_params)

    # Collect results
    total_tokens = 0
    result = []
    for output in outputs:
        assert len(output.outputs) == n_samples
        for o in output.outputs:
            result.append(o.text)
            total_tokens += len(o.token_ids)

    assert len(result) == len(prompts) * n_samples
    return result, total_tokens


def parse_args():
    parser = argparse.ArgumentParser()

    # model and prompts
    parser.add_argument("--solver_model_name", type=str, required=True, help="Model for initial solving")
    parser.add_argument("--verifier_model_name", type=str, required=True, help="Model for verification")
    parser.add_argument("--prompt_dir", type=str, default="prompts",
                        help="Prompt directory, should contain inference_prompt.md and verification_prompt.md")

    # dataset
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_subset_ratio", type=float, default=1.0)
    parser.add_argument("--save_subset_ratio", type=float, default=0.01) # full datasets can be big

    # vllm server initialization
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # solver vllm sampling params (vllm native default values, except max_new_tokens)
    parser.add_argument("--solver_max_new_tokens", type=int, default=8192)
    parser.add_argument("--solver_temperature", type=float, default=0.7) # some models do not recommend low T
    parser.add_argument("--solver_top_k", type=int, default=-1)
    parser.add_argument("--solver_top_p", type=float, default=0.9) # generally helps
    parser.add_argument("--solver_n_samples", type=int, default=1)

    # verifier vllm sampling params
    parser.add_argument("--verifier_max_new_tokens", type=int, default=None)
    parser.add_argument("--verifier_temperature", type=float, default=None)
    parser.add_argument("--verifier_top_k", type=int, default=None)
    parser.add_argument("--verifier_top_p", type=float, default=None)

    # dataset generation configurations
    parser.add_argument("--data_generation_kwargs", type=str, default="", help="""
                        Generation kwargs as comma-separated key=value pairs.
                        Examples: 'sat_type=2,num_samples=1000,min_vars=3' for SAT,
                                  'size=4,num_samples=500,min_empty=6' for Sudoku,
                                  'size=6,num_samples=300,min_val=-5,max_val=5' for Matrix Multiplication""")

    # miscellaneous
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=8) # not used yet
    parser.add_argument("--include_bad_solves", action='store_true', help='include bad solves for verifier')
    parser.add_argument("--no_verify", action='store_true', help='not verify')
    parser.add_argument("--oracle_solver", action='store_true', help='no solver model, just use groundtruth')

    # solver caching
    parser.add_argument("--solver_cache_root", type=str, default="solver_cache", help="Root directory for solver cache")
    parser.add_argument("--no_load_solver_cache", action='store_true', help='Disable loading from solver cache')
    parser.add_argument("--no_save_solver_cache", action='store_true', help='Disable saving to solver cache')

    args = parser.parse_args()

    # verifier sampling params default to solver's
    if args.verifier_max_new_tokens is None:
        args.verifier_max_new_tokens = args.solver_max_new_tokens
    if args.verifier_temperature is None:
        args.verifier_temperature = args.solver_temperature
    if args.verifier_top_k is None:
        args.verifier_top_k = args.solver_top_k
    if args.verifier_top_p is None:
        args.verifier_top_p = args.solver_top_p

    assert args.solver_model_name in MODEL_SIZES and args.verifier_model_name in MODEL_SIZES

    if args.solver_n_samples > 1:
        assert args.solver_temperature > 0.0

    return args


if __name__ == "__main__":
    ####################################### SETUP #######################################
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.system(f'chmod -fR 777 {args.output_dir}')

    print('==============================================================================')
    os.system('nvidia-smi')
    pprint(vars(args))
    print('==============================================================================')

    # dataset functions
    process_data_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['process_data']
    extract_answer_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['extract_answer']
    oracle_verifier_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['oracle_verifier']
    dataset_validator_fn = DATA_SPECIFIC_FUNCTIONS[args.dataset_name]['dataset_validator']

    # preprocess and subset dataset (all absolutely deterministic, double checked)
    if args.dataset_name in ["sat", "sudoku", "matmul"]:
        # special generated dataset that requires on-the-fly generation, using output_dir as temp dir
        dataset = process_data_fn(args.output_dir, args.data_generation_kwargs)
    else:
        dataset = process_data_fn()
    dataset = dataset.shuffle(seed=args.seed)
    assert isinstance(dataset, Dataset) and set(['question', 'answer']).issubset(set(dataset.column_names))
    assert dataset_validator_fn(dataset) # validate dataset, especially for our custom ones like 3sat
    if args.dataset_subset_ratio < 1.0:
        subset_size = math.ceil(len(dataset) * args.dataset_subset_ratio)
        dataset = dataset.select(range(subset_size))
    assert len(dataset) > 0
    print(f"Dataset size: {len(dataset)}")

    # prompts
    with open(f"{args.prompt_dir}/inference_prompt.md", "r") as f:
        inference_prompt = f.read()
    with open(f"{args.prompt_dir}/verification_prompt.md", "r") as f:
        verification_prompt = f.read()

    solver_cache = SolverCacheManager(cache_root=args.solver_cache_root)
    cache_params = {
        'solver_model_name': args.solver_model_name,
        'dataset_name': args.dataset_name,
        'dataset_subset_ratio': args.dataset_subset_ratio,
        'data_generation_kwargs': args.data_generation_kwargs,
        'solver_temperature': args.solver_temperature,
        'solver_max_new_tokens': args.solver_max_new_tokens,
        'solver_top_k': args.solver_top_k,
        'solver_top_p': args.solver_top_p,
        'solver_n_samples': args.solver_n_samples,
        'seed': args.seed,
        'inference_prompt': inference_prompt,
        'dataset': dataset,
    }


    ####################################### SOLVER #######################################
    model, tokenizer = None, None
    outputs, solver_total_tokens = None, None

    # oracle solver if specified, just append groundtruth answer
    if args.oracle_solver:
        outputs = [f"\\boxed{{{str(ans)}}}" for ans in dataset['answer'] for _ in range(args.solver_n_samples)]
        solver_total_tokens = 0

    # load solver outputs from cache (automatic unless disabled)
    elif not args.no_load_solver_cache:
        outputs, solver_total_tokens = solver_cache.load(cache_params)

    run_solver = (outputs == None)
    if run_solver:
        # still no outputs, initialize solver and tokenizer to run solver
        mistral_kwargs = {
            "tokenizer_mode": "mistral",
            "load_format": "mistral",
            "config_format": "mistral",
        } if 'mistralai' in args.solver_model_name.lower() else {}
        model = LLM(
            model=args.solver_model_name,
            dtype=torch.bfloat16,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            **mistral_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.solver_model_name) # initialize solver tokenizer

        # solve!
        prompts = [[{"role":
                    "user", "content": inference_prompt.format(question=question)
        }] for question in dataset['question']]
        prompts = process_prompts(tokenizer, prompts)
        outputs, solver_total_tokens = inference(
            model=model,
            prompts=prompts,
            temperature=args.solver_temperature,
            max_new_tokens=args.solver_max_new_tokens,
            top_k=args.solver_top_k,
            top_p=args.solver_top_p,
            n_samples=args.solver_n_samples,
            seed=args.seed,
        )
        print("============================ Sample solver prompt begin ============================")
        print(prompts[0])
        print("============================ Sample solver prompt end ============================")
        print("============================ Sample solver output begin ============================")
        print(outputs[0])
        print("============================ Sample solver output end ============================")

        # save to cache automatically (unless disabled)
        if not args.no_save_solver_cache:
            solver_cache.save(cache_params, outputs, solver_total_tokens)


    ####################################### EVALUATE SOLVER #######################################
    solver_total = len(dataset) * args.solver_n_samples
    assert len(outputs) == solver_total, (len(outputs), solver_total)

    records = []
    solver_correct_count, bad_solve_count = 0, 0

    for output_i, output in enumerate(outputs):
        data_i = output_i // args.solver_n_samples
        extracted_answer = extract_answer_fn(output)
        # keep track of number of bad solves for logging
        if extracted_answer == None:
            bad_solve_count += 1
        # evaluate answer and create record
        if ((extracted_answer == None) and args.include_bad_solves) or extracted_answer != None:
            is_correct = oracle_verifier_fn(
                data_row=dataset[data_i],
                solver_extracted_answer=extracted_answer,
            )
            assert isinstance(is_correct, bool)
            solver_correct_count += is_correct
            records.append({
                "data_row": dataset[data_i],
                "solver_correct": is_correct,
                "solver_full_output": output,
                'solver_extracted_answer': extracted_answer,
            })

    solver_incorrect_count = solver_total - solver_correct_count
    solver_accuracy = solver_correct_count / solver_total # include bad solver outputs
    del dataset # should no longer need this

    metrics = {
        "solver": {
            "total": solver_total,
            "bad_count": bad_solve_count,
            "correct_count": solver_correct_count,
            "incorrect_count": solver_incorrect_count,
            "accuracy": solver_accuracy,
            "gflops": solver_total_tokens * 2 * MODEL_SIZES[args.solver_model_name],
        },
    }


    ####################################### VERIFIER #######################################
    if not args.no_verify:
        set_seed(args.seed) # set seed here so the output of loading solver cache are the same to not

        # initialize verifier and tokenizer
        if (args.solver_model_name != args.verifier_model_name) or (not run_solver):
            del model, tokenizer
            gc.collect()

            mistral_kwargs = {
                "tokenizer_mode": "mistral",
                "load_format": "mistral",
                "config_format": "mistral",
                "reasoning_parser": "mistral",
                "tool_call_parser": "mistral",
            } if 'mistralai' in args.verifier_model_name.lower() else {}
            model = LLM(
                model=args.verifier_model_name,
                dtype=torch.bfloat16,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=args.seed,
                **mistral_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.verifier_model_name)

        # verify!
        prompts = [[{
            "role": "user",
            "content": verification_prompt.format(
                question=record['data_row']['question'],
                response=record['solver_full_output'],
            )
        }] for record in records]
        prompts = process_prompts(tokenizer, prompts)
        outputs, verifier_total_tokens = inference(
            model=model,
            prompts=prompts,
            temperature=args.verifier_temperature,
            max_new_tokens=args.verifier_max_new_tokens,
            top_k=args.verifier_top_k,
            top_p=args.verifier_top_p,
            n_samples=1,
            seed=args.seed,
        )
        print("============================ Sample verification prompt start ============================")
        print(prompts[0] if prompts else '(no parseable solver outputs)')
        print("============================ Sample verification prompt end ============================")
        print("============================ Sample verification output start ============================")
        print(outputs[0] if prompts else '(no parseable solver outputs)')
        print("============================ Sample verification output end ============================")

        ####################################### EVALUATE VERIFIER #######################################
        # update records
        assert len(outputs) == len(records), (len(outputs), len(records))
        for output, record in zip(outputs, records):
            record['verifier_response'] = output
            record['verifier_verdict'] = extract_verifier_answer(output) # bools or none

        # evaluate verifier and update metrics
        verifier_total = len(records)
        verifier_correct_count, bad_verify_count = 0, 0
        tp = tn = fp = fn = 0

        for record in records:
            # solver
            solver_correct = record['solver_correct']
            # verdict
            verdict = record['verifier_verdict']
            if verdict == None:
                bad_verify_count += 1
                continue
            assert isinstance(solver_correct, bool) and isinstance(verdict, bool)
            # accuracy count (verifier agrees with correctness)
            if (solver_correct and verdict) or (not solver_correct and not verdict):
                verifier_correct_count += 1
            # confusion matrix
            if solver_correct and verdict:               tp += 1
            elif (not solver_correct) and (not verdict): tn += 1
            elif (not solver_correct) and verdict:       fp += 1
            else:                                        fn += 1

        verifier_incorrect_count = verifier_total - verifier_correct_count
        verifier_accuracy = verifier_correct_count / verifier_total if verifier_total > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0 # too confident in incorrect answers
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0 # too skeptical in correct answers
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics.update({
            "verifier": {
                "total": verifier_total,
                "bad_count": bad_verify_count,
                "correct_count": verifier_correct_count,
                "incorrect_count": verifier_incorrect_count,
                "accuracy": verifier_accuracy,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "gflops": verifier_total_tokens * 2 * MODEL_SIZES[args.verifier_model_name],
            },
        })


    # save records (optionally subset)
    num_save = math.ceil(args.save_subset_ratio * len(records))
    with open(f"{args.output_dir}/record.json", "w") as f:
        json.dump(records[:num_save], f, indent=4)
    os.system(f'chmod -fR 777 {args.output_dir}')

    # save and log metrics
    print("\n============================ Final Metrics ============================")
    pprint(metrics)
    metrics_path = f"{args.output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    os.system(f'chmod -fR 777 {args.output_dir}')
    print(f'saved metrics to {metrics_path}')
