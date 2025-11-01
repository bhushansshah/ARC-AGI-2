#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict, Tuple, Any

import numpy as np
import dotenv
import tinker
from tinker import types

dotenv.load_dotenv()

import wandb


@dataclass
class Config:
    model_name: str
    prompts_path: str  # list of {challenge_id, prompt}
    solutions_path: str  # dict mapping challenge_id -> solution
    epochs: int = 5
    batch_size: int = 1
    learning_rate: float = 5e-4
    max_sequence_length: int = 32500
    lora_rank: int = 32
    base_checkpoint_dir: str = "./test_time_finetuning"
    run_name: str = "default_run"
    # Checkpoint loading
    load_checkpoint_path: Optional[str] = None
    # Weights & Biases (optional)
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[str] = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Test-time fine-tuning for ARC-AGI challenges")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file (list of {challenge_id, prompt})")
    parser.add_argument("--solutions_path", type=str, required=True, help="Path to solutions JSON file (dict mapping challenge_id -> solution)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_seq_length", type=int, default=32500)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--base_checkpoint_dir", type=str, default="./test_time_finetuning")
    parser.add_argument("--run_name", type=str, default="default_run", help="Run name for organizing checkpoints")
    # Checkpoint loading
    parser.add_argument("--load_checkpoint", type=str, default=None, dest="load_checkpoint", help="Path to checkpoint to load before training (remote path or local)")
    # WandB options
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/org")
    parser.add_argument("--wandb_mode", type=str, default=None, help="W&B mode: online|offline|disabled")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")

    args = parser.parse_args()
    return Config(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        solutions_path=args.solutions_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_sequence_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        base_checkpoint_dir=args.base_checkpoint_dir,
        run_name=args.run_name,
        load_checkpoint_path=args.load_checkpoint,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )


def build_clients_and_tokenizer(config: Config) -> Tuple[tinker.TrainingClient, Any]:
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is not set in environment")
    service_client = tinker.ServiceClient(api_key=api_key)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    # Load checkpoint if specified
    if config.load_checkpoint_path:
        print(f'[{now()}] Loading checkpoint from: {config.load_checkpoint_path}')
        training_client.load_state(config.load_checkpoint_path)
        print(f'[{now()}] Successfully loaded training state from checkpoint')
        

    tokenizer = training_client.get_tokenizer()
    return training_client, tokenizer


def load_prompts_and_solutions(prompts_path: str, solutions_path: str) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Load prompts and solutions.
    Prompts file: list of dicts with 'challenge_id' and 'prompt' fields
    Solutions file: dict mapping challenge_id -> solution
    """
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    with open(solutions_path, "r") as f:
        solutions = json.load(f)

    if not isinstance(prompts, list):
        raise ValueError("prompts file must be a list of dicts (each with 'challenge_id' and 'prompt').")
    if not isinstance(solutions, dict):
        # be permissive: if solutions is a list of {challenge_id, solution} convert to dict
        if isinstance(solutions, list):
            sol_map = {}
            for entry in solutions:
                if 'challenge_id' in entry and 'solution' in entry:
                    sol_map[entry['challenge_id']] = entry['solution']
            solutions = sol_map
        else:
            raise ValueError("solutions file must be a dict mapping challenge_id -> solution (or a list convertible to that).")

    return prompts, solutions


def group_prompts_by_base_id(prompts: List[dict]) -> Dict[str, List[dict]]:
    """
    Group prompts by their base challenge ID (part before the last underscore).
    E.g., '0934a4d8_1', '0934a4d8_2' -> base_id='0934a4d8'
    """
    groups = defaultdict(list)
    for prompt_entry in prompts:
        if 'challenge_id' not in prompt_entry:
            continue
        challenge_id = prompt_entry['challenge_id']
        parts = challenge_id.rsplit('_', 1)
        if len(parts) == 2:
            base_id = parts[0]
        else:
            base_id = challenge_id
        groups[base_id].append(prompt_entry)
    return dict(groups)


def create_training_examples(prompt_entries: List[dict], solutions: Dict[str, Any]) -> List[dict]:
    """
    Create training examples for a base model fine-tuning setup.
    Each example is structured such that:
      - 'prompt': input text for the model
      - 'output': string formatted as:
            {
                "output": [[...]]
            }
      - 'challenge_id': identifier
    """
    examples = []

    for entry in prompt_entries:
        cid = entry.get('challenge_id')
        prompt_text = entry.get('prompt', "") or ""

        # Retrieve solution by challenge_id or from entry
        solution = None
        if cid is not None:
            solution = solutions.get(cid)
        if solution is None and 'solution' in entry:
            solution = entry['solution']

        # Build model output string in strict JSON format
        if solution is not None:
            # Wrap solution inside {"output": [[...]]}
            formatted_output = {"output": solution}
        else:
            formatted_output = {"output": []}

        # Serialize cleanly (indentation optional for readability)
        try:
            output_text = json.dumps(formatted_output, ensure_ascii=False)
        except Exception:
            output_text = '{"output": []}'

        # Ensure both fields are strings
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)
        if not isinstance(output_text, str):
            output_text = str(output_text)

        examples.append({
            "prompt": prompt_text,
            "output": output_text,
            "challenge_id": cid
        })

    return examples

def filter_prompts_by_length(prompts: list[dict], tokenizer, max_len: int) -> list[dict]:
    out: list[dict] = []
    for p in prompts:
        input_ids = tokenizer.encode(p["prompt"] + p["output"])
        if len(input_ids) <= max_len:
            out.append(p)
    return out


def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = example['prompt']
    
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(example['output'], add_special_tokens=False)

    # store the input and output values in a test file
    open("test_finetuning_input_output.txt", "a").write(f"PROMPT:\n{prompt}\nOUTPUT:\n{example['output']}\n\n")


    completion_weights = [1] * len(completion_tokens)
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
    
    # print the first  example['prompt'] and example['output'] for debugging
    print("--- Example Prompt ---")
    print(example['prompt'][:100])  # print first 1000 chars
    print("--- Example Output ---")
    print(example['output'][:100])  # print first 1000 chars

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]
 
    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


def batch_generator(examples: List[types.Datum], batch_size: int, shuffle: bool = True) -> Iterator[List[types.Datum]]:
    idxs = list(range(len(examples)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield [examples[j] for j in idxs[i:i+batch_size]]


def collate_batch(batch: List[types.Datum]) -> List[types.Datum]:
    # keep as-is, tinker client expects list of Datum
    return batch


def compute_mean_nll(logprobs_list, weights_list):
    total = 0.0
    total_w = 0.0
    for lp_seq, w_seq in zip(logprobs_list, weights_list):
        if hasattr(lp_seq, 'tolist'):
            lp_seq = lp_seq.tolist()
        elif hasattr(lp_seq, 'to_numpy'):
            lp_seq = lp_seq.to_numpy()

        if hasattr(w_seq, 'tolist'):
            w_seq = w_seq.tolist()
        elif hasattr(w_seq, 'to_numpy'):
            w_seq = w_seq.to_numpy()

        lp = np.array(lp_seq, dtype=float)
        w = np.array(w_seq, dtype=float)

        if lp.shape != w.shape:
            min_len = min(lp.shape[0], w.shape[0])
            lp = lp[:min_len]
            w = w[:min_len]

        total += (-lp * w).sum()
        total_w += w.sum()

    return float(total / total_w) if total_w > 0 else None


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def save_checkpoint(training_client: tinker.TrainingClient, checkpoint_dir: str, base_id: str, step_num: int, metadata: dict | None = None):
    train_remote_path = None
    sampler_remote_path = None
    try:
        remote_name = f"{base_id}-{step_num:06d}"
        if hasattr(training_client, 'save_state'):
            fut = training_client.save_state(name=remote_name)
            result_obj = fut.result() if hasattr(fut, 'result') else fut
            train_remote_path = getattr(result_obj, 'path', None)
            if train_remote_path:
                print(f'[{now()}] Saved remote training state -> {train_remote_path}')

        if hasattr(training_client, 'save_weights_for_sampler'):
            fut = training_client.save_weights_for_sampler(name=remote_name)
            result_obj = fut.result() if hasattr(fut, 'result') else fut
            sampler_remote_path = getattr(result_obj, 'path', None)
            if sampler_remote_path:
                print(f'[{now()}] Saved remote weights for sampler -> {sampler_remote_path}')
    except Exception as e:
        print(f'[{now()}] Warning: remote checkpoint save failed: {e}')

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_meta = {
        'step': step_num,
        'timestamp': now(),
        'base_id': base_id,
        'train_remote_path': train_remote_path,
        'sampler_remote_path': sampler_remote_path
    }
    if metadata:
        checkpoint_meta.update(metadata)

    local_name = f'{base_id}_step_{step_num}.json'
    path = os.path.join(checkpoint_dir, local_name)
    with open(path, 'w') as f:
        json.dump(checkpoint_meta, f, indent=2)
    print(f'[{now()}] Saved local checkpoint metadata -> {path}')
    return path, train_remote_path, sampler_remote_path


async def call_forward_backward(training_client: tinker.TrainingClient, batch):
    if hasattr(training_client, 'forward_backward_async'):
        maybe_coro = training_client.forward_backward_async(batch, loss_fn='cross_entropy')
        res = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
        if hasattr(res, 'result_async'):
            return await res.result_async()
        if hasattr(res, 'result'):
            return res.result()
        return res
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, training_client.forward_backward, batch)


async def call_optim_step(training_client: tinker.TrainingClient, adam_params):
    if hasattr(training_client, 'optim_step_async'):
        maybe_coro = training_client.optim_step_async(adam_params)
        res = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
        if hasattr(res, 'result_async'):
            return await res.result_async()
        if hasattr(res, 'result'):
            return res.result()
        return res
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, training_client.optim_step, adam_params)


async def train_challenge_group(
    config: Config,
    base_id: str,
    training_client: tinker.TrainingClient,
    tokenizer,
    processed_examples: List[types.Datum],
    checkpoint_dir: str,
):
    """Train a model for a specific challenge group."""
    global_step = 0
    num_examples = len(processed_examples)

    if num_examples == 0:
        print(f'[{now()}] No examples for {base_id}, skipping.')
        return

    steps_per_epoch = max(1, math.ceil(num_examples / config.batch_size))
    print(f'[{now()}] Training {base_id}: epochs={config.epochs}, batch_size={config.batch_size}, examples={num_examples}')

    # Initialize W&B for this challenge group if enabled
    wb_run = None
    if config.wandb_enabled and wandb is not None:
        try:
            run_name = f"{config.wandb_run_name or config.run_name}_{base_id}"
            wb_kwargs = {
                'project': config.wandb_project or 'ARC-AGI-TestTime',
                'name': run_name,
                'reinit': True,
            }
            if config.wandb_entity:
                wb_kwargs['entity'] = config.wandb_entity
            if config.wandb_mode:
                os.environ['WANDB_MODE'] = config.wandb_mode
            if config.wandb_tags:
                tags = [t.strip() for t in config.wandb_tags.split(',') if t.strip()]
                if tags:
                    wb_kwargs['tags'] = tags
            wb_run = wandb.init(**wb_kwargs)
            wandb.config.update({
                'base_id': base_id,
                'model_name': config.model_name,
                'lora_rank': config.lora_rank,
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'num_examples': num_examples,
            }, allow_val_change=True)
        except Exception as e:
            print(f'[{now()}] Warning: Failed to initialize wandb for {base_id}: {e}')

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        epoch_loss_accum = 0.0
        epoch_items = 0

        for batch in batch_generator(processed_examples, config.batch_size, shuffle=True):
            global_step += 1
            batch = collate_batch(batch)

            # Linear LR schedule
            step = global_step - 1
            lr_mult = max(0.0, 1.0 - step / (steps_per_epoch * config.epochs))
            current_lr = config.learning_rate * lr_mult
            adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

            try:
                fwd_res = await call_forward_backward(training_client, batch)
            except Exception as e:
                print(f'[{now()}] forward_backward failed at step {global_step}: {e}')
                continue

            try:
                _ = await call_optim_step(training_client, adam_params)
            except Exception as e:
                print(f'[{now()}] optim_step failed at step {global_step}: {e}')

            # Extract logprobs
            train_logprobs = []
            try:
                lf_outputs = getattr(fwd_res, 'loss_fn_outputs', None)
                if isinstance(lf_outputs, list):
                    for entry in lf_outputs:
                        if isinstance(entry, dict) and 'logprobs' in entry:
                            train_logprobs.append(entry['logprobs'])
                elif isinstance(lf_outputs, dict) and 'logprobs' in lf_outputs:
                    train_logprobs = [lf_outputs['logprobs']]
            except Exception as e:
                print(f'[{now()}] Could not extract loss_fn_outputs: {e}')

            train_weights = [d.loss_fn_inputs['weights'] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights) if train_logprobs else None

            if train_nll is not None:
                batch_total_tokens = 0.0
                for w in train_weights:
                    if hasattr(w, 'tolist'):
                        batch_total_tokens += np.array(w.tolist()).sum()
                    elif hasattr(w, 'to_numpy'):
                        batch_total_tokens += w.to_numpy().sum()
                    else:
                        batch_total_tokens += np.array(w).sum()

                epoch_loss_accum += train_nll * batch_total_tokens
                epoch_items += batch_total_tokens

            if config.wandb_enabled and wandb is not None and train_nll is not None:
                try:
                    wandb.log({
                        'train/nll': train_nll,
                        'train/lr': current_lr,
                        'train/epoch': epoch,
                        'train/step': global_step,
                    }, step=global_step)
                except Exception as e:
                    print(f'[{now()}] Warning: wandb.log failed: {e}')

            if global_step % 10 == 0:
                avg_loss = (epoch_loss_accum / epoch_items) if epoch_items > 0 else None
                print(f'[{now()}] {base_id} Epoch {epoch} step {global_step} avg_loss={avg_loss} lr={current_lr}')

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = (epoch_loss_accum / epoch_items) if epoch_items > 0 else None
        print(f'[{now()}] {base_id} Finished epoch {epoch}/{config.epochs} time={epoch_time:.1f}s avg_loss={epoch_avg_loss}')

        if config.wandb_enabled and wandb is not None:
            try:
                wandb.log({
                    'epoch/avg_loss': epoch_avg_loss,
                    'epoch/time_sec': epoch_time,
                    'epoch/index': epoch,
                }, step=global_step)
            except Exception as e:
                print(f'[{now()}] Warning: wandb.log (epoch) failed: {e}')

    # Save final checkpoint
    local_path, train_remote_path, sampler_remote_path = save_checkpoint(
        training_client, checkpoint_dir, base_id, global_step,
        metadata={'epoch': config.epochs, 'final': True}
    )

    if config.wandb_enabled and wandb is not None:
        try:
            wandb.log({
                'checkpoint/final_local_path': local_path,
                'checkpoint/final_train_remote_path': train_remote_path,
                'checkpoint/final_sampler_remote_path': sampler_remote_path,
            }, step=global_step)
        except Exception as e:
            print(f'[{now()}] Warning: wandb.log (final checkpoint) failed: {e}')

    print(f'[{now()}] {base_id} Training finished. Total steps: {global_step}')

    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass


def main():
    config = parse_args()

    # Load prompts and solutions
    print(f'[{now()}] Loading prompts and solutions...')
    prompts, solutions = load_prompts_and_solutions(
        config.prompts_path,
        config.solutions_path
    )
    print(f'[{now()}] Loaded {len(prompts)} prompts and {len(solutions)} solutions.')

    # Group prompts by base ID
    prompt_groups = group_prompts_by_base_id(prompts)
    print(f'[{now()}] Grouped into {len(prompt_groups)} base challenge groups.')

    # Process each challenge group
    for base_id, prompt_entries in sorted(prompt_groups.items()):
        print('\n' + "=" * 80)
        print(f'[{now()}] Processing challenge group: {base_id} ({len(prompt_entries)} challenges)')
        print("=" * 80)

        # Build new training client for this challenge group (will load checkpoint if specified)
        training_client, tokenizer = build_clients_and_tokenizer(config)

        # Create training examples for this group (uses prompt_entries which are raw entries from your prompts JSON)
        examples = create_training_examples(prompt_entries, solutions)
        print(f'[{now()}] Created {len(examples)} training examples for {base_id}')

        # Filter by length
        filtered_examples = filter_prompts_by_length(examples, tokenizer, config.max_sequence_length)
        print(f'[{now()}] After length filtering: {len(filtered_examples)} examples')

        # Process examples into types.Datum objects expected by tinker client
        processed_examples = [process_example(ex, tokenizer) for ex in filtered_examples]

        # Create checkpoint directory for this challenge group
        checkpoint_dir = os.path.join(config.base_checkpoint_dir, config.run_name, base_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Train (async)
        asyncio.run(train_challenge_group(
            config, base_id, training_client, tokenizer,
            processed_examples, checkpoint_dir
        ))

    print(f'\n[{now()}] All challenge groups processed!')


if __name__ == "__main__":
    main()
 