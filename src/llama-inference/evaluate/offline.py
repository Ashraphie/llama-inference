import argparse
from ..data import Evaluation
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
import multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NUM_GPUS = torch.cuda.device_count()
TOKENIZER = None

def parse_args():
    parser = argparse.ArgumentParser(description="Evalute the generated JSON file with a Critic model.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to JSON file with model generations.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--output-json", type=str, required=True, help="Path to save results with prompts and verdicts.")
    return parser.parse_args()
    
def worker_init_fn(_):
    global TOKENIZER, MODEL_ID
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    TOKENIZER = tok
    
def collate_chat(batch):
    assert TOKENIZER is not None, "TOKENIZER not initialized"
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.padding_side = "left"
    prompts = [
        TOKENIZER.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        for msgs in batch
    ]
    enc = TOKENIZER(prompts, return_tensors="pt", padding=True)
    return enc, prompts

def main():
    args = parse_args()
    global MODEL_ID, NUM_GPUS
    critic = LLM(
        model=MODEL_ID,
        tensor_parallel_size=max(1, NUM_GPUS),
        dtype="bfloat16",
        trust_remote_code=False,
    )

    output_json_file = args.input_json
    aa_lcr_eval = Evaluation(output_json_file, split="test")

    loader = DataLoader(
        aa_lcr_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_chat,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )

    eos_id = critic.get_tokenizer().eos_token_id
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4,
        stop_token_ids=[eos_id],
    )

    outputs_count = 0
    correct_count = 0
    results_log = []

    for (_enc, prompts) in tqdm(loader, desc="Generating"):
        generations = critic.generate(prompts, sampling_params)

        for prompt, out in zip(prompts, generations):
            text = out.outputs[0].text if out.outputs else ""
            verdict = text.strip()

            # Human-readable logs
            print("\n=== Prompt ===")
            print(prompt)
            print(f"--- Verdict: {verdict} ---\n")

            if verdict == "CORRECT":
                correct_count += 1
            outputs_count += 1

            results_log.append({
                "prompt": prompt,
                "verdict": verdict,
            })

    # Save detailed JSON log
    with open(args.output_json, "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\nFinal accuracy: {correct_count}/{outputs_count}")
    print(f"Saved detailed results to {args.output_json}")
    
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()