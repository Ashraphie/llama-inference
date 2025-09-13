import os
import json
import argparse
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from data import AA_LCR
from vllm import LLM, SamplingParams

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER = None

def parse_args():
    parser = argparse.ArgumentParser(description="Generate answers with vLLM and save to JSON.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling (<=0 disables).")
    parser.add_argument("--output-json", type=str, default="outputs.json", help="Path to save JSON results.")
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
    NUM_GPUS = torch.cuda.device_count()

    aa_lcr = AA_LCR()
    loader = DataLoader(
        aa_lcr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_chat,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )

    # Initialize vLLM with tensor parallel across GPUs
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=max(1, NUM_GPUS),
        dtype="bfloat16",
        trust_remote_code=False,
    )
    eos_id = llm.get_tokenizer().eos_token_id
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k and args.top_k > 0 else None,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[eos_id],
    )

    outputs_count = 0
    all_results = []  # Collect results for JSON dump

    for (_enc, prompts) in tqdm(loader, desc="Generating"):
        results = llm.generate(prompts, sampling_params)
        for out in results:
            text = out.outputs[0].text if out.outputs else ""
            print(f"\n\n =====> Sample {outputs_count}: {text}")
            all_results.append({
                "id": outputs_count,
                "answer": text
            })
            outputs_count += 1

    # Save all results to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_results)} results to {args.output_json}")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
