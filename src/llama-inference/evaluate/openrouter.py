import argparse
import json
import time
from typing import List, Dict

from ..data import Evaluation
from tqdm import tqdm
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated JSON with a Critic model via OpenRouter.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to JSON file with model generations.")
    parser.add_argument("--api-key", type=str, required=True, help="OpenRouter API key.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="deepseek/deepseek-chat-v3.1:free",
        help="OpenRouter model ID (e.g., 'openrouter/anthropic/claude-3.5-sonnet').",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to save results with messages and verdicts.",
    )
    
    parser.add_argument("--print-every", type=int, default=10, help="Print a sample every N steps.")
    return parser.parse_args()


def build_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def chat_once(client: OpenAI, model: str, messages: List[Dict], max_retries: int = 5, timeout_s: float = 60.0) -> str:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=4,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise TimeoutError("Max API call retries exceeded") from e
            time.sleep(delay)
            delay = min(delay * 2, timeout_s)
    return ""


def main():
    args = parse_args()
    client = build_client(args.api_key)

    aa_lcr_eval = Evaluation(args.input_json, split="test")

    outputs_count = 0
    correct_count = 0
    results = []

    for i, messages in tqdm(enumerate(aa_lcr_eval), desc="Generating"):
        # Print human-readable sample
        if i%args.print_every == 0:
            print(f"\n=== Sample Messages at step {i} ===")
            for m in messages:
                print(f"{m['role'].upper()}: {m['content']}\n")

        verdict = chat_once(client, args.model_id, messages).strip()
        print(f"--- Verdict: {verdict} ---\n")

        if verdict.upper().startswith("CORRECT"):
            correct_count += 1
        if not verdict.upper().startswith("CORRECT") and not verdict.upper().startswith("INCORRECT"):
            print(f"==> Warning: Unexpected verdict '{verdict}'")
        outputs_count += 1

        results.append({
            "messages": messages,
            "verdict": verdict,
        })

    # Write all results to JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal accuracy: {correct_count}/{outputs_count}")
    print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()