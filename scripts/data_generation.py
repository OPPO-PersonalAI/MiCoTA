import os
import json
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description='data generation')
    parser.add_argument('--model', type=str, default="DeepSeek-R1-Distill-Qwen-32B",
                        help='model name or path')
    parser.add_argument('--dataset', type=str,
                        help='dataset')
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default="result.json",
                        help='output path')
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--max_model_len', type=int, default=16384)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=16384)
    return parser.parse_args()

def build_prompt(question):
    return f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

def calculate_token_length(response, tokenizer):
    tokens = tokenizer.tokenize(response)
    return len(tokens) < 16000

def main():
    args = parse_arguments()
    
    disable_caching()
    
    print(f"Loading Dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    ds = ds[args.split]
    problems = ds["problem"]
    answers = ds["answer"]
    print(f"The dataset has been loaded. There are a total of {len(problems)} samples")
    
    print(f"Loading Model: {args.model}")
    model = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enforce_eager=True
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_special_tokens=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    all_results = []
    total_batches = (len(problems) + args.batch_size - 1) // args.batch_size
    
    print(f" Start processing, total {total_batches} batches ")
    for i in range(0, len(problems), args.batch_size):
        batch_idx = i // args.batch_size + 1
        print(f" Processing batch {batch_idx}/{total_batches}")
        
        batch_problems = problems[i:i + args.batch_size]
        batch_answers = answers[i:i + args.batch_size]
        batch_prompts = [build_prompt(problem) for problem in batch_problems]
        
        batch_outputs = model.generate(batch_prompts, sampling_params)
        
        batch_results = []
        for j, output in enumerate(batch_outputs):
            generated_text = output.outputs[0].text
            if calculate_token_length(generated_text, tokenizer):
                result = {
                    "instruction": batch_prompts[j],
                    "output": generated_text,
                    "answer": batch_answers[j]
                }
                batch_results.append(result)
        
        all_results.extend(batch_results)
        print(f" Batch {batch_idx} processing completed, valid samples: {len(batch_results)}/{len(batch_problems)}")
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f" All batch processing completed, results saved to: {args.output}")
    print(f" processes a total of {len(problems)} samples and generates valid results {len(all_results)} ")

if __name__ == "__main__":
    main()