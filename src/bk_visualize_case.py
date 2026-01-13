
import json
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "models/mistral-7b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input results.jsonl")
    parser.add_argument("--output_file", default=None, help="Path to save Markdown report")
    parser.add_argument("--max_cases", type=int, default=20, help="Number of cases to generate")
    args = parser.parse_args()

   
    if args.output_file is None:
        input_dir = os.path.dirname(args.input_file)
        args.output_file = os.path.join(input_dir, "case_study_report.md")

    print(f"Loading Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

   
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:1",
        local_files_only=True,
        use_safetensors=False
    )

    print(f"Reading data from {args.input_file}...")
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        return

    data = []
    with open(args.input_file, 'r') as f:
        for line in f: data.append(json.loads(line))

    print(f"Generating explanations for {min(len(data), args.max_cases)} cases...")

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("# Bio-KGCDG Case Study Report\n\n")
        f_out.write(f"**Source File:** `{args.input_file}`\n\n")
        f_out.write("---\n\n")

        for i, item in enumerate(data[:args.max_cases]):
            query = item['query']
            paths = item.get('bk_selected_paths', [])
            docs = item.get('bk_retrieved_evidence', [])

            path_str = "\n".join([f"- {p}" for p in paths[:2]]) if paths else "No path found."
            doc_str = docs[0][:600] + "..." if docs else "No literature found."  

            prompt = f"""[INST] You are a biomedical expert. Analyze the Causal Relationship based on the evidence.

[Graph Paths]:
{path_str}

[Literature Snippet]:
{doc_str}

[Query]: Is there a causal relationship in "{query}"?

Structure:
1. Analysis: (Explain the logic connecting entities)
2. Conclusion: (Causal/Non-causal)
[/INST]
Analysis:"""

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:1")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Analysis:" in full_output:
                explanation = full_output.split("Analysis:")[-1].strip()
            elif "[/INST]" in full_output:
                explanation = full_output.split("[/INST]")[-1].strip()
            else:
                explanation = full_output

            f_out.write(f"### Case {i + 1}: {query}\n\n")
            f_out.write(f"**Graph Paths:**\n```\n{path_str}\n```\n")
            f_out.write(f"**Literature Evidence:**\n> {doc_str}\n\n")
            f_out.write(f"**Model Reasoning:**\n{explanation}\n\n")
            f_out.write("---\n")

            print(f"Processed Case {i + 1}")

    print(f"\n✅ Success! Report saved to: {args.output_file}")


if __name__ == "__main__":
    main()
