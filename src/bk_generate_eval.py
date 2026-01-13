import json
import torch
import os
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


MODEL_PATH = "models/llama3-8b-instruct"
DEVICE = "cuda:1"


def load_truth_file(filepath):
    truth_map = {}
    truth_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            val = None
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    qid = str(item.get('qid', ''))
                    raw_val = item.get('ground_truth', item.get('label', item.get('value', 1)))
                    val = int(raw_val)
                    if qid: truth_map[qid] = val
                else:
                    val = int(item)
            except:
                parts = line.split()
                if parts:
                    last_col = parts[-1]
                    if last_col.replace('-', '').isdigit():
                        val = int(last_col)
                    elif last_col.lower() == 'true':
                        val = 1
                    elif last_col.lower() == 'false':
                        val = 0
            if val is not None: truth_list.append(val)
    return truth_map, truth_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--truth_file", required=True)
    parser.add_argument("--output_score", required=True)
    parser.add_argument("--output_explanation", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Loading Llama-3 from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE,
        local_files_only=True
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    print(f"Reading Ground Truth from {args.truth_file}...")
    truth_map, truth_list = load_truth_file(args.truth_file)

    print(f"Reading results from {args.input_file}...")
    data = []
    with open(args.input_file, 'r') as f:
        for line in f: data.append(json.loads(line))

    y_true = []
    y_pred = []
    explanations = []

    print(f">>> Starting Inference with Threshold = {args.threshold} ...")

    
    debug_print_count = 0

    for idx, item in enumerate(tqdm(data)):
        qid = str(item.get('qid', ''))
        if qid in truth_map:
            raw_label = truth_map[qid]
        elif idx < len(truth_list):
            raw_label = truth_list[idx]
        else:
            raw_label = 1
        gold_label = 1 if raw_label > 0 else 0

        query = item['query']
        paths = item.get('bk_selected_paths', [])
        docs = item.get('bk_retrieved_evidence', [])

        path_str = " | ".join(paths[:2]) if paths else "No specific path found."
        doc_str = " ".join(docs[:1]) if docs else "No literature found."
        if len(doc_str) > 2000: doc_str = doc_str[:2000] + "..."

        
        messages = [
            {"role": "system", "content": "You are a biomedical expert."},
            {"role": "user", "content": f"""Task: Estimate the probability (0-100%) that the Head entity causes the Tail entity.

[Query]: {query}
[Path]: {path_str}
[Evidence]: {doc_str}

Instruction:
1. If the text mentions a mechanism, association, or side effect link, the probability should be HIGH (>50%).
2. Only give LOW probability (<30%) if the evidence is missing or explicitly denies the link.
3. Be optimistic: Indirect paths are valid evidence.

Output Format:
Reason: [Brief Analysis]
Probability: [Number]%"""}
        ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=128,
                eos_token_id=terminators,
                do_sample=False  
            )

        response = outputs[0][input_ids.shape[-1]:]
        generated = tokenizer.decode(response, skip_special_tokens=True).strip()

        
        match = re.search(r"Probability:.*?(\d+)", generated, re.IGNORECASE)
        score_val = 0.0

        if match:
            try:
                raw_score = float(match.group(1))
                score_val = raw_score / 100.0  # 转为 0.0 - 1.0
            except:
                score_val = 0.0

        
        if debug_print_count < 5:
            print(f"\n[DEBUG] QID:{qid} | Gold:{gold_label} | RawOut: {generated} | ParsedScore: {score_val}")
            debug_print_count += 1

        pred = 1 if score_val >= args.threshold else 0

        y_true.append(gold_label)
        y_pred.append(pred)

        if args.output_explanation:
            explanations.append({
                "qid": qid, "query": query, "gold": gold_label, "pred": pred,
                "confidence_score": score_val,
                "model_explanation": generated
            })

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"Llama-3 Evaluation Results (Threshold={args.threshold})")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print("=" * 50)

    with open(args.output_score, 'w') as f:
        f.write(
            f"Threshold: {args.threshold}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAccuracy: {accuracy:.4f}\n")

    if args.output_explanation:
        with open(args.output_explanation, 'w', encoding='utf-8') as f:
            for exp in explanations:
                f.write(json.dumps(exp) + "\n")


if __name__ == "__main__":
    main()
