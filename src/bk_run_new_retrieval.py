import json
import argparse
from tqdm import tqdm
import os
import re
from bk_kgi_selector import KGISelector
from bk_path_retriever import PathGuidedRetriever

def parse_path_string(path_data):
    
    if isinstance(path_data, list):
        return path_data

    if isinstance(path_data, str):
       
        clean_str = re.sub(r'^[\-\*]\s+', '', path_data).strip()

        
        if "->" in clean_str:
            return [x.strip() for x in clean_str.split("->")]

        if " - " in clean_str:
            return [x.strip() for x in clean_str.split(" - ")]

        return [clean_str]

    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Original results.jsonl")
    parser.add_argument("--output_file", required=True, help="New enhanced_results.jsonl")
    args = parser.parse_args()

    print("Initializing KGI Selector (PubMedBERT)...")
    kgi = KGISelector()

    print("Initializing Path Retriever (PubMed API)...")
    retriever = PathGuidedRetriever()

    data = []
    with open(args.input_file, 'r') as f:
        for line in f: data.append(json.loads(line))

    print(f"Processing {len(data)} items...")

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            query = item['query']

            raw_paths = item.get('paths', item.get('bk_all_paths', []))

            if not raw_paths:
                raw_paths = item.get('bk_selected_paths', [])

            clean_candidates = []
            for p in raw_paths:
                parsed = parse_path_string(p)
                if len(parsed) > 1: 
                    clean_candidates.append(parsed)

            if clean_candidates:
                best_path_info = kgi.select_best_path(query, clean_candidates)
                selected_path_list = best_path_info['path_list']
                path_str_display = best_path_info['path_str']
            else:
                head = item.get('head', query.split()[0])
                tail = item.get('tail', query.split()[-1])
                selected_path_list = [head, tail]
                path_str_display = f"{head} -> {tail}"

            evidence_chain = retriever.retrieve_chain_evidence(selected_path_list)

            item['bk_selected_paths'] = [path_str_display]
            item['bk_retrieved_evidence'] = [evidence_chain]

            f_out.write(json.dumps(item) + "\n")

    print(f"✅ Enhanced data saved to {args.output_file}")

if __name__ == "__main__":
    main()
