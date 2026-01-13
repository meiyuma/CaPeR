import json
import argparse
from tqdm import tqdm
import os
import re
import time

from bk_kgi_selector import KGISelector
from bk_path_retriever import PathGuidedRetriever
from bk_agent_verifier import BKAgent



MODEL_PATH = "models/llama3-8b-instruct"

def parse_path_string(path_data):
    if isinstance(path_data, list): return path_data
    if isinstance(path_data, str):
        clean_str = re.sub(r'^[\-\*]\s+', '', path_data).strip()
        if "->" in clean_str: return [x.strip() for x in clean_str.split("->")]
        if " - " in clean_str: return [x.strip() for x in clean_str.split(" - ")]
        return [clean_str]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    
    print(">>> Stage 1: Initializing Modules...")
    kgi = KGISelector() 
    retriever = PathGuidedRetriever() 
    agent = BKAgent(MODEL_PATH)  

    
    data = []
    print(f"Reading from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass

    print(f"Processing {len(data)} items with Agentic Retrieval...")

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            query = item.get('query', '')
            if not query and 'head' in item:
                query = f"Is there a causal relationship between {item['head']} and {item['tail']}?"

           
            raw_paths = item.get('paths', item.get('bk_all_paths', item.get('bk_selected_paths', [])))
            clean_candidates = []
            if raw_paths:
                if isinstance(raw_paths, str): raw_paths = [raw_paths]
                for p in raw_paths:
                    parsed = parse_path_string(p)
                    if len(parsed) > 1: clean_candidates.append(parsed)

            if clean_candidates:
                best_path_info = kgi.select_best_path(query, clean_candidates)
                selected_path_list = best_path_info['path_list']
                path_str_display = best_path_info['path_str']
            else:
                head = item.get('head', 'Entity1')
                tail = item.get('tail', 'Entity2')
                selected_path_list = [head, tail]
                path_str_display = f"{head} -> {tail}"

            
            evidence_chain_strs = []

            for i in range(len(selected_path_list) - 1):
                node_a = selected_path_list[i]
                node_b = selected_path_list[i + 1]
                step_label = f"{node_a} -> {node_b}"

               
                evidence = retriever.search_pubmed_snippet(node_a, node_b)

               
                if evidence:
                    is_valid, reason = agent.verify_evidence(step_label, evidence)
                else:
                    is_valid, reason = False, "No document found."

                
                final_evidence_entry = ""
                if is_valid:
                    final_evidence_entry = f"**Step {i + 1} [Verified]**: {evidence}"
                else:
                  
                    new_query = agent.rewrite_query(node_a, node_b, reason, global_context=query)
                   
                    retry_evidence = retriever.search_pubmed_custom(new_query)

                    if retry_evidence:
                        final_evidence_entry = f"**Step {i + 1} [Recovered by Agent]**: (Query: {new_query}) {retry_evidence}"
                    else:
                        final_evidence_entry = f"**Step {i + 1} [Missing]**: Could not find evidence even after retry."

                evidence_chain_strs.append(final_evidence_entry)
                time.sleep(0.2)  
            full_evidence_text = "\n\n".join(evidence_chain_strs)

            
            item['bk_selected_paths'] = [path_str_display]
            item['bk_retrieved_evidence'] = [full_evidence_text]

            f_out.write(json.dumps(item) + "\n")

    print(f"✅ Enhanced Agent data saved to {args.output_file}")


if __name__ == "__main__":
    main()
