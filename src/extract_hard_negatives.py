import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_result", required=True, help="Path to enhanced_agent.jsonl")
    parser.add_argument("--output_file", required=True, help="Path to save hard_negatives.jsonl")
    args = parser.parse_args()

    hard_negatives = []

    with open(args.agent_result, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                evidence = item.get("bk_retrieved_evidence", [""])[0]

               
                if "[Missing]" in evidence or "No document found" in evidence or "No valid evidence" in evidence:
        
                    path_str = item['bk_selected_paths'][0]  # "A -> B -> C"
                    
                    stops = path_str.replace("->", "-").replace(" - ", " - ")

                    hn_item = {
                        "e1": item.get('head', ''),
                        "e2": item.get('tail', ''),
                        "metapaths": [{
                            "pathid": 9999,  
                            "stops": stops,
                            "rel_score": -1.0,  
                            "label": 0  
                        }]
                    }
                    hard_negatives.append(hn_item)
            except:
                continue

    with open(args.output_file, 'w') as f:
        for item in hard_negatives:
            f.write(json.dumps(item) + "\n")

    print(f"Extracted {len(hard_negatives)} hard negative paths.")


if __name__ == "__main__":
    main()
