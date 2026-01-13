import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    print(f"Formatting {args.input_file}...")

    formatted_data = []
    with open(args.input_file, 'r') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
            except:
                continue

                
            head = item.get('e1')
            tail = item.get('e2')

            if not head or not tail:
                print(f"Skipping line {idx}: Missing e1/e2")
                continue

            
            raw_metapaths = item.get('metapaths', [])
            extracted_paths = []

            for p in raw_metapaths:
               
                if isinstance(p, dict) and 'stops' in p:
                    extracted_paths.append(p['stops'])
               
                elif isinstance(p, str):
                    extracted_paths.append(p)


            query = f"Is there a causal relationship between {head} and {tail}?"

            
            label = item.get('ground_truth', item.get('label', 1))

            new_item = {
                "qid": item.get("qid", str(idx)),
                "query": query,
                "head": head,
                "tail": tail,
                "bk_all_paths": extracted_paths,  
                "paths": extracted_paths,  
                "label": int(label)
            }

            formatted_data.append(new_item)

    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for item in formatted_data:
            f_out.write(json.dumps(item) + "\n")

    print(f"✅ Converted {len(formatted_data)} items.")
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
