
import json
import csv
import argparse
import os


def convert_to_tsv(input_file, output_file):
    print(f"Converting {input_file} to {output_file}...")

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return

    rows = []
    seen_text = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
               
                text = item.get('text', item.get('abstract', '')).replace('\t', ' ').replace('\n', ' ')
                title = item.get('title', '').replace('\t', ' ').replace('\n', ' ')
                pmid = item.get('pmid', '0')

                if text and text not in seen_text:
                    rows.append([pmid, text, title])
                    seen_text.add(text)
            except:
                pass

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'text', 'title'])
        writer.writerows(rows)

    print(f"Done. Total documents: {len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input raw jsonl file")
    parser.add_argument("--output", required=True, help="Output tsv file")
    args = parser.parse_args()

    convert_to_tsv(args.input, args.output)
