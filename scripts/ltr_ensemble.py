import json
import argparse
from collections import defaultdict


def load_rankings(file_path):
    
    rankings = defaultdict(dict)
    full_data = {}

    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = str(item.get('qid', ''))
            full_data[qid] = item

            paths = item.get('bk_selected_paths', [])
            for rank, path in enumerate(paths):
                rankings[qid][path] = rank

    return rankings, full_data


def rrf_fusion(ranking_list, k=60):
    fused_scores = defaultdict(float)
    for rankings in ranking_list:
        for path, rank in rankings.items():
            fused_scores[path] += 1.0 / (k + rank + 1)
    return fused_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs='+', required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    all_rankings = []
    base_data = {}

    for fpath in args.input_files:
        ranks, data = load_rankings(fpath)
        all_rankings.append(ranks)
        base_data.update(data)

    print(f"Fusing results from {len(args.input_files)} models...")

    with open(args.output_file, 'w') as f_out:
        for qid, item in base_data.items():
            qid_rankings = [r[qid] for r in all_rankings if qid in r]
            if not qid_rankings: continue

            path_scores = rrf_fusion(qid_rankings)
            sorted_paths = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)
            final_paths = [p[0] for p in sorted_paths[:args.top_k]]

            out_obj = item.copy()
            out_obj['bk_selected_paths'] = final_paths
            out_obj['fusion_method'] = 'RRF'

            f_out.write(json.dumps(out_obj) + "\n")

    print(f"Ensemble results saved to {args.output_file}")


if __name__ == "__main__":
    main()
