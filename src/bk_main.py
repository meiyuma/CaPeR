import os
import sys
import json
import torch
import argparse
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    from ptc_wrapper import PTCRanker
    from bridge import MutualVerifier
    from contriever.src.contriever import load_retriever
    from contriever.src.index import Indexer
except ImportError:
    pass  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ptc_ckpt", required=True)
    parser.add_argument("--retriever_path", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--corpus_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    device = "cuda:1"

    # 1. Ranker
    ptc_ranker = PTCRanker(args.ptc_ckpt, device=device)

    # 2. Retriever
    retriever, tokenizer, _ = load_retriever(args.retriever_path)
    retriever.to(device).eval()

    index = Indexer(vector_sz=768, n_subquantizers=0, n_bits=8)
    index.deserialize_from(args.index_dir)

    doc_id_map = {}
    with open(args.corpus_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2: doc_id_map[str(parts[0])] = parts[1]

    # 3. Verifier
    verifier = MutualVerifier(device=device)

    # 4. Run
    data = []
    with open(args.test_file, 'r') as f:
        for line in f: data.append(json.loads(line))

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            query = f"{item['e1']} {item['e2']}"
            paths = [p['stops'] for p in item.get('metapaths', [])]
            if not paths: continue

            top_paths = ptc_ranker.rank(query, paths, top_k=3)

            # Retrieve
            with torch.no_grad():
                inputs = tokenizer([top_paths[0]], padding=True, truncation=True, return_tensors='pt').to(device)
                q_emb = retriever(**inputs, normalize=True).cpu().numpy()

            ids_scores = index.search_knn(q_emb, top_docs=10)
            docs = [doc_id_map.get(str(did), "") for did in ids_scores[0][0]]
            docs = [d for d in docs if d]
            if not docs: continue

            # Verify
            with torch.no_grad():
                p_in = tokenizer(top_paths, padding=True, truncation=True, return_tensors='pt', max_length=128).to(
                    device)
                d_in = tokenizer(docs, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
                p_emb = retriever(**p_in, normalize=True)
                d_emb = retriever(**d_in, normalize=True)

            f_paths, f_docs = verifier.verify(p_emb, d_emb, top_paths, docs, threshold=0.4)

            res = {
                "qid": item.get('qid', ''), "query": query,
                "bk_selected_paths": f_paths, "bk_retrieved_evidence": f_docs
            }
            f_out.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
