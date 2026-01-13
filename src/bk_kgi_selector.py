import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


MODEL_PATH = "models/pubmedbert_kgi"





class KGISelector:
    def __init__(self, model_path=MODEL_PATH, device='cuda:1'):
        print(f"Loading KGI Model from {model_path}...")
        self.model = SentenceTransformer(model_path, device=device)

    def select_best_path(self, query, paths):
        
        if not paths:
            return None

        
        path_texts = [" -> ".join(p) if isinstance(p, list) else p for p in paths]

        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        path_embs = self.model.encode(path_texts, convert_to_tensor=True)

        
        cosine_scores = util.cos_sim(query_emb, path_embs)[0]

        
        best_idx = torch.argmax(cosine_scores).item()

        return {
            "path_list": paths[best_idx],  
            "path_str": path_texts[best_idx],  
            "score": cosine_scores[best_idx].item()
        }



if __name__ == "__main__":
    selector = KGISelector()
    q = "Is there a relationship between insulin and hypoglycemia?"
    candidates = [
        ["insulin", "decreases", "blood glucose", "causes", "hypoglycemia"],  
        ["insulin", "treats", "diabetes"],  
        ["aspirin", "treats", "headache"]  
    ]
    result = selector.select_best_path(q, candidates)
    print(f"Query: {q}")
    print(f"Selected: {result['path_str']} (Score: {result['score']:.4f})")
