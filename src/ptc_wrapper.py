
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np


class PTCRanker:
    def __init__(self, model_path, device='cuda:1'):
        self.device = device
        print(f"Loading PTC Ranker from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config,
                                                                        local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def rank(self, entity_pair_str, candidate_paths, top_k=5):
        """
        entity_pair_str: "Drug A Disease B"
        candidate_paths: List of strings ["Drug A - Gene X - Disease B", ...]
        """
        if not candidate_paths:
            return []
        inputs = self.tokenizer(
            [entity_pair_str] * len(candidate_paths),
            candidate_paths,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            scores = outputs.logits.squeeze(-1).cpu().numpy()  # [Batch]

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        top_paths = [candidate_paths[i] for i in ranked_indices]
        top_scores = [float(scores[i]) for i in ranked_indices]

        return top_paths
