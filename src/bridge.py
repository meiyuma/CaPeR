
import torch
import torch.nn.functional as F


class MutualVerifier:
    def __init__(self, device='cuda:1'):
        self.device = device

    def verify(self, path_embeddings, doc_embeddings, path_texts, doc_texts, threshold=0.0):
       
        p_norm = F.normalize(path_embeddings, p=2, dim=1)
        d_norm = F.normalize(doc_embeddings, p=2, dim=1)

        sim_matrix = torch.mm(p_norm, d_norm.transpose(0, 1))

        path_scores, _ = torch.max(sim_matrix, dim=1) 

        doc_scores, _ = torch.max(sim_matrix, dim=0) 

        valid_path_idx = torch.where(path_scores > threshold)[0].cpu().numpy()
        valid_doc_idx = torch.where(doc_scores > threshold)[0].cpu().numpy()

        if len(valid_path_idx) == 0:
            valid_path_idx = [torch.argmax(path_scores).item()]

        if len(valid_doc_idx) == 0:
            valid_doc_idx = [torch.argmax(doc_scores).item()]

        final_paths = [path_texts[i] for i in valid_path_idx]
        final_docs = [doc_texts[i] for i in valid_doc_idx]

        return final_paths, final_docs
