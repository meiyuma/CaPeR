import torch
from torch.utils.data import Dataset
import copy

def verbalize_path_string(path_str):
    
    rel_map = {
        "treats": "treats",
        "palliation": "palliates",
        "causes": "causes",
        "leads to": "leads to",
        "associated": "is associated with",
        "binds": "binds to",
        "upregulates": "upregulates",
        "downregulates": "downregulates",
        "interacts": "interacts with",
        "resembles": "resembles",
        "expresses": "expresses",
        "regulates": "regulates",
        "includes": "includes",
        "presents": "presents",
        "participates": "participates in"
    }

    parts = path_str.split(' - ')
    if len(parts) < 2: return path_str + "."

    sentence_parts = []
    for token in parts:
        lower_token = token.lower()
        if lower_token in rel_map:
            sentence_parts.append(rel_map[lower_token])
        else:
            sentence_parts.append(token)

    text = " ".join(sentence_parts)

    text = text[0].upper() + text[1:] if text else text
    if not text.endswith('.'):
        text += "."

    return text


def verbalize_with_labels(nodelabels_str, stops_str):
    labels = nodelabels_str.split(' - ')
    nodes = stops_str.split(' - ')

    if len(labels) != len(nodes):
        return verbalize_path_string(stops_str)

    sentence = f"{labels[0]} {nodes[0]}"
    for i in range(1, len(nodes)):
        sentence += f" is linked to {labels[i]} {nodes[i]}"

    return sentence + "."


class RankingData(Dataset):
    def __init__(self, data, tokenizer, neg_num=20, include_node_labels=False, include_rel_types=False):
        self.data = data
        self.tokenizer = tokenizer
        self.neg_num = neg_num
        self.include_node_labels = include_node_labels
        self.include_rel_types = include_rel_types

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        query = f"{item['e1']} {item['e2']}"
        meta = [path for path in item['metapaths']][:self.neg_num]
        neg = []

        for path in meta:

            if self.include_node_labels and 'nodelabels' in path:
                content = verbalize_with_labels(path['nodelabels'], path['stops'])
            else:
                content = verbalize_path_string(path['stops'])
            neg.append(content)

    
        neg = neg + ['<padding_passage>'] * (self.neg_num - len(neg))

        
        rel_score = [float(path.get('rel_score', 0.0)) for path in item['metapaths']][:self.neg_num]
        rel_score = rel_score + [0.0] * (self.neg_num - len(rel_score))

        
        evidence_weight = [float(path.get('evidence_weight', 1.0)) for path in item['metapaths']][:self.neg_num]
        evidence_weight = evidence_weight + [0.0] * (self.neg_num - len(evidence_weight))

        passages = neg

       
        return {
            "query": [query] * len(passages),
            "passages": passages,
            "rel_score": rel_score,
            "evidence_weight": evidence_weight
        }

    def collate_fn(self, data):
       
        query = sum([x['query'] for x in data], [])
        passages = sum([x['passages'] for x in data], [])
        rel_score = sum([x['rel_score'] for x in data], [])
        evidence_weight = sum([x['evidence_weight'] for x in data], [])

        # Tokenize
        features = self.tokenizer(query, passages, padding=True, truncation=True, return_tensors="pt", max_length=500)

        # Convert to Tensor
        features['rel_score'] = torch.Tensor(rel_score)
        features['evidence_weight'] = torch.Tensor(evidence_weight)

        return features


class RankingDataGPT(Dataset):
    def __init__(self, data, neg_num=20, include_node_labels=False, include_rel_types=False):
        self.data = data
        self.neg_num = neg_num
        self.include_node_labels = include_node_labels
        self.include_rel_types = include_rel_types

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        query = f"{item['e1']} {item['e2']}"
        meta = [path for path in item['metapaths']][:self.neg_num]
        neg = []
        for path in meta:
            if self.include_node_labels and 'nodelabels' in path:
                content = verbalize_with_labels(path['nodelabels'], path['stops'])
            else:
                content = verbalize_path_string(path['stops'])
            neg.append(content)
        return query, neg


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response: new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['metapaths'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['metapaths'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rel_score' in item['metapaths'][j + rank_start]:
            item['metapaths'][j + rank_start]['rel_score'] = cut_range[j]['rel_score']
    return item


def write_file(rank_results, file, is_truth=True):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            metapaths = rank_results[i]['metapaths']
            for meta in metapaths:
                if is_truth:
                    f.write(f"{rank_results[i]['qid']} 0 {meta['pathid']} {rank}\n")
                else:
                    f.write(f"{rank_results[i]['qid']} Q0 {meta['pathid']} {rank} {meta['rel_score']} rank\n")
                rank += 1
        return True
