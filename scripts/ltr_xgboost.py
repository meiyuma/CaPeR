import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import xgboost as xgb
import os, sys, json, pickle, argparse
import dt_utils as du
import utils as u



class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def save(self, path, name):
        if not os.path.isdir(path): os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, name + '_ngram_model.pkl')
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        map_location = None if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(model_path, map_location=map_location))


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_vocab(dataset, context_size):
    text = []
    for item in dataset:
        q = f"{item['e1']} {item['e2']}"
        meta = [path for path in item['metapaths']][:5]
        for path in meta:
           
            content = q + ' ' + path['stops']
            text.append(content)
    text = ' '.join(text).lower().split()
    vocab = set(text)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    ngrams = []
    if len(text) > context_size:
        ngrams = [([text[i - j - 1] for j in range(context_size)], text[i]) for i in range(context_size, len(text))]
    return vocab, word_to_ix, ngrams


def get_embedding_features(model, sentence, word_to_ix, device):
    tokens = sentence.lower().split()
    embeddings = []
    for word in tokens:
        if word in word_to_ix:
            idx = torch.tensor([word_to_ix[word]], dtype=torch.long).to(device)
            emb = model.embeddings(idx).detach().cpu().numpy()[0]
            embeddings.append(emb)
    if len(embeddings) == 0: return np.zeros(128)
    return np.mean(np.array(embeddings), axis=0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()

    set_seed(1111)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 128

    print(f"Loading data...")
    train_data = [json.loads(line) for line in open(args.train_path)]
    test_data = [json.loads(line) for line in open(args.test_path)]
    full_data = train_data + test_data

 
    vocab, word_to_ix, ngrams = create_vocab(full_data, CONTEXT_SIZE)
    print(f"Vocab size: {len(vocab)}")
    print(f"Path: {args.checkpoint_path}")

    ngram_model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to(DEVICE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(ngram_model.parameters(), lr=0.001)

    ngram_model_path = os.path.join(args.checkpoint_path, f'{args.task}_ngram_model.pkl')
    u.folder_check(args.checkpoint_path)

    if os.path.exists(ngram_model_path):
        print("Loading existing N-Gram model...")
        ngram_model.load(ngram_model_path)
    else:
        print("Training N-Gram Model...")
        for epoch in range(args.epoch):
            total_loss = 0
            for context, target in tqdm(ngrams, desc=f"Epoch {epoch}"):
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long).to(DEVICE)
                target_t = torch.tensor([word_to_ix[target]], dtype=torch.long).to(DEVICE)
                ngram_model.zero_grad()
                log_probs = ngram_model(context_idxs)
                loss = loss_function(log_probs, target_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch} Loss: {total_loss}")
        ngram_model.save(args.checkpoint_path, args.task)

    
    print("Preparing XGBoost Features...")

    def prepare_features_robust(data_list):
        data_list.sort(key=lambda x: str(x['qid']))
        embs = []
        rels = []
        qids_int = []
        current_qid_str = None
        current_qid_idx = -1

        for item in tqdm(data_list):
            q_str = str(item['qid'])
            if q_str != current_qid_str:
                current_qid_str = q_str
                current_qid_idx += 1

            q_text = f"{item['e1']} {item['e2']}"
            meta = [path for path in item['metapaths']][:5]

            for r, path in enumerate(meta):
                content = q_text + ' ' + path['stops']
                embedding = get_embedding_features(ngram_model, content, word_to_ix, DEVICE)
                embs.append(embedding)
                rels.append(r)
                qids_int.append(current_qid_idx)

        return np.vstack(embs), np.array(rels), np.array(qids_int)

    X_train, y_train, qid_train = prepare_features_robust(train_data)

   
    print(f"Training XGBRanker... Shape: {X_train.shape}")
    ranker = xgb.XGBRanker(
        tree_method="hist",
        lambdarank_num_pair_per_sample=8,
        objective="rank:ndcg",
        lambdarank_pair_method="topk"
    )
    ranker.fit(X_train, y_train, qid=qid_train)
    print("XGBoost Model Saved.")

    
    print("Predicting on Test Set...")
    output_file = os.path.join(args.checkpoint_path, 'test_pred_xgboost.jsonl')

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(test_data):
            q_text = f"{item['e1']} {item['e2']}"
            meta = [path for path in item['metapaths']][:5]

            if not meta:
                item['bk_selected_paths'] = [f"{item['e1']} -> {item['e2']}"]
                f_out.write(json.dumps(item) + "\n")
                continue

            item_embs = []
            for path in meta:
                content = q_text + ' ' + path['stops']
                emb = get_embedding_features(ngram_model, content, word_to_ix, DEVICE)
                item_embs.append(emb)

            X_test_item = np.vstack(item_embs)
            scores = ranker.predict(X_test_item)

            path_score_pairs = []
            for i, path_obj in enumerate(meta):
               
                p_str = path_obj.get('stops', path_obj.get('path', ''))
                path_score_pairs.append({
                    'path_str': p_str,
                    'score': float(scores[i])
                })

            path_score_pairs.sort(key=lambda x: x['score'], reverse=True)
            top_paths = [p['path_str'] for p in path_score_pairs]

            out_obj = item.copy()
            out_obj['bk_selected_paths'] = top_paths
            if 'query' not in out_obj:
                out_obj['query'] = f"Is there a causal relationship between {item['e1']} and {item['e2']}?"

            f_out.write(json.dumps(out_obj) + "\n")

    print(f"Done! Results saved to {output_file}")


if __name__ == "__main__":
    main()
