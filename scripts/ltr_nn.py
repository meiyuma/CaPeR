import json, sys, logging
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AdamW
import torch
from tqdm import tqdm
from rank_loss import RankLoss
import numpy as np
import os
import argparse
import dt_utils as du
import ltr_eval as leval
import utils as u

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
LOCAL_MODEL_PATH = "models/roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=LOCAL_MODEL_PATH)
    parser.add_argument('--loss_type', type=str, default='rank_net')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--neg_num', type=int, default=5)
    parser.add_argument('--include_node_labels', type=bool, default=False)
    parser.add_argument('--include_rel_types', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=False)
    args = parser.parse_args()
    return args


def train(args):
    accelerator = Accelerator()
    neg_num = args.neg_num

    print(f"Loading model from {args.model_name}...")
    config = AutoConfig.from_pretrained(args.model_name, local_files_only=True)
    config.num_labels = 1

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, local_files_only=True)

    data = [json.loads(line) for line in open(args.train_path)]
    dataset = du.RankingData(data, tokenizer, neg_num=neg_num, include_node_labels=args.include_node_labels,
                             include_rel_types=args.include_rel_types)
    print("Sample data:", dataset[0])

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=1, shuffle=True,
                                              num_workers=0)
    optimizer = AdamW(model.parameters(), 5e-5)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    loss_function = getattr(RankLoss, args.loss_type)

    best_acc = 0.0
    best_all = {}
    best_run = 0

    for epoch in range(args.epoch):
        accelerator.print(f'Training epoch: {epoch}')
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            out = model(batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = out.logits
            
            logits = logits.view(-1, neg_num)

           
            y_true = batch['rel_score']
            y_true = y_true.view(-1, neg_num) 

            
            evidence_weights = batch.get('evidence_weight', None)
            if evidence_weights is not None:
                evidence_weights = evidence_weights.view(-1, neg_num)

            if args.loss_type in ['list_net', 'rank_net']:
                loss = loss_function(logits, y_true, weights=evidence_weights)
            else:
                loss = loss_function(logits, y_true)


            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()

            loss_report.append(loss.item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))

        # Eval
        all_metrics = eval(args, epoch, model, tokenizer)
        logging.info(f"{epoch=} {all_metrics=} \n")
        ep_acc = all_metrics['NDCG@5']

        if ep_acc > best_acc:
            best_acc = ep_acc
            best_all = all_metrics
            best_run = epoch
            logging.info(f"Congratulations! New best accuracy: {best_acc}")

            print(f"Saving model to {args.checkpoint_path}...")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.checkpoint_path)
                tokenizer.save_pretrained(args.checkpoint_path)

    logging.info(f"# BEST: {best_acc} # All: {best_all} # Epoch: {best_run}\n")
    return model, tokenizer, best_all


def eval(args, epc, model=None, tokenizer=None):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, local_files_only=True)
        model = model.cuda()

    model.eval()
    data = [json.loads(line) for line in open(args.test_path)]
    du.write_file(data, args.checkpoint_path + '/truth.eval', is_truth=True)
    pred_ranked = f"{args.checkpoint_path}/test_pred_ep{str(epc)}.jsonl"

    with open(pred_ranked, 'w', encoding='utf-8') as out_file:
        reranked_data = []
        for item in tqdm(data):
            q = f"{item['e1']} {item['e2']}"
            passages = [psg['stops'] for i, psg in enumerate(item['metapaths'])][:5]
            if len(passages) == 0:
                reranked_data.append(item)
                continue

            
            verbalized_passages = [du.verbalize_path_string(p) for p in passages]

            features = tokenizer([q] * len(verbalized_passages), verbalized_passages, padding=True, truncation=True,
                                 return_tensors="pt",
                                 max_length=500)
            features = {k: v.cuda() for k, v in features.items()}
            with torch.no_grad():
                scores = model(features['input_ids'], attention_mask=features['attention_mask']).logits
                normalized_scores = [float(score[0]) for score in scores]
            ranked = np.argsort(normalized_scores)[::-1]
            response = ' > '.join([str(ss + 1) for ss in ranked])
            ranked_result = du.receive_permutation(item, response, rank_start=0, rank_end=5)

            
            top_paths = []
            if 'metapaths' in ranked_result:
                for mp in ranked_result['metapaths'][:5]:
                    # 优先取 stops，没有则取 path
                    top_paths.append(mp.get('stops', mp.get('path', '')))

            ranked_result['bk_selected_paths'] = top_paths

            
            if 'query' not in ranked_result:
                ranked_result['query'] = f"Is there a causal relationship between {item['e1']} and {item['e2']}?"

            reranked_data.append(ranked_result)
            jout = json.dumps(ranked_result) + '\n'
            out_file.write(jout)

        pred_out = f"{args.checkpoint_path}/pred.eval"
        du.write_file(reranked_data, pred_out, is_truth=False)

    all_metrics = leval.run_eval(args.checkpoint_path + '/truth.eval', pred_out)
    return all_metrics


if __name__ == '__main__':
    args = parse_args()
    u.folder_check(args.checkpoint_path)
    log_file = args.checkpoint_path + "/log.log"
    with open(args.checkpoint_path + '/params.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%m/%d/%Y %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, 'w+'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logging.info('====Input Arguments====')
    logging.info(json.dumps(vars(args), indent=2, sort_keys=False))

    if args.do_train:
        train(args)
