# CaPeR
CaPeR:Mitigating Hallucinations in Biomedical Causal Reasoning

Please download large models such as llama3-8b-instruct first and then place them in the models folder.
### Structure Learning
This step is intended to enable the Ranker to train a better model by leveraging "path narrativization" and "negative samples".

```bash
bash run_ltr.sh ade
rm *.pkl
rm test_pred_xgboost.jsonl
```
### Agentic Retrieval

This step leverages PubMedBERT to perform path-guided retrieval, verification, and retry operations.

```bash
python src/bk_run_agent_retrieval.py \
    --input_file src/ptc/checkpoints/ade/test_pred_ensemble.jsonl \
    --output_file data/ade/final_verified_llama3.jsonl
```

### Inference & Evaluation

This step calculates the final F1-score, Precision, and Recall.

```bash
python src/bk_generate_eval.py \
    --input_file data/ade/final_verified_llama3.jsonl \
    --truth_file data/ade/test_truth.jsonl \
    --output_score data/ade/score_final_llama3.txt \
    --output_explanation data/ade/explanation_final_llama3.jsonl \
```


### Visualization

explanation_final_llama3.jsonl

