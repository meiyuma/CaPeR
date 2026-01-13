import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BKAgent:
    def __init__(self, model_path, device="cuda:1"):
        print(f"🤖 Initializing Llama-3 Agent from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            local_files_only=True
        )
        self.device = device
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def _chat_generate(self, messages, max_len=128):
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, max_new_tokens=max_len,
                eos_token_id=self.terminators,
                do_sample=True, temperature=0.6, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True).strip()

    def verify_evidence(self, path_desc, evidence):
        
        if not evidence or len(evidence) < 10:
            return False, "No valid evidence found."

        system_msg = "You are a biomedical fact-checker."
        user_msg = (
            f"Claim: \"{path_desc}\"\n"
            f"Document: \"{evidence[:800]}\"\n\n"
            "Does the Document support the Claim?\n"
            "Answer YES only if the document explicitly mentions a relationship.\n"
            "Answer NO if it is irrelevant.\n"
            "Format:\nVerdict: [YES/NO]\nReason: [Short explanation]"
        )
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        output = self._chat_generate(messages, max_len=128)
        is_supported = "YES" in output.upper().split("\n")[0]
        return is_supported, output

    
    def rewrite_query(self, head, tail, feedback, global_context=""):
        
        system_msg = "You are a search engine expert for PubMed."

        
        context_hint = f"Context: This is part of a study on '{global_context}'." if global_context else ""

        user_msg = (
            f"Task: Find connection between \"{head}\" and \"{tail}\".\n"
            f"{context_hint}\n"
            f"Previous search failed. Reason: {feedback}\n\n"
            "Generate ONE refined PubMed search query.\n"
            "Strategy: Use specific medical mechanisms or synonyms relevant to the Context.\n"
            "Output ONLY the query string."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        new_query = self._chat_generate(messages, max_len=64)
        new_query = new_query.replace("Query:", "").replace("\"", "").strip()
        return new_query
