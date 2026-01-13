
import json
import time
import os
import requests  
from Bio import Entrez
from tqdm import tqdm


YOUR_EMAIL = ""  
YOUR_API_KEY = ""  

Entrez.email = YOUR_EMAIL.strip()
Entrez.api_key = YOUR_API_KEY.strip()


def get_entities_from_file(filepath):
    entities = set()
    print(f"Reading entities from {filepath}...")
    with open(filepath, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'e1' in item and item['e1']: entities.add(item['e1'].strip())
                if 'e2' in item and item['e2']: entities.add(item['e2'].strip())
            except:
                pass
    return list(entities)


def search_pubmed_ids(term, api_key, max_results=5):
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": term,  
        "retmode": "json",
        "retmax": max_results,
        "api_key": api_key
    }

    try:
        
        response = requests.get(base_url, params=params, timeout=10)

        
        if response.status_code == 400:
            print(f"\n[ERROR 400] Bad Request. 请检查 API Key: {api_key[:5]}*** 是否正确？")
            return []

        response.raise_for_status()
        data = response.json()

        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            return data["esearchresult"]["idlist"]
        return []

    except Exception as e:
        print(f"\n[Search Error] {term}: {e}")
        return []


def fetch_abstracts(entities, output_file, max_results=5):
    
    if "YOUR_" in Entrez.api_key:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("错误: 你还没有在 src/fetch_pubmed.py 中替换真实的 API KEY！")
        print("请打开文件修改 YOUR_API_KEY 变量。")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    existing_pmids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'pmid' in d: existing_pmids.add(str(d['pmid']))
                except:
                    pass

    print(f"Found {len(existing_pmids)} existing articles.")
    f_out = open(output_file, 'a', encoding='utf-8')

    for entity in tqdm(entities, desc="Fetching PubMed"):
        if len(entity) < 2: continue
        new_ids = search_pubmed_ids(entity, Entrez.api_key, max_results)
        target_ids = [pid for pid in new_ids if str(pid) not in existing_pmids]

        if not target_ids:
            time.sleep(0.2)  
            continue
        try:
            handle = Entrez.efetch(db="pubmed", id=target_ids, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            if 'PubmedArticle' in records:
                for article in records['PubmedArticle']:
                    pmid = str(article['MedlineCitation']['PMID'])
                    if pmid in existing_pmids: continue

                    article_data = article['MedlineCitation']['Article']
                    title = article_data.get('ArticleTitle', '')

                    abstract_raw = article_data.get('Abstract', {}).get('AbstractText', [])
                    if isinstance(abstract_raw, list):
                        abstract = " ".join(abstract_raw)
                    else:
                        abstract = str(abstract_raw)

                    if abstract:
                        entry = {
                            "pmid": pmid,
                            "title": title,
                            "text": abstract,
                            "query_entity": entity
                        }
                        f_out.write(json.dumps(entry) + "\n")
                        f_out.flush()
                        existing_pmids.add(pmid)

            time.sleep(0.4)  

        except Exception as e:
            print(f"\n[Fetch Error] {entity}: {e}")
            continue

    f_out.close()
    print("Fetching complete.")


if __name__ == "__main__":
    BASE_DIR = "/mnt/disk/mameiyu_user/CP/BK"
    input_data = os.path.join(BASE_DIR, "data/ade/test_truth.jsonl")
    output_data = os.path.join(BASE_DIR, "data/corpus/raw_pubmed.jsonl")

    os.makedirs(os.path.dirname(output_data), exist_ok=True)

    entities = get_entities_from_file(input_data)
    fetch_abstracts(entities, output_data, max_results=5)
