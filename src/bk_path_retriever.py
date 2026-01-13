import time
from Bio import Entrez
import logging
import random

class PathGuidedRetriever:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PathRetriever")

    def _fetch_text(self, query, retries=3):
        for attempt in range(retries):
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=1, sort="relevance")
                record = Entrez.read(handle)
                handle.close()

                id_list = record["IdList"]
                if not id_list:
                    return None

                handle = Entrez.efetch(db="pubmed", id=id_list[0], rettype="abstract", retmode="text")
                text = handle.read().strip()
                handle.close()

                text = text.replace("\n", " ")
                if len(text) > 800: text = text[:800] + "..." 
                return text

            except Exception as e:
                wait_time = random.uniform(1, 3)
                self.logger.warning(f"Error fetching {query}: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        return None

    def search_pubmed_snippet(self, term_a, term_b):
       
        query = f"{term_a} AND {term_b}"
        return self._fetch_text(query)

    def search_pubmed_custom(self, custom_query):
        
        clean_query = custom_query.replace('"', '').replace("'", "")
        return self._fetch_text(clean_query)

    def retrieve_chain_evidence(self, path_list):
       
        return "Legacy method, use Agent loop instead."
