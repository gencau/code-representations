from typing import Dict, Any
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from src.baselines.backbones.base_backbone import BaseBackbone

from pyserini.search import LuceneSearcher

class BM25Backbone(BaseBackbone):

    def __init__(self,
                 name: str,
                 data_path: str,
                 topk: int,
                 k1: int,
                 b : int,
                 **kwargs):
        self.name = name
        self._data_path = data_path
        self._topk = topk
        self._k1 = k1
        self._b = b

    def _search_index(self, query: str, index_dir: str, k1, b):
        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(k1=k1, b=b) 
        hits = searcher.search(query, k=self._topk)

        files = []
        scores = []

        print(f"Top {self._topk} results for '{query}':")
        for i, hit in enumerate(hits):
            doc = searcher.doc(hit.docid)
            doc = hit.lucene_document.get('raw')
            doc_dict = json.loads(doc)
            print(f"{i+1:2d}. {hit.docid:4}\n Document: {doc_dict.get('path')}\n(score: {hit.score:.4f})")
            files.append(doc_dict.get('path'))
            scores.append(hit.score)
        return files, scores

    def _iterate_on_context(self, issue : str, repo_content: Dict[str, str], repo_dir: str, k1, b):

        # Pass issue description, paths and file content
        predicted_files, scores = self._search_index(issue, repo_dir + "/indexes", k1, b)

        return predicted_files, scores
    

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str], 
                      repo_dir: str, k1, b, **kwargs) -> Dict[str, Any]:

        retrieved_files, scores = self._iterate_on_context(issue_description, repo_content, repo_dir, k1, b)

        return {
            "final_files": list(retrieved_files),
            "rank_scores": list(scores)
        }
