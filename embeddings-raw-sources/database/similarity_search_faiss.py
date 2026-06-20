import numpy as np
from typing import List, Tuple
import torch
from tqdm import tqdm
import os; os.environ["KMP_DUPLICATE_LIB_OK"]="True"

import faiss                     # pip install faiss-cpu 
faiss.omp_set_num_threads(4) 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


class SimilaritySearchFAISS:
    """
    Thin wrapper around a FAISS IndexLSH locality-sensitive-hash (LSH) index.

    Parameters
    ----------
    dim : int
        Dimensionality of each vector.
    n_bits : int, default=2 * dim
        Number of hyper-plane hash bits.  FAISS recommends 2–4×dim.
    normalize : bool, default=True
        Whether to L2-normalize vectors before adding / querying.
    """

    def __init__(self, dim: int, index_type: str, n_bits: int | None = None, *, normalize: bool = True):
        if index_type == "lsh":
            self.type = "lsh"
        elif index_type == "flat":
            self.type = "flat"
        else:
            print(f"Unsupported index: {index_type}")
            raise
    
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        print("Loading model")
        self.model = SentenceTransformer('thenlper/gte-large', trust_remote_code=True).to(self.device)
        
        print("Done loading model")
        self.dim = dim
        self.n_bits = 2 * dim
        self.normalize = normalize

        # Build the index: IndexLSH supports only float32
        print("Building index...")
        if (self.type == "lsh"):
            print(f"Creating LSH index")
            self.index = faiss.IndexLSH(dim, self.n_bits)
        else:
            print(f"Creating FlatL2 index")
            self.index = faiss.IndexFlatL2(self.dim)
        print("Index built")
        
        # Keep an auxiliary list so we can map FAISS ids → payload (e.g., filenames, docs …)
        self._payload: List = []

    def _prep(self, vecs: np.ndarray) -> np.ndarray:
        """Ensure float32 + optional L2-normalisation (in-place safe)."""
        vecs = vecs.astype(np.float32, copy=False)
        if self.normalize:
            faiss.normalize_L2(vecs)
        return vecs

    @staticmethod
    def stream_vectors(collection, batch=10_000):
        offset = 0
        while True:
            # pull embeddings *and* text + metadata
            chunk = collection.get(
                include=["embeddings", "documents", "metadatas"],
                limit=batch,
                offset=offset,
            )
            if not chunk["ids"]:
                break
            vecs  = np.asarray(chunk["embeddings"], dtype=np.float32)
            docs  = chunk["documents"]
            metas = chunk["metadatas"]
            yield vecs, docs, metas
            offset += batch

    def add(self, collection, batch: int = 20_000):
        for vecs, docs, metas in tqdm(
                self.stream_vectors(collection, batch=batch),
                total=collection.count()):
            faiss.normalize_L2(vecs)
            self.index.add(vecs)
            # keep parallel Python list: [{doc, meta}, …]
            self._payload.extend(
                {"doc": d if d is not None else "", "meta": m or {}}  # guard None
                for d, m in zip(docs, metas)
            )


    def search(self, query: np.ndarray, k: int = 5, exclude_tests : bool = True, max_retrieve: int = 100):
        """
        Return a list of (faiss_id, similarity, payload) tuples.
        """

        search_k = k
        if exclude_tests:
            # apply a factor to k, since we'll be excluding test files
            search_k = max_retrieve

        # Ensure 2-D float32 + optional L2-norm
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = self._prep(query.copy())

        # ---- call the *index*, not this wrapper ----
        distances, indices = self.index.search(query, search_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            sim = 1.0 - dist / 2.0          # convert L2 → cosine for unit vectors
            results.append((int(idx), sim, self._payload[idx]))

        filtered = []
        if exclude_tests:
            # A match is considered a test file if located under test, tests or testing directory.
            for idx, score, payload in results:
                doc_text = payload["doc"]
                meta     = payload["meta"]
                path = meta.get("filename", "").lstrip("/")  # drop any leading slash
                segments = path.split("/")                   # e.g. "pytest/foo" → ["pytest","foo"]

                # skip only if *any* segment is exactly "test", "tests" or "testing"
                if any(seg in {"test","tests","testing"} for seg in segments):
                    continue

                filtered.append((idx, score, payload))
        
        print(f"Found {len(filtered)} results after filtering.")
        print(f"Top k is {k}, max retrieve is {max_retrieve}.")
        return filtered[:k]

    def reset(self):
        """Erase the current index (payload is also cleared)."""
        self.index.reset()
        self._payload.clear()

    def _aggregate_embeddings_mean(self, chunk_embeddings: list[np.ndarray]) -> np.ndarray | None:
        """
        Mean-pool a list of chunk embeddings and return a single, unit-length vector.

        Returns
        -------
        np.ndarray | None
            (dim,) float32 vector or None if the list is empty.
        """
        if not chunk_embeddings:
            return None

        # Stack → mean → float32
        agg = np.mean(np.vstack(chunk_embeddings), axis=0).astype(np.float32)

        # L2-normalise so cosine-sim == 1 – (||a-b||² / 2) inside IndexLSH
        faiss.normalize_L2(agg.reshape(1, -1))
        return agg


    def search_similar_chunks(
        self,
        bug_description: str,
        *,
        k: int = 100,            # how many neighbours to ask FAISS for
        retrieve_max: int = 100, # final cut after filtering
        exclude_tests: bool = True
    ):
        """
        Return up to ``retrieve_max`` code chunks/files most similar to ``bug_description``.

        Each result is a tuple: (payload, similarity_score)

        Notes
        -----
        * We still split the description into 512-token chunks, embed each, and
        **mean-pool** them into a single query vector.
        * Similarity is cosine (FAISS LSH distance → similarity conversion is done
        inside `SimilaritySearch.search`).
        """

        # 1. Split and embed the bug description
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=512,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ">"],
        )
        chunks = splitter.split_text(bug_description)
        chunk_embs = [self.model.encode(txt) for txt in chunks]

        query_vec = self._aggregate_embeddings_mean(chunk_embs)
        if query_vec is None:
            return []

        query_vec = query_vec.reshape(1, -1).astype(np.float32)   # <- **fix**

        faiss_hits = self.search(query_vec, k=k, exclude_tests=exclude_tests, max_retrieve=retrieve_max)        # [(idx, score, payload), …
        
        filtered = faiss_hits

        results   = []
        for idx, score, payload in filtered:
            doc_text = payload["doc"]
            meta     = payload["meta"]
            results.append((doc_text, meta, score))      # same 3-tuple as before
            if len(results) >= retrieve_max:
                break

        return results
