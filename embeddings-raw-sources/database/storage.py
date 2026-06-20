import shutil
import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.errors import NotFoundError

class Storage:
    def __init__(self, path, collection_name):
        self.chroma_client = chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True))
        self.collection_name = collection_name
        self.collection = None

    def create_collection(self):
        self.collection = self.chroma_client.create_collection(name=self.collection_name,
                                                               metadata={"hnsw:space":"cosine"}, 
                                                               get_or_create=True
                                                               )
    def set_collection(self):
        self.collection = self.chroma_client.get_collection(self.collection_name)
        
    def collection_exists(self) -> bool:
        try:
            # Attempt to get the collection
            collection = self.chroma_client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' exists.")
            return True
        except:
            return False

    def get_chroma_client(self):
        return self.chroma_client
    
    def add(self, texts, embeddings, metadatas, ids):
        self.collection.add(documents=texts,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            ids=ids)

    # Returns the top k most similar documents to the query_text
    def query(self, query_embeddings, top_k=5):
        return self.collection.query(query_texts=query_embeddings, n_results=top_k, 
                                     include=["documents", "metadatas", "distances"]
                                    )
    
    def similarity_search_with_score(self, query_embedding, k=5, metadata_filter=None, exclude_tests=True, retrieve_max = 0):
        search_k = k
        if exclude_tests:
            # apply a factor to k, since we'll be excluding test files
            search_k = retrieve_max

        # Prepare the query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": search_k, 
            "include": ["documents", "metadatas", "distances"]
        }        
        if metadata_filter:
            # ChromaDB expects filters to be under the 'where' key
            query_params["where"] = metadata_filter
        
        # Perform the query
        results = self.collection.query(**query_params)
        
        # Extract documents, metadatas, and distances
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        

        # Combine the results into a list of tuples
        combined_results = list(zip(documents, metadatas, distances))

        print(f"Found {len(combined_results)}.")
        
        ## Note: this is already accounted for in the initial dataset filtering
        """
        filtered = []
        if exclude_tests:
            # A match is considered a test file if located under test, tests or testing directory.
            for doc, meta, dist in combined_results:
                path = meta.get("filename", "").lstrip("/")  # drop any leading slash
                segments = path.split("/")                   # e.g. "pytest/foo" → ["pytest","foo"]

                # skip only if *any* segment is exactly "test", "tests" or "testing"
                if any(seg in {"test","tests","testing"} for seg in segments):
                    continue

                filtered.append((doc, meta, dist))
        else:
            filtered = combined_results

        print(f"Found {len(filtered)} results after filtering.")
        print(f"Top k is {k}, max retrieve is {retrieve_max}.")

        if len(filtered) < 1:
            print(">>>>>>>> WARNING: Filtered is 0!!!")

        return filtered[:k]
        """
        return combined_results[:k]

    
    def fetch_all(self, batch=10_000):
        offset, ids, vecs = 0, [], []
        while True:
            chunk = self.collection.get(
                include=["embeddings"],   # or ["embeddings", "metadatas"] if you need meta
                limit=batch,
                offset=offset,
            )
            if not chunk["ids"]:                      # done
                break
            ids.extend(chunk["ids"])
            vecs.append(np.asarray(chunk["embeddings"], dtype=np.float32))
            offset += batch

        if not vecs:                               # <- nothing fetched
            print(f"WARNING!!!!! No vectors returned for collection {self.collection_name}")
            return [], np.empty((0, self.dim), dtype=np.float32)
    
        return ids, np.vstack(vecs)
    
    def cleanup(self):
        self.chroma_client.reset()
        shutil.rmtree("chroma_db")


