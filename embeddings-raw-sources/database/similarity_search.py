from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

import utils.file_utils as utils
from database.storage import Storage

class SimilaritySearch:
    def __init__(self, path, collection_name):
        # instance of the database
        self.storage_instance = Storage(path=path, collection_name=collection_name)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        #self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True).to(self.device)
        #self.model = SentenceTransformer('Salesforce/SFR-Embedding-Code-400M_R', trust_remote_code=True)
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        #self.model = SentenceTransformer('thenlper/gte-large')
        #self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        #self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Code-400M_R")
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        #self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")

        self.collection_name = collection_name

    def createOrSetCollection(self):
        try:
            self.storage_instance.create_collection()
        except:
            self.storage_instance.set_collection()

    def collectionExists(self) -> bool:
        return self.storage_instance.collection_exists()
    
    # Generate embeddings for a file (eventually for each file in the repo)
    def processRecord(self, output_dir, language="py", batch_size=16):
        starting_id = 1

        # Initialize lists outside the loop to accumulate data across all files
        file_texts = []
        files_embeddings = []
        ids = []
        files_metadata = []
        
        # Now we need to generate embeddings for each file in the repo snapshot 
        directory = Path(output_dir)

        if (language == "py"):
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, chunk_size=512,
                                                           chunk_overlap=0,
                                                           separators=RecursiveCharacterTextSplitter.get_separators_for_language("python"))
        elif (language == "java"):
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, 
                                                           chunk_size=512,
                                                           chunk_overlap=0,
                                                           separators=RecursiveCharacterTextSplitter.get_separators_for_language("java"))
        elif (language == "kt"):
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, 
                                                           chunk_size=512,
                                                           chunk_overlap=0,
                                                           separators=RecursiveCharacterTextSplitter.get_separators_for_language("kotlin"))
        
        for file_path in directory.rglob(f'*.{language}'):
            if not file_path.is_file() or utils.is_test_file(str(file_path)):
                print(f"Skipping test file or directory: {file_path}")
                continue  # Skip directories and test files
            print(f"Processing file {file_path} extracted from repo {directory.name}")

            if file_path.suffix == f'.{language}':
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        file_content = f.read()
                    except:
                        continue

                    if len(file_content) == 0:
                        continue

                    all_code_chunks = text_splitter.split_text(file_content)

                    # Skip if empty file (ex: __init__.py)
                    if len(all_code_chunks) == 0:
                        continue

                    filename = utils.get_path_within_repo(output_dir, str(file_path))
                    
                    for chunk in all_code_chunks:
                        file_texts.append(chunk)
                        file_metadata = {
                            "filename":filename,
                            "directory":str(file_path.parent),
                        }

                        files_metadata.append(file_metadata)
                        ids.append("ID_" + str(starting_id))
                        starting_id += 1
        
        print("Now creating embeddings in batches...")
        # Create embeddings in batches
        for i in tqdm(range(0, len(file_texts), batch_size), desc="Encoding Batches"):
            batch_texts = file_texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                files_embeddings.extend(batch_embeddings)

                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception as e:
                print(f"Error encoding batch starting at index {i}: {e}")
                # Optionally, handle or log the error, skip the batch, etc.
                files_embeddings.extend([np.zeros(self.model.get_sentence_embedding_dimension())] * len(batch_texts))
            
        print("Done creating embeddings. Storing into database...")

        if len(file_texts) > 0:
            # chroma_db has a max batch size, but let's aim smaller so we don't fill the temp directory
            max_batch_size = 1024
            for i in range(0, len(file_texts), max_batch_size):
                batch_texts = file_texts[i:i + max_batch_size]
                batch_embeddings = files_embeddings[i:i + max_batch_size]
                batch_metadata = files_metadata[i:i + max_batch_size]
                batch_ids = ids[i:i + max_batch_size]

                self.storage_instance.add(texts=batch_texts,
                                          embeddings=batch_embeddings,
                                          metadatas=batch_metadata,
                                          ids=batch_ids)

            print(f"Files embeddings for repository {directory.name} added to storage")
        else:
            print("-------- No chunks generated, maybe a parse error?")

    def _aggregate_embeddings_mean(self, chunk_embeddings):
        """
        Aggregate embeddings using mean pooling.
        
        Args:
            chunk_embeddings (list of np.array): List of embeddings for each chunk.
            
        Returns:
            np.array: Aggregated embedding.
        """
        if len(chunk_embeddings) == 0:
            return None
        aggregated_embedding = np.mean(chunk_embeddings, axis=0)
        return aggregated_embedding

    def search_similar_chunks(self, bug_description, k=100, retrieve_max = 100):
        # Split the bug description into chunks (not sure I need to do this)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, 
                                                       chunk_size=512,
                                                       chunk_overlap=0,
                                                       separators=["\n\n", "\n", " ", ">"])
        texts = text_splitter.split_text(bug_description)

        bug_description_embeddings = [self.model.encode(text) for text in texts]
        print(bug_description_embeddings)
        aggregated_bug_description_embedding = self._aggregate_embeddings_mean(np.array(bug_description_embeddings))

        query_embeddings = np.array([aggregated_bug_description_embedding])
        print(f"Aggregated bug description embedding: {aggregated_bug_description_embedding}")
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = query_embeddings.reshape(1, -1)
        else:
            query_embeddings = [query_embeddings]
        
        # Perform a similarity search in the database
        all_results = []
        
        for i, text_embedding in enumerate(query_embeddings):
            # using a buffer to account for the filter
            text_results = self.storage_instance.similarity_search_with_score(text_embedding, k=k, exclude_tests=True, retrieve_max=retrieve_max)
            all_results.extend(text_results)

        # Sort the results by distance
        all_results.sort(key=lambda x: x[2])
            
        return all_results
    
    def cleanup(self):
        # For now, cleanup when exiting
        self.storage_instance.cleanup()
