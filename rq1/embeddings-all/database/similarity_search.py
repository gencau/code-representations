from pathlib import Path
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import utils.file_utils as utils
from database.storage import Storage
from code.code_chunk import parse_python_text
from llm.prompts import getDocstringSummaryPrompt, generateDoctringPrompt
from llm.groq_api import GroqClient

class SimilaritySearch:
    def __init__(self, path, collection_name):
        # instance of the database
        self.storage_instance = Storage(path=path, collection_name=collection_name)

        # Embedding model
        self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        self.collection_name = collection_name
        # Using Groq for now as local inference is way too slow
        self.llm_client = GroqClient()

        self.python_separators = [
            # 1. Split CLASSES first
            r'\n(?=\s*class )',  # Split before classes  
            # 2. Split FUNCTIONS next
            r'\n(?=\s*(def |@|async def ))',  # Split before functions
            # 3. Keep docstrings with their code blocks
            r'(?<=""")\s*\n+',  # Split after docstrings (now safe)
            r"(?<=''')\s*\n+",
            # 4. Then other splits
            r'\n\n+',
            r'\n',
            r' ',
            ''
        ]

    def createOrSetCollection(self):
        try:
            self.storage_instance.create_collection()
        except:
            self.storage_instance.set_collection()

    def collectionExists(self) -> bool:
        return self.storage_instance.collection_exists()
    
    def generateDocstring(self, docstring : str, text : str) -> str:
        if not docstring:
            # Ask LLM to generate a docstring
            prompt = generateDoctringPrompt(text)
        else:
            # Ask LLM to summarize the docstring (is this required?)
            prompt = getDocstringSummaryPrompt(docstring)

        try:
            docstring = self.llm_client.formatAndSend(prompt)
        except RuntimeError as e:
            print("Request failed after retries:", str(e))
        
        return docstring


    # Generate embeddings for a file (eventually for each file in the repo)
    def processRecord(self, output_dir, batch_size=32):
        starting_id = 1

        # Initialize lists outside the loop to accumulate data across all files
        file_texts = []
        files_embeddings = []
        ids = []
        files_metadata = []
        
        # Now we need to generate embeddings for each file in the repo snapshot 
        directory = Path(output_dir)

        for file_path in directory.rglob('*.py'):
            if not file_path.is_file():
                continue  # Skip directories
            print(f"Processing file {file_path} extracted from repo {directory.name}")

            if file_path.suffix == '.py':
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        file_content = f.read()
                    except:
                        continue

                    if len(file_content) == 0:
                        continue

                    # Extract AST from chunks to get: docstrings, functions, classes
                    all_code_chunks = parse_python_text(file_content, file_path)

                    # Skip if empty file (ex: __init__.py)
                    if len(all_code_chunks) == 0:
                        continue

                    chunks_w_docstrings = []
                    for chunk in all_code_chunks:

                        # In case chunk is still too large
                        text_splitter = RecursiveCharacterTextSplitter (chunk_size=2048,
                                                                        chunk_overlap=500,
                                                                        separators=self.python_separators,
                                                                        length_function=len)
                        texts = text_splitter.split_text(chunk.code)

                        # Doesn't make sense to generate docstrings for all pieces
                        # of a function since we don't know where it's split, so
                        # generate just for the function overall                        
                        docstring = self.generateDocstring(chunk.docstring, texts[0])
                        # Remove extra token that the LLM sometimes generates
                        docstring = docstring.replace('```','')
                        docstring = docstring.replace('python', '')

                        # Make sure docstring is within triple double quotes (LLM doesn't always listen)
                        docstring  = '"""' + docstring + '"""'
                        if docstring:
                            if not chunk.docstring:
                                # insert docstring at start
                                texts[0] = '\n' + docstring + '\n' + texts[0]
                            else:
                                # replace docstring
                                texts[0].replace(chunk.docstring, docstring)
                            chunk.docstring = docstring
                        else:
                            chunk.docstring = ""

                        chunks_w_docstrings.append(''.join(texts))
                    

                    # Now put all chunks back together before splitting again
                    content_w_docstrings = ''.join(chunks_w_docstrings)
                    text_splitter = RecursiveCharacterTextSplitter (chunk_size=1024,
                                                                    chunk_overlap=100,
                                                                    separators=self.python_separators,
                                                                    length_function=len)
                    texts = text_splitter.split_text(content_w_docstrings)


                    for text in texts:
                        filename = utils.get_path_within_repo(output_dir, str(file_path))
                        file_metadata = {
                            "filename": filename,
                            "directory": str(file_path.parent),
                            "chunk_type": "text",
                            "chunk_name": str(file_path),
                            "docstring": ""
                        }

                        file_texts.append(text)
                        files_metadata.append(file_metadata)
                        ids.append("ID_" + str(starting_id))
                        starting_id += 1
            else:
                # For other types of files
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4096,
                    chunk_overlap=0,
                    separators=["\n\n", "\n", " ", ">"],
                    length_function=len
                )
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        if not content.strip():
                            continue
                            
                        # Split text using recursive splitter
                        text_chunks = text_splitter.split_text(content)
                        
                        for chunk in text_chunks:
                            filename = utils.get_path_within_repo(output_dir, str(file_path))
                            metadata = {
                                "filename": filename,
                                "directory": str(file_path.parent),
                                "chunk_type": "text",
                                "chunk_name": file_path.name,
                                "docstring": ""
                            }
                            
                            file_texts.append(chunk)
                            files_metadata.append(metadata)
                            ids.append(f"ID_{starting_id}")
                            starting_id += 1

                except UnicodeDecodeError:
                    print(f"Skipping binary/non-text file: {file_path}")
                    continue

        print("Now creating embeddings in batches...")
        # Create embeddings in batches
        for i in tqdm(range(0, len(file_texts), batch_size), desc="Encoding Batches"):
            batch_texts = file_texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(batch_texts)
                files_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch starting at index {i}: {e}")
                # Optionally, handle or log the error, skip the batch, etc.
                files_embeddings.extend([None] * len(batch_texts))
            
        print("Done creating embeddings. Storing into database...")
        if len(file_texts) > 0:
            self.storage_instance.add(
                texts=file_texts,
                embeddings=files_embeddings,
                metadatas=files_metadata,
                ids=ids
            )
            print(f"Files embeddings for repository {directory.name} added to storage")
        else:
            print("-------- No chunks generated, maybe a parse error?")

    def search_similar_chunks(self, bug_description):
        # Split the bug description into chunks (not sure I need to do this)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096,
                                                       chunk_overlap=0,
                                                       separators=["\n\n", "\n", " ", ">"],
                                                       length_function=len)
        texts = text_splitter.split_text(bug_description)

        # Extract links as metadata
        metadata = []
        for text in texts:
            metadata.append({"text":text,"filename":"", "directory":""})

        # Generate embeddings for each chunk of the bug description
        bug_description_embeddings = [self.model.encode(text) for text in texts]

        # Perform a similarity search in the database
        results = []
        for i, text_embedding in enumerate(bug_description_embeddings):
            text_results = self.storage_instance.similarity_search_with_score(text_embedding)
            results.extend(text_results)

        return results
    
    def cleanup(self):
        # For now, cleanup when exiting
        self.storage_instance.cleanup()