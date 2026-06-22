from pathlib import Path
from tqdm import tqdm
import os

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
    def __init__(self, collection_name):
        # instance of the database
        self.storage_instance = Storage(collection_name=collection_name)

        # Embedding model (heard that GTE is pretty good for code search)
        self.model = SentenceTransformer('thenlper/gte-large')
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
            prompt = getDocstringSummaryPrompt(docstring)

        try:
            docstring = self.llm_client.formatAndSend(prompt)
        except RuntimeError as e:
            print("Request failed after retries:", str(e))
        
        print(f"LLM returned: {docstring}")
        return docstring


    # Generate embeddings for a file (eventually for each file in the repo)
    def processRecord(self, output_dir, batch_size=32):
        """
            Processes a record by splitting it into chunks and storing them
            in a vector database.
            Splits are currently done using AST for python.
            TODO: Investigate if only using recursive text splitter would be
            best (would create bigger chunks of context).
        """
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

                    for chunk in all_code_chunks:
                        # Further divide the code chunk, if too large
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096,
                                                       chunk_overlap=0,
                                                       separators=self.python_separators,
                                                       length_function=len)
                        texts = text_splitter.split_text(chunk.code)

                        for text in texts:
                            docstring = self.generateDocstring(chunk.docstring, text)
                            # Remove extra token that the LLM sometimes generates
                            docstring = docstring.replace('```','')
                            docstring = docstring.replace('python', '')

                            # Make sure docstring is within triple double quotes (LLM doesn't always listen)
                            docstring  = '"""' + docstring + '"""'
                            if docstring:
                                if not chunk.docstring:
                                    # insert docstring at start
                                    text = docstring + '\n' + text
                                else:
                                    # replace docstring
                                    text.replace(chunk.docstring, docstring)
                                chunk.docstring = docstring
                            else:
                                chunk.docstring = ""

                            file_metadata = {
                                "filename": os.path.basename(str(file_path)),
                                "chunk_type": chunk.type,
                                "chunk_name": chunk.name,
                                "docstring": chunk.docstring
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
                            metadata = {
                                "filename": os.path.basename(str(file_path)),
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
                batch_embeddings = self.model.encode(batch_texts, batch_size=batch_size, show_progress_bar=True)
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

    def search_similar_chunks(self, bug_description, output_dir):
        # Split the bug description into chunks (not sure I need to do this)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096,
                                                       chunk_overlap=0,
                                                       separators=["\n\n", "\n", " ", ">"],
                                                       length_function=len)
        texts = text_splitter.split_text(bug_description)

        # Extract links as metadata
        # TODO: Here we could also extract paths/filenames (provided they are valid).
        metadata = []
        provided_filepath = ""
        paths_list = []
        for text in texts:
            links = utils.extract_links(text)
            paths = utils.extract_paths(text)
            if (len(links) == 0 and len(paths) == 0):
                # no links
                continue            
            else:
                if len(links) > 0:
                    valid_links = []
                    for link in links:
                        print("Link: ", link)
                        # Get filename from link, if valid github repo link
                        provided_filepath = utils.extract_file_path_from_url(link)
                        if (len(provided_filepath) > 0):                            
                            valid_links.append(os.path.basename(provided_filepath))
                            print(f"Extracted file name: {provided_filepath} from bug description")

                if len(paths) > 0:
                    valid_paths = []
                    for path in paths:
                        print(f"Will be using filename: {os.path.basename(path)}")
                        root, ext = os.path.splitext(path)
                        if (ext == ".py"):
                            valid_paths.append(os.path.basename(path))

                paths_list.append(valid_links)
                paths_list.append(valid_paths)
        # Generate embeddings for each chunk of the bug description
        # Ideally, should have just one? Mmm, unless we have code snippets in it.
        # TODO: See how to extract the code snippet from the descriptions (LLM or some other kind of magic?).
        bug_description_embeddings = [self.model.encode(text) for text in texts]

        # Perform a similarity search in the database
        # Using chunks just in case, but bug description is usually short enough
        results = []
        for i, text_embedding in enumerate(bug_description_embeddings):
            link_results = []
            chunk_results = []

            if len(paths_list) > 0:
                print("Using metadata search with links")
                link_results = self.storage_instance.similarity_search_with_score(text_embedding, filter={"filename": {"$in": paths_list}})  
                chunk_results.extend(link_results)
            
            text_results = self.storage_instance.similarity_search_with_score(text_embedding)
            chunk_results.extend(text_results)

            # Sort all combined results by score (ascending order)
            chunk_results.sort(key=lambda x: x[1])
            # Get the top 5 shortest distances
            top5 = chunk_results[:5]
            results.extend(top5)

        # Now sort all the results together and return the overall top 5
        results.sort(key=lambda x: x[1])
        return results[:5]
    
    def cleanup(self):
        # Database reset
        self.storage_instance.cleanup()