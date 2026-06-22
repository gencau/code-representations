from pathlib import Path
from tqdm import tqdm
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import utils.file_utils as utils
from code.code_chunk import parse_python_text
from llm.summaries import generateFileSummary, generateMethodSummary, generateClassSummary, generateRetrievalQueries, generateBugReports
from llm.api.groq_api import GroqClient
from utils.file_view_utils import filter_python_minimal, filter_java_kotlin_minimal
from utils.token_splitting import make_hf_tokenizer, make_packed_method_chunks

class RecordProcessor:
    representation_type = ['query', 'summary', 'report', 'path']
    chunking_granularity = ['file', 'character', 'token']
    language_mapping = {
        'py': 'python',
        'java': 'java',
        'kt': 'kotlin'
    }

    def __init__(self, representation_type='report', chunking_granularity='token'):

        self.representation_type = representation_type
        self.chunking_granularity = chunking_granularity
        if representation_type not in self.representation_type:
            raise ValueError(f"Unsupported representation type: {representation_type}")
        if chunking_granularity not in self.chunking_granularity:
            raise ValueError(f"Unsupported chunking granularity: {chunking_granularity}")
        
        # Embedding model
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        #self.model = SentenceTransformer('thenlper/gte-large')
        #self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        #self.model = SentenceTransformer('Salesforce/SFR-Embedding-Code-400M_R', trust_remote_code=True)
        self.llm_client = GroqClient()

        # Python separators for code splitting
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

    def process_record(self, output_dir, language, batch_size=32):
        # For each file, generate a summary        
        record_data = self.split_and_generate(language, output_dir)
        file_embeddings = self.create_embeddings(record_data['file_summaries'], batch_size)
    
        return record_data, file_embeddings#, class_embeddings, method_embeddings
    
    def split_and_generate(self, language, output_dir):
        starting_id = 1
        file_summaries = []
        ids = []
        files_metadata = []

        class_summaries = []
        class_metadata = []
        class_ids = []

        method_summaries = []
        method_metadata = []
        method_ids = []

        errors = 0

        directory = Path(output_dir)
        summary_processing_time = 0
        for file_path in directory.rglob(f'*{language}'):
            if not file_path.is_file() or utils.is_test_file(file_path.as_posix()):
                continue  # Skip directories and test files
            print(f"Processing file {file_path} extracted from repo {directory.name}")

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    file_content = f.read()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    errors += 1
                    continue

                if len(file_content.strip()) == 0:
                    continue

                filename = utils.get_path_within_repo(output_dir, str(file_path))

                # Remove imports and package declarations
                if language in ['.java', '.kt']:
                    file_content = filter_java_kotlin_minimal(file_content)
                elif language == '.py':
                    file_content = filter_python_minimal(file_content)

                if len(file_content.strip()) == 0:
                    continue

                texts = []
                text_splitter = None
                # Now split based on chunking type
                if (self.representation_type == 'path'):
                    # We don't chunk file paths
                        texts = [filename]
                        print(f"Using file path as representation: {texts}")

                if (self.chunking_granularity == 'file'):
                    text_splitter = RecursiveCharacterTextSplitter (chunk_size=76000,
                                                                    chunk_overlap=0,
                                                                    separators=self.python_separators,
                                                                    length_function=len)
                elif (self.chunking_granularity == 'character'):
                    text_splitter = RecursiveCharacterTextSplitter (chunk_size=2048,
                                                                    chunk_overlap=60,
                                                                    separators=self.python_separators,
                                                                    length_function=len)
                else:   # token-based
                    tokenizer = make_hf_tokenizer(self.model)
                    chunks = make_packed_method_chunks(
                        code=file_content,
                        language=self.language_mapping.get(language),
                        file_path=filename,
                        tokenizer=tokenizer,
                        chunk_tokens=512,
                        overlap_tokens=20,
                    )
                    print(f"Produced {len(chunks)} chunks")
                    texts = [chunk.text for chunk in chunks]
                
                if texts == []:
                    if text_splitter is not None:
                        texts = text_splitter.split_text(file_content)
                        print(f"Produced {len(texts)} text splits")
                    else:
                        print("<no text splits produced>")
                        continue

                for text in texts:
                    if not text.strip():
                        continue

                    if (self.representation_type == 'path'):
                        docstring = text
                    elif (self.representation_type == 'summary'): # full file summary
                        start_time = time.time()
                        docstring = generateFileSummary(text, self.llm_client)
                        end_time = time.time()
                        summary_processing_time += (end_time - start_time)
                        print(f"Generated file summary in {end_time - start_time:.2f} seconds")
                    elif (self.representation_type == 'query'):
                        ## Try different number of queries: 5, 10, 15, 20, see if linear increase
                        docstring = generateRetrievalQueries(text, self.llm_client, num_queries=5, num_tokens=30)
                        print(f"File summary: {docstring}")
                    else:   # bug reports
                        start_time = time.time()
                        docstring = generateBugReports(text, self.llm_client, num_reports=5, num_tokens=30)
                        end_time = time.time()
                        print(f"Bug report: {docstring}")
                        summary_processing_time += (end_time - start_time)
                        print(f"Generated bug report in {end_time - start_time:.2f} seconds")

                    if not docstring or len(docstring) == 0:
                        errors += 1
                        print(f"Error generating summaries for file {filename}. Total errors: {errors}")

                    # For queries, we get several documents, so we should check if a array is returned and 
                    # Create one document for each
                    if isinstance(docstring, list):
                        # index one document per expansion
                        for i, doc in enumerate(docstring):
                            file_metadata = {
                                "type": "file",
                                "filename": filename,
                                "directory": str(output_dir),
                                "query_index": i
                            }
                            print(f"Metadata: {file_metadata}")

                            file_summaries.append(doc)
                            files_metadata.append(file_metadata)
                            ids.append("ID_" + str(starting_id))
                            starting_id += 1
                    else:
                        # index once
                        file_metadata = {
                            "type": "file",
                            "filename": filename,
                            "directory": str(output_dir)
                        }
                        print(f"File-level metadata: {file_metadata}")

                        file_summaries.append(docstring)
                        files_metadata.append(file_metadata)
                        ids.append("ID_" + str(starting_id))
                        starting_id += 1
        return {
            "file_summaries": file_summaries,
            "file_metadata": files_metadata,
            "file_ids": ids,
            "class_summaries": class_summaries,
            "class_metadata": class_metadata,
            "class_ids": class_ids,
            "method_summaries": method_summaries,
            "method_metadata": method_metadata,
            "method_ids": method_ids,
            "errors": errors,
            "summary_processing_time": summary_processing_time
        }

    def create_embeddings(self, file_summaries, batch_size):
        print("Now creating embeddings in batches...")

        files_embeddings = []

        # Create embeddings in batches
        for i in tqdm(range(0, len(file_summaries), batch_size), desc="Encoding Batches"):
            batch_texts = file_summaries[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(batch_texts)
                files_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch starting at index {i}: {e}")
                # Optionally, handle or log the error, skip the batch, etc.
                files_embeddings.extend([None] * len(batch_texts))
            
        return files_embeddings
    
    def embed_bug_description(self, bug_description):
        # Split the bug description into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096,
                                                        chunk_overlap=0,
                                                        separators=["\n\n", "\n", " ", ">"],
                                                        length_function=len)
        texts = text_splitter.split_text(bug_description)

        # Generate embeddings for each chunk of the bug description
        bug_description_embeddings = [self.model.encode(text) for text in texts]

        return bug_description_embeddings