import json
import re
import ast
from typing import Dict, Any, Optional
from tenacity import wait_random_exponential, retry, stop_after_attempt

from langchain_ollama import ChatOllama
from src.baselines.backbones.agent.prompts.agent_context_prompt import AgentContextPrompt
from src.baselines.backbones.base_backbone import BaseBackbone
from src.utils.tokenization_utils import TokenizationUtils as tk
from src.baselines.backbones.agent.context.summary_retriever import SummaryRetriever


class BestOfNRankerBackbone(BaseBackbone):
    def __init__(self, name: str, model_name: str, prompt: AgentContextPrompt, 
                 retriever: SummaryRetriever, experiment: Optional[str] = None):
        super().__init__(name)
        self._model_name = model_name
        self._prompt = prompt
        self._retriever = retriever
        self._experiment = experiment
        self._tokenizer = tk(self._model_name)
        self._max_tokens = self._tokenizer._context_size

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages, tools=None):
        try:
            model = ChatOllama(model=self._model_name,
                                temperature=0.0,
                                num_predict=4096,
                                top_k=20,
                                top_p=0.8,
                                repeat_penalty=1.05) 
            
            response = model.invoke(messages)
            return response
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            return e

    def deduplicate_files(self, files):
        """
            Removes duplicates and invalid formats (dict) inserted within JSON objects.
        """
        unique = []
        seen = set()

        for item in files:
            # For dictionaries, convert to a tuple of sorted (key, value) pairs.
            if isinstance(item, dict):
                #skip it
                continue
            else:
                # Assuming item is already hashable (like a string)
                key = item
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _parse_json_response(self, response: str) -> list:
        files = []
        patterns = [
            r"```json\s*(\{.*?\})\s*```",   # JSON inside markdown code blocks
            r'</think>\s*(\{.*?\})',         # JSON after a </think> marker
            r"(\{.*?\})"                    # Fallback: any JSON-like object
        ]

        # Extract and process JSON objects
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for json_str in matches:
                try:
                    data = json.loads(json_str)
                    if "files" in data:
                        files.extend(data["files"])
                except json.JSONDecodeError:
                    print("#### JSON decoding error for match:", json_str)

        # Remove markdown code blocks before inline extraction to avoid double matching.
        cleaned_result = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
        # Also remove <think> tags so that we don't match files mentioned there
        cleaned_result = re.sub(r"<think>.*?</think>", "", cleaned_result, flags=re.DOTALL)
        if len(files) == 0:
            inline_matches = re.findall(r'`([^`]+)`', cleaned_result)
            # Filter inline matches: accept only candidates that appear to contain files (i.e. contain '.')
            valid_inline_matches = [m for m in inline_matches if '.' in m]
            files.extend(valid_inline_matches)

        # Get rid of any duplicates
        files = self.deduplicate_files(files)

        return files

    def _rerank_results(self, issue: str, file_list: list, base_sha: str):
        if len(file_list) <= 1:
            return file_list, []
        
        # the list of files is outputted by a LLM, there an be a lot of crap in it
        files = []
        for file in file_list:
            if isinstance(file, str):
                files.append(file)

        if (len(files) == 0):
            print("No files to rerank")
            return file_list, []

        file_summaries = {}
        for file in file_list:
            # Get context for all files in the list
            # Need base_sha to get collection
            if not isinstance(file, str):
                continue
            summary = self._retriever.find_summary(base_sha, file)
            print(f"Summary for file {file} is {summary}")
            if summary.get("documents"):
                file_summaries[file] = summary['documents'][0]
            else:
                print(f"No results retrieved in database for {file}")
        
        files_prompt = ""
        total_tokens = 0
        prompt_entries = []
        base_tokens = self._tokenizer.count_text_tokens(self._prompt.get_rerank_base_prompt())

        for file, entry in file_summaries.items():
            # Now fill the context with filename, file summary. Might need to prompt several times.
            entry_text = f"File: {file}\nSummary: {entry}\n\n"
            num_tokens = self._tokenizer.count_text_tokens(entry_text)
            if (base_tokens + total_tokens + num_tokens < self._max_tokens):
                files_prompt += '\n' + entry_text
                total_tokens += num_tokens
            else:
                if files_prompt:
                    prompt_entries.append(files_prompt)
                files_prompt = entry_text
                total_tokens = num_tokens

        if len(files_prompt) > 0:
            prompt_entries.append(files_prompt)

        # Prompt LLM with this chunk now
        combined_files = []
        metadatas = []
        for prompt in prompt_entries:
            full_prompt = self._prompt.chat(self._prompt.get_rerank_system_prompt(), self._prompt.get_rerank_prompt(issue, prompt))
            print(f"Re-rank Full prompt {full_prompt}")

            result = self.chat_completion_request(full_prompt)
            print(f"Result from LLM: {result}")
            # Try to interpret this mess
            json_obj = self._parse_json_response(result.content)

            print(f"Reranked Chat result: {json_obj}")
            combined_files.extend(json_obj)
            metadatas.append(result.response_metadata)

        # Combine all files into one final JSON object.
        print("Combined RERANKED JSON result:", combined_files)
        return combined_files, metadatas

    def _iterate_on_context(self, issue : str, filelist: str, base_sha: str):
        """
            Calculate # tokens in base prompt (without project content) to get remaining tokens for project content
            Split the project content by this amount of tokens until all are split
            Call the LLM, saving each returned file list in memory
        """
        system_prompt = self._prompt.get_system_prompt()
        base_prompt = self._prompt.base_prompt(issue)
        
        # Now count the tokens in the static parts of the prompt + issue description
        # This uses the model's specific tokenizer
        tk_count = self._tokenizer.count_text_tokens(base_prompt) + self._tokenizer.count_text_tokens(system_prompt)
        
        print(f"Static prompt + issue description has {tk_count} tokens")

        # Define the files tags and calculate token count.
        # Issue tags are already counted with the base prompt
        start_tag = "<FILES>"
        end_tag = "</FILES>"
        start_tag_token_count = self._tokenizer.count_text_tokens(start_tag)
        end_tag_token_count = self._tokenizer.count_text_tokens(end_tag)
        tokens_per_chunk = self._max_tokens - tk_count - start_tag_token_count - end_tag_token_count
        
        print(f"{tokens_per_chunk} tokens remaining for the project context")

        chunks = []
        current_chunk = ""

        # this is to make sure that we don't cut in the middle of a path
        for file_path in filelist:
            candidate = file_path if not current_chunk else current_chunk + '\n' + file_path
            if self._tokenizer.count_text_tokens(candidate) <= tokens_per_chunk:
                current_chunk  = candidate
            else:
                # if over the max token count, create a chunk with previous content
                # and start a new one
                chunks.append(current_chunk)
                current_chunk = file_path
        
        if current_chunk:
            chunks.append(current_chunk)

        results_list = []
        files_list = []
        response_metadata = []
        for chunk_text in chunks:
            # Call LLM with this chunk
            full_prompt = self._prompt.chat(self._prompt.get_system_prompt(), self._prompt.full_prompt(issue, chunk_text))
            print(f"Full prompt {full_prompt}")

            # Ensure the chunk ends with the required closing tag.
            for message in full_prompt:
                if (message["role"] == "user"):
                    if not message['content'].rstrip().endswith(end_tag):
                        message['content'] = message['content'].rstrip() + " " + end_tag

            result = self.chat_completion_request(full_prompt)
            print(f"Chat results: {result}")
            results_list.append(result.content)
            response_metadata.append(result.response_metadata)

        for result in results_list:
            data = self._parse_json_response(result)
            if len(data) > 0:
                files_list.extend(data)

        print(f"Final files list: {files_list}")
        reranked_results, rerank_metadata = self._rerank_results(issue, files_list, base_sha)

        return files_list, reranked_results, results_list, response_metadata, rerank_metadata
    

    def localize_bugs(self, issue_description: str, repo_content: str, base_sha: str, **kwargs) -> Dict[str, Any]:
        print(f"Handling top-10 list: {repo_content} for {base_sha}")
        filelist = ast.literal_eval(repo_content)
        reranked_files,  response_metadata = self._rerank_results(issue_description, filelist, base_sha)

        clean_reranked_files = []            
        for f in reranked_files:
            if not isinstance(f, str):
                continue
            if f in repo_content:
                clean_reranked_files.append(f)
        return {
            "reranked_files": reranked_files,
            "clean_reranked_files":clean_reranked_files,
            "response_metadata": response_metadata,
        }
