from typing import Dict, Any, Optional
from tenacity import wait_random_exponential, retry, stop_after_attempt

from langchain_ollama import ChatOllama
from src.baselines.backbones.agent.prompts.agent_code_prompt import AgentCodePrompt
from src.baselines.backbones.base_backbone import BaseBackbone
from src.utils.tokenization_utils import TokenizationUtils as tk
from src.baselines.utils.prompt_utils import parse_json_response

MAX_OUTPUT_TOKENS = 1024
BUFFER = 200

class OllamaCodeChatBackbone(BaseBackbone):
    def __init__(self, name: str, model_name: str, prompt: AgentCodePrompt, 
                 experiment: Optional[str] = None):
        super().__init__(name)
        self._model_name = model_name
        self._prompt = prompt
        self._experiment = experiment
        self._tokenizer = tk(self._model_name)
        self._max_tokens = self._tokenizer._context_size 
        self._model = ChatOllama(model=self._model_name,
                                 temperature=0.0,
                                 repeat_penalty=1.05,
                                 num_predict=MAX_OUTPUT_TOKENS,
                                 top_k=20,
                                 top_p=0.8) 

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages, tools=None):
        try:
            response = self._model.invoke(messages)
            return response
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            return e
        
    # A helper function to split a file’s text into manageable chunks
    def _split_entry(self, text, max_allowed_tokens):
        """
        Splits the input text into chunks that are each no longer than max_allowed_tokens.
        Assumes that self._tokenizer.tokenize() returns a list of tokens and that 
        self._tokenizer.detokenize() reconstructs text from tokens.
        """
        tokens = self._tokenizer._encode(text)
        chunks = []
        # Split tokens into chunks of size max_allowed_tokens
        for i in range(0, len(tokens), max_allowed_tokens):
            chunk_tokens = tokens[i : i + max_allowed_tokens]
            chunk_text = self._trim_non_utf(self._tokenizer._decode(chunk_tokens))
            chunks.append(chunk_text)
        return chunks
    
    def _trim_non_utf(self, s: str) -> str:
        # Remove NULs
        s = s.replace("\x00", "")
        # Force valid Unicode scalar values by round-tripping through UTF-8.
        # Any lone surrogates / invalids become �.
        return s.encode("utf-8", errors="replace").decode("utf-8")


    def _prompt_llm(self, issue: str, project_content: dict):     
        files_prompt = ""
        total_tokens = 0
        prompt_entries = []
        
        base_tokens = self._tokenizer.count_text_tokens(self._prompt.get_system_prompt() + self._prompt.get_base_prompt(issue))

        # number of files in one prompt
        num_files_in_prompt = 0

        # list of all numbers of prompts, to calculate a mean later
        num_files_in_prompt_list = []
        for file, entry in project_content.items():
            # Now fill the context with filename, file content. Will need to prompt several times.
            entry_text = self._trim_non_utf(f"File: {file}\nCode: {entry}\n\n")

            max_allowed_for_entry = self._max_tokens - base_tokens - MAX_OUTPUT_TOKENS - BUFFER

            num_tokens = self._tokenizer.count_text_tokens(entry_text)

            if num_tokens > max_allowed_for_entry:
                # Split the file into chunks that each fit within the available token budget.
                chunks = self._split_entry(entry_text, max_allowed_for_entry)
                for chunk in chunks:
                    chunk_tokens = self._tokenizer.count_text_tokens(chunk)
                    # Try to fit this chunk into the current prompt.
                    if base_tokens + total_tokens + chunk_tokens < self._max_tokens:
                        files_prompt += "\n" + chunk
                        total_tokens += chunk_tokens
                        num_files_in_prompt += 1
                    else:
                        # If it doesn't fit, flush the current prompt and start a new one.
                        if files_prompt:
                            prompt_entries.append(files_prompt)
                            num_files_in_prompt_list.append(num_files_in_prompt)
                        files_prompt = chunk
                        total_tokens = chunk_tokens
                        num_files_in_prompt = 1
            else:
                if (base_tokens + total_tokens + num_tokens < self._max_tokens):
                    files_prompt += '\n' + entry_text
                    total_tokens += num_tokens
                    num_files_in_prompt += 1
                else:
                    if files_prompt:
                        prompt_entries.append(files_prompt)
                    files_prompt = entry_text
                    total_tokens = num_tokens
                    print(f"GEN --- There were {num_files_in_prompt} files included in this prompt.")
                    num_files_in_prompt_list.append(num_files_in_prompt)
                    num_files_in_prompt = 1

        if len(num_files_in_prompt_list) > 0:
            print(f"GEN --- There was on average {sum(num_files_in_prompt_list) / len(num_files_in_prompt_list)} files per prompt for this repo")
        else:
            print("ERROR -- no files were included for this repository")

        if len(files_prompt) > 0:
            prompt_entries.append(files_prompt)

        # Prompt LLM with this chunk now
        combined_files = []
        results_list = []
        metadata_list = []
        for prompt in prompt_entries:
            full_prompt = self._prompt.chat(self._prompt.get_system_prompt(), self._prompt.full_prompt(issue, prompt))
            print(f"Full prompt {full_prompt}")

            result = self.chat_completion_request(full_prompt)
            print(f"Result from LLM: {result}")
            # Try to interpret this mess
            json_obj = parse_json_response(result.content)

            print(f"Processed chat result: {json_obj}")
            combined_files.extend(json_obj)
            results_list.append(result.content)
            metadata_list.append(result.response_metadata)

        # Combine all files into one final JSON object.
        print("File list for this project: ", combined_files)
        return combined_files, results_list, metadata_list

    def _iterate_on_context(self, issue : str, repo_content: Dict[str, str]):
        """
            Calculate # tokens in base prompt (without project content) to get remaining tokens for project content
            Split the project content by this amount of tokens until all are split
            Call the LLM, saving each returned file list in memory
        """
        predicted_files, results_list, response_metadata = self._prompt_llm(issue, repo_content)

        return predicted_files, results_list, response_metadata
    

    def localize_bugs(self, issue_description: str, repo_content: Dict[str, str], **kwargs) -> Dict[str, Any]:
        generated_files, results_list, response_metadata = self._iterate_on_context(issue_description, repo_content)

        # Find valid files from generated files
        final_files = [f for f in generated_files if not isinstance(f, dict) and f in repo_content]
    
        return {
            "all_generated_files": generated_files,
            "final_files": final_files,
            "raw_responses": results_list,
            "response_metadata": response_metadata
        }
