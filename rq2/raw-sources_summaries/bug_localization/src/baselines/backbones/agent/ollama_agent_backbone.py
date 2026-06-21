from typing import Dict, Any, Optional
from tenacity import wait_random_exponential, retry, stop_after_attempt

from langchain_ollama import ChatOllama
from src.baselines.backbones.agent.prompts.agent_context_prompt import AgentContextPrompt
from src.baselines.backbones.base_backbone import BaseBackbone
from src.utils.tokenization_utils import TokenizationUtils as tk
from src.baselines.backbones.agent.context.summary_retriever import SummaryRetriever
from src.baselines.utils.prompt_utils import parse_json_response

MAX_OUTPUT_TOKENS = 1024
SAFETY = 200

class OllamaAgentBackbone(BaseBackbone):
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
                                repeat_penalty=1.05,
                                num_predict=MAX_OUTPUT_TOKENS,
                                top_k=20,
                                top_p=0.8) 
            
            response = model.invoke(messages)
            return response
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            return e

    def _prompt_llm(self, issue: str, file_list: list, base_sha: str, rerank: False):
        if len(file_list) <= 1:
            return file_list, [], []
        
        # the list of files is output by a LLM, there can be a lot of crap in it
        files = []
        for file in file_list:
            if isinstance(file, str):
                files.append(file)

        if (len(files) == 0):
            print("No files to rerank")
            return file_list, [], []

        file_summaries = {}
        for file in file_list:
            # Get context for all files in the list
            if not isinstance(file, str):
                continue
            summary = self._retriever.find_summary(base_sha, file)
            print(f"Summary for file {file} is {summary}")
            if summary.get("documents"):
                if len(summary['documents']) == 1:
                    file_summaries[file] = summary['documents'][0]
                else:
                    print(f"Multiple summaries found for {file}, combining them")
                    file_summaries[file] = "\n".join(f"{i+1}. {doc}" for i, doc in enumerate(summary['documents']))
            else:
                print(f"No results retrieved in database for {file}")
        
        files_prompt = ""
        total_tokens = 0
        prompt_entries = []
        
        if (rerank):
            base_tokens = self._tokenizer.count_text_tokens(self._prompt.get_rerank_system_prompt() + self._prompt.get_rerank_base_prompt())
        else:
            base_tokens = self._tokenizer.count_text_tokens(self._prompt.get_system_prompt() + self._prompt.get_base_prompt(""))

        max_token_allocation = self._max_tokens - MAX_OUTPUT_TOKENS - SAFETY
        for file, entry in file_summaries.items():
            # Now fill the context with filename, file summary. Might need to prompt several times.
            entry_text = f"File: {file}\nSummary: {entry}\n\n"
            entry_text = entry_text.replace("<|endoftext|>", "")
            num_tokens = self._tokenizer.count_text_tokens(entry_text)
            if (base_tokens + total_tokens + num_tokens < max_token_allocation):
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
        results_list = []
        metadata_list = []
        for prompt in prompt_entries:
            if rerank:
                full_prompt = self._prompt.chat(self._prompt.get_rerank_system_prompt(), self._prompt.get_rerank_prompt(issue, prompt))
            else:
                full_prompt = self._prompt.chat(self._prompt.get_system_prompt(), self._prompt.full_prompt(issue, prompt))
            print(f"(Re-rank) Full prompt {full_prompt}")

            result = self.chat_completion_request(full_prompt)
            print(f"Result from LLM: {result}")
            # Try to interpret this mess
            json_obj = parse_json_response(result.content)

            print(f"Chat result: {json_obj}")
            combined_files.extend(json_obj)
            results_list.append(result.content)
            metadata_list.append(result.response_metadata)

        # Combine all files into one final JSON object.
        if rerank:
            print("Combined RERANKED JSON result:", combined_files)
        else:
            print("File list for this project: ", combined_files)
        return combined_files, results_list, metadata_list

    def _iterate_on_context(self, issue : str, repo_content: Dict[str, str], base_sha: str):
        """
            Calculate # tokens in base prompt (without project content) to get remaining tokens for project content
            Split the project content by this amount of tokens until all are split
            Call the LLM, saving each returned file list in memory
        """

        files_list = list(repo_content.keys())
        predicted_files, results_list, response_metadata = self._prompt_llm(issue, files_list, base_sha, False)

        return predicted_files, results_list, response_metadata, "", "", ""
    

    def localize_bugs(self, issue_description: str, repo_content: Dict[str, str], base_sha: str, **kwargs) -> Dict[str, Any]:
        generated_files, results_list, response_metadata, \
            reranked_files, reranked_results, reranked_metadata = self._iterate_on_context(issue_description, repo_content, base_sha)

        # Find valid files from generated files
        final_files = [f for f in generated_files if not isinstance(f, dict) and f in repo_content]
        final_reranked_files = [f for f in reranked_files if not isinstance(f, dict) and f in repo_content]
        return {
            "all_generated_files": generated_files,
            "final_files": final_files,
            "raw_responses": results_list,
            "response_metadata": response_metadata,
            "reranked_files": reranked_files,
            "final_reranked_files": final_reranked_files,
            "reranked_results": reranked_results,
            "reranked_metadata": reranked_metadata
        }
