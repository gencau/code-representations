import json
import re
from typing import Dict, Any, Optional
from tenacity import wait_random_exponential, retry, stop_after_attempt

from langchain_ollama import ChatOllama
from src.baselines.backbones.agent.context.summary_retriever import SummaryRetriever
from src.baselines.backbones.agent.prompts.agent_context_prompt import AgentContextPrompt
from src.baselines.backbones.base_backbone import BaseBackbone
from src.utils.tokenization_utils import TokenizationUtils as tk
from src.baselines.utils.prompt_utils import parse_list_files_completion

BUFFER = 200

class OllamaRankBackbone(BaseBackbone):
    def __init__(self, name: str, model_name: str, type: str, prompt: AgentContextPrompt, retriever: Optional[SummaryRetriever] = None, experiment: Optional[str] = None):
        super().__init__(name)
        self._model_name = model_name
        self._prompt = prompt
        self._experiment = experiment
        self._type = type
        self._retriever = retriever
        self._tokenizer = tk(self._model_name)
        self._max_tokens = self._tokenizer._context_size


    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages, tools=None):
        try:
            model = ChatOllama(model=self._model_name,
                                temperature=0.0,
                                num_predict=2048,
                                top_k=20,
                                top_p=0.5) 
            
            response = model.invoke(messages)
            return response
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            return e
    
    def _rank_w_sources(self, issue: str, repo_content: Dict[str, str]) -> tuple[list, list, int]:
        # For each file, retrieve content and format for prompt
        file_w_content = []
        for f, content in repo_content.items():
            if content is not None:
                file_w_content.append(f"{f}: {content}")
            else:
                file_w_content.append(f"{f}: No content found.")

        ranked, metadata, num_files_viewed = self._rank_results(issue, file_w_content, system_prompt=self._prompt.get_rerank_system_prompt(),
                                  user_prompt_template=self._prompt.get_rerank_base_prompt())
        return [e.split(":")[0].strip() for e in ranked], metadata, num_files_viewed

    def _rank_w_summaries(self, issue: str, file_list: list, base_sha: str) -> tuple[list, list, int]:
        # For each file, retrieve summary and format for prompt
        file_w_summaries = []
        for f in file_list:
            summary = self._retriever.find_summary(collection=base_sha, filename=f)
            if summary and "documents" in summary and len(summary['documents']) > 0:
                file_w_summaries.append(f"{f}: {summary['documents'][0]}")
            else:
                file_w_summaries.append(f"{f}: No summary found.")

        system_prompt = self._prompt.get_rerank_system_prompt()
        user_prompt_template = self._prompt.get_rerank_base_prompt()
        ranked, metadata, num_files_viewed = self._rank_results(issue, file_w_summaries, system_prompt, user_prompt_template)
        return [e.split(":")[0].strip() for e in ranked], metadata, num_files_viewed

    def _reorder_window(self, window: list, reranked_paths: list) -> list:
        """Reorder window entries to match reranked_paths order; unmatched entries go to the end."""
        def extract_path(entry: str) -> str:
            return entry.split(":")[0].strip()

        path_to_entry = {extract_path(e): e for e in window}
        reordered = []
        seen = set()
        for path in reranked_paths:
            if path in path_to_entry and path not in seen:
                reordered.append(path_to_entry[path])
                seen.add(path)
        # Preserve unmatched entries at the end
        for entry in window:
            if extract_path(entry) not in seen:
                reordered.append(entry)
        return reordered

    def _rank_results(self, issue: str, content_list: list, system_prompt: str, user_prompt_template: str) -> tuple[list, list, int]:
        valid_files = [f for f in content_list if isinstance(f, str)]
        if len(valid_files) <= 1:
            return valid_files, [], len(valid_files)

        max_tokens = self._max_tokens - BUFFER
        n_sys = self._tokenizer.count_text_tokens(system_prompt)

        # Pre-truncate any file whose content alone already exceeds the context
        truncated_files = []
        for f in valid_files:
            n_user = self._tokenizer.count_text_tokens(user_prompt_template.format(issue=issue, files=f))
            if n_sys + n_user > max_tokens:
                f = self._tokenizer._truncate(f, max_tokens - n_sys)
                print("WARNING: File content truncated to fit in context.")
            truncated_files.append(f)

        # Determine how many files fit in one context window
        window_size = 0
        for i in range(1, len(truncated_files) + 1):
            n_user = self._tokenizer.count_text_tokens(
                user_prompt_template.format(issue=issue, files="\n".join(truncated_files[:i]))
            )
            if n_sys + n_user > max_tokens:
                break
            window_size = i
        window_size = max(window_size, 1)

        # Single prompt with as many files as fit in the context window
        window = truncated_files[:window_size]
        full = self._prompt.chat(system_prompt, user_prompt_template.format(issue=issue, files="\n".join(window)))
        print(f"Sending single prompt with {len(window)} files.")
        resp = self.chat_completion_request(full)
        reranked_paths = parse_list_files_completion(resp.content)
        ranked = self._reorder_window(window, reranked_paths) + truncated_files[window_size:]

        return ranked, [resp.response_metadata], len(window)

    def _iterate_on_context(self, issue : str, file_list: list, repo_content: Dict[str, str], base_sha: str) -> tuple[list, list, int]:
        if self._type == "rank-w-filenames":
            print("Ranking files based on filenames only.")
            ranked_results, rank_metadata, num_files_viewed = self._rank_results(issue, file_list, system_prompt=self._prompt.get_rerank_system_prompt(),
                                                               user_prompt_template=self._prompt.get_rerank_base_prompt())
        elif self._type == "rank-w-summaries" or self._type == "rank-w-bug-reports":
            ranked_results, rank_metadata, num_files_viewed = self._rank_w_summaries(issue, file_list, base_sha)
        elif self._type == "rank-w-sources":
            filtered_content = {f: repo_content.get(f) for f in file_list if f in repo_content}
            ranked_results, rank_metadata, num_files_viewed = self._rank_w_sources(issue, filtered_content)
        else:
            raise ValueError(f"Unsupported rank backbone type {self.type}")

        print(f"RANKED Final files list: {ranked_results}")
        return ranked_results, rank_metadata, num_files_viewed


    def localize_bugs(self, issue_description: str, file_list: list, repo_content: Dict[str, str], base_sha: str, **kwargs) -> Dict[str, Any]:
        ranked_files, rank_metadata, num_files_viewed = self._iterate_on_context(issue_description, file_list, repo_content, base_sha)

        return {
            "ranked_files": ranked_files,
            "rank_metadata": rank_metadata,
            "num_files_viewed": num_files_viewed,
        }
