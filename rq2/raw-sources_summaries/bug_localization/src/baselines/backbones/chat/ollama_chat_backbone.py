from typing import Dict, Any, Optional, List

import os
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests
import json

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.baselines.utils.prompt_utils import batch_project_context, parse_list_files_completion
from src.baselines.utils.type_utils import ChatMessage


class OllamaChatBackbone(BaseBackbone):

    def __init__(
            self,
            ollama_endpoint: str,
            name: str,
            ollama_model: str,
            prompt: ChatBasePrompt,
            parameters: Dict[str, Any],
            experiment: str
    ):
        super().__init__(name)
        self._model_name = ollama_model
        self._ollama_endpoint = ollama_endpoint.rstrip('/')
        self._prompt = prompt
        self._parameters = parameters

    def build_chat_message(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        for msg in messages:
            if msg["role"] == "system":
                continue            
            elif msg["role"] == "user" and self._model_name.startswith("codellama"):
                #insert codellama-specific tags
                msg["content"] = "[INST]" + msg["content"] + "[/INST]"

        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    

    def get_response(self, messages: List[ChatMessage], max_tokens: int=150):
        prompt = self.build_chat_message(messages)
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "temperature": self._parameters.get("temperature", 0.0),
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.05
        }

        url = f"{self._ollama_endpoint}/api/generate"
        response = requests.post(url, json=payload)
        response.raise_for_status()

        complete_response = ""
        # Process the streaming response line by line.
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            # Append the chunk's "response" text.
            complete_response += chunk.get("response", "")
            # Optionally, check if done is true.
            if chunk.get("done", False):
                break

        print(f"Generated response: {complete_response}")
        # Wrap the generated text into a structure mimicking the OpenAI ChatCompletion response.
        # This could be changed.
        simulated_response = {
            "choices": [
                {
                    "message": {
                        "content": complete_response
                    }
                }
            ]
        }
        return simulated_response

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def localize_bugs(self, issue_description: str, repo_content: dict[str, str]) -> Dict[str, Any]:
        
        batched_project_contents = batch_project_context(
            self._model_name, self._prompt, issue_description, repo_content, True
        )

        files = set()
        final_files = set()
        raw_completions = []
        for batched_project_content in batched_project_contents:
            messages = self._prompt.chat(issue_description, batched_project_content)

            completion = self.get_response(messages)
            raw_completion_content = completion["choices"][0]["message"]["content"]
            raw_completions.append(raw_completion_content)
            files.update(parse_list_files_completion(raw_completion_content))

        # Happens only if context is larger than context window
        if len(batched_project_contents) > 1:
            messages = self._prompt.chat(issue_description, {f: repo_content[f] for f in files if f in repo_content})
            completion = self.get_response(messages)
            raw_completion_content = completion["choices"][0]["message"]["content"]
            raw_completions.append(raw_completion_content)
            final_files.update(parse_list_files_completion(raw_completion_content))
        else:
            final_files = [f for f in files if f in repo_content]

        return {
            "all_generated_files": list(files),
            "final_files": list(final_files),
            "raw_completions": raw_completions,
            "batches_count": len(batched_project_contents)
        }
