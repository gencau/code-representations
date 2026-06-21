from typing import List

from src.baselines.utils.type_utils import ChatMessage
from src.baselines.backbones.agent.prompts.agent_prompt_templates import AGENT_SYSTEM_PROMPT_TEMPLATE, \
                                                                         AGENT_ISSUE_PROMPT_TEMPLATE, \
                                                                         AGENT_CONTEXT_PROMPT_TEMPLATE, \
                                                                         AGENT_RERANK_PROMPT_TEMPLATE


class AgentContextPrompt():

    def base_prompt(self, issue_description: str) -> str:
        return AGENT_ISSUE_PROMPT_TEMPLATE.format(issue_description)
    
    def full_prompt(self, issue_description, project_content: str) -> str:
        return self.base_prompt(issue_description) + '\n' + AGENT_CONTEXT_PROMPT_TEMPLATE.format(project_content)

    def get_fixed_prompt(self, issue_description: str) -> str:
        return self.base_prompt(issue_description)
    
    def get_system_prompt(self) -> str:
        return AGENT_SYSTEM_PROMPT_TEMPLATE
    
    def get_rerank_system_prompt(self) -> str:
        return "You are a file reranker."
    
    def get_rerank_prompt(self, issue: str, files: str) -> str:
        return AGENT_RERANK_PROMPT_TEMPLATE.format(issue, files)

    def chat(self, system_prompt: str, prompt: str) -> List[ChatMessage]:
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
