from typing import List

from src.baselines.utils.type_utils import ChatMessage
from src.baselines.backbones.agent.prompts.agent_prompt_templates import AGENT_CODE_PROMPT_TEMPLATE_2, AGENT_SYSTEM_PROMPT_TEMPLATE, \
                                                                         AGENT_CODE_RERANK_PROMPT_TEMPLATE


class AgentCodePrompt():

    def get_base_prompt(self, issue):
        return AGENT_CODE_PROMPT_TEMPLATE_2.format(issue, "")
        
    def full_prompt(self, issue_description, project_content: str) -> str:
        return AGENT_CODE_PROMPT_TEMPLATE_2.format(issue_description, project_content)
    
    def get_system_prompt(self) -> str:
        return AGENT_SYSTEM_PROMPT_TEMPLATE
    
    def get_rerank_system_prompt(self) -> str:
        return "You are a file ranker."
    
    def get_rerank_base_prompt(self) -> str:
        return AGENT_CODE_RERANK_PROMPT_TEMPLATE
    
    def get_rerank_prompt(self, issue_description, project_content: str) -> str:
        return AGENT_CODE_RERANK_PROMPT_TEMPLATE.format(issue_description, project_content)
    
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
