from typing import List

from src.baselines.utils.type_utils import ChatMessage
from src.baselines.backbones.agent.prompts.agent_prompt_templates import AGENT_SYSTEM_PROMPT_TEMPLATE, \
                                                                         AGENT_SUMMARY_PROMPT_TEMPLATE,  AGENT_RERANK_SUMMARIES_PROMPT_TEMPLATE


class AgentSummaryPrompt():

    def get_base_prompt(self, issue):
        return AGENT_SUMMARY_PROMPT_TEMPLATE.format(issue, "")
        
    def full_prompt(self, issue_description, project_content: str) -> str:
        return AGENT_SUMMARY_PROMPT_TEMPLATE.format(issue_description, project_content)
    
    def get_system_prompt(self) -> str:
        return AGENT_SYSTEM_PROMPT_TEMPLATE
    
    def get_rerank_system_prompt(self) -> str:
        return "You are a file ranker."
    
    def get_rerank_base_prompt(self) -> str:
        return AGENT_RERANK_SUMMARIES_PROMPT_TEMPLATE
    
    def get_rerank_prompt(self, issue_description, project_content: str) -> str:
        return AGENT_RERANK_SUMMARIES_PROMPT_TEMPLATE.format(issue_description, project_content)
    
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
