from typing import List
from src.baselines.utils.type_utils import ChatMessage
from src.baselines.backbones.agent.prompts.agent_context_prompt import AgentContextPrompt
from src.baselines.backbones.agent.prompts.agent_bug_summary_prompt_templates import AGENT_BUG_REPORT_PROMPT_TEMPLATE, AGENT_SYSTEM_PROMPT_TEMPLATE, AGENT_BUG_REPORT_RANK_PROMPT_TEMPLATE


class AgentBugSummaryPrompt(AgentContextPrompt):

    def get_base_prompt(self):
        return AGENT_BUG_REPORT_PROMPT_TEMPLATE
        
    def full_prompt(self, issue_description, project_content: str) -> str:
        return AGENT_BUG_REPORT_PROMPT_TEMPLATE.format(issue_description, project_content)

    def get_rerank_base_prompt(self) -> str:
        return AGENT_BUG_REPORT_RANK_PROMPT_TEMPLATE
        
    def get_rerank_prompt(self, issue: str, files: str) -> str:
        return AGENT_BUG_REPORT_RANK_PROMPT_TEMPLATE.format(issue, files)
    
    def get_system_prompt(self) -> str:
        return AGENT_SYSTEM_PROMPT_TEMPLATE
    
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