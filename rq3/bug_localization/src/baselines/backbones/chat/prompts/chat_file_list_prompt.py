from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.baselines.backbones.chat.prompts.chat_prompt_templates import FILE_LIST_PROMPT_TEMPLATE, FILE_LIST_PROMPT_TEMPLATE_2


class ChatFileListPrompt(ChatBasePrompt):
    def base_prompt(self, issue_description: str, project_content: dict[str, str]) -> str:
        file_paths = '\n'.join(project_content.keys())

        return FILE_LIST_PROMPT_TEMPLATE.format(issue_description, file_paths)
