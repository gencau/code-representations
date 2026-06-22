import anthropic
import os

class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'])
        self.model_name = "claude-3-5-sonnet-20241022"
  
    def chat(self, messages):
        response = self.client.messages.create(
            model=self.model_name,
            temperature=0.0,
            max_tokens=4096,
            messages=messages,
            stream=False,
            system="You are a coding assistant",
        )

        return response.content[0].text

    def formatAndSend(self, prompt) -> str:
        messages=[
            {"role": "user", "content": prompt },
        ]
        return self.chat(messages)