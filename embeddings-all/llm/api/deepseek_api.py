from openai import OpenAI
import os


class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], 
                             base_url="https://api.deepseek.com/v1",
                             default_headers={  
                                "Host": "api.deepseek.com"  # Required header
                            })

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.0,
            messages=messages,
            stream=False
        )

        return response.choices[0].message.content
    

    def formatAndSend(self, prompt) -> str:
        messages=[
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": prompt },
        ]
        return self.chat(messages)


