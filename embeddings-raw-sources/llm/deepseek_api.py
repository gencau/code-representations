from openai import OpenAI
import os


class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.0,
            messages=messages,
            stream=False
        )

        return response.choices[0].message.content
    

    '''messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],'''

