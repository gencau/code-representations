from openai import OpenAI
import os


class GroqClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['GROQ_API_KEY'], base_url="https://api.groq.com/openai/v1")

    def chat(self, messages) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
 

