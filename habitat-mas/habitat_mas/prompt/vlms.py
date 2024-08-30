"""VLM Helper Functions."""
import base64
import numpy as np
from openai import OpenAI


class GPT4V:
    """GPT4V VLM."""

    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)

    def query(self, prompt_seq, temperature=0, max_tokens=512):
        """Queries GPT-4V."""
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                content.append({'type': 'text', 'text': elem})
            elif isinstance(elem, np.ndarray):
                base64_image_str = base64.b64encode(elem).decode('utf-8')
                image_url = f'data:image/jpeg;base64,{base64_image_str}'
                content.append(
                    {'type': 'image_url', 'image_url': {'url': image_url}})

        messages = [{'role': 'user', 'content': content}]

        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content
