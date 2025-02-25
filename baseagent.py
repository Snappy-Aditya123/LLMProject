import json
from openai import OpenAI

class BaseAgent:
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions  # Fixed typo from self.insctructions
        self.ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    async def run(self):
        raise NotImplementedError("Subclasses must implement the run method.")

    def _query_ollama(self, prompt, temperature=0.5, max_tokens=300):
        try:
            response = self.ollama_client.chat.completions.create(
                model="llama3",
                messages=[
                    {"role": "system", "content": self.instructions},  # Fixed typo
                    {"role": "user", "content": f"my cv: {prompt}"}
                ],
                temperature=temperature,  # Now accepts dynamic temperature
                max_tokens=max_tokens,  # Now accepts dynamic max_tokens
            )

            # Ensure response structure is correct
            if hasattr(response, "choices") and response.choices:
                print(response.choices[0].message.content)
                return response.choices[0].message.content
                
            return "Error: No valid response received from Ollama."

        except Exception as e:
            return f"API Error: {str(e)}"

    def _parse_json_safely(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start:end + 1]
                return json.loads(json_str)
            return {"error": "No JSON found"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON Parsing Error: {str(e)}"}
