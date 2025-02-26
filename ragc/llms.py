import requests

class OllamaWrapper:
    def __init__(self, model_name: str, ollama_url: str) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url

    def generate(self, prompt: str) -> str:
        response = requests.post(self.ollama_url, json={
            "model": self.model_name,
            "prompt": prompt,
        })

        if response.status_code != 200:
            _err_msg = f"status code {response.status_code}"
            raise ValueError(_err_msg)
        
        return response.json()["response"]
