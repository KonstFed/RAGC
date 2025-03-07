from pathlib import Path

import ollama

from ragc.graphs import SemanticParser
from ragc.retrieval.node_retrieval import LowGranularityRetrieval
from ragc.utils import load_secrets

SECRETS = load_secrets()


def generate_prompt(query, relevant_snippets):
    prompt = "You are an expert programming assistant. Your task is to help me with the following snippets that may help you:\n\n"

    for i, (file_id, code) in enumerate(relevant_snippets, start=1):
        c_snippet = f"{i}. **Code Snippet {i}** from **{file_id}**:\n```\n{code}\n```\n"
        prompt += c_snippet + "\n"

    prompt += f"**Question/Task**: {query}\n\n"
    prompt += """Provide the following in your response:
- A clear explanation of your solution or answer.
- The modified or generated code (if applicable).
- Any additional notes or best practices related to the task.

Let’s get started!"""

    return prompt


def inference(repo_path: Path, query: str) -> str:
    client = ollama.Client(host=SECRETS["OLLAMA_URL"])
    parser = SemanticParser()
    retrieval = LowGranularityRetrieval(repo_path, parser, "unclemusclez/jina-embeddings-v2-base-code")

    relevant_snippets = retrieval.retrieve(query, 5)
    prompt = generate_prompt(query, relevant_snippets)

    response = client.generate(
        model="deepseek-r1",
        prompt=prompt,
    )
    return response.response


if __name__ == "__main__":
    repo_path = Path("/home/konstfed/Documents/diplom/RAGC/data/holostyak-bot")
    query = "Create DB for admin users"
    print(inference(repo_path, query))
