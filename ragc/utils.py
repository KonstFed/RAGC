import json


def load_secrets() -> dict:
    with open(".secrets.json") as f:
        settings = json.load(f)

    return settings