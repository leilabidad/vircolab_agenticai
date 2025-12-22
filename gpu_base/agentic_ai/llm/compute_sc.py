import requests
import json
import torch
import time

few_shot_block = """
Input: Patient has severe chest pain and shortness of breath.
Output: {"risk_level":"high","score":0.95,"issues":["chest pain","shortness of breath"]}

Input: Patient reports mild headache only.
Output: {"risk_level":"low","score":0.2,"issues":["headache"]}
"""

def compute_sc_batch(llm_text_batch):
    results = []
    url = "http://localhost:11434/api/generate"

    for llm_text in llm_text_batch:
        prompt = f"""
You must output ONLY valid JSON following the schema.

SCHEMA:
{{
    "risk_level": "string",
    "score": "number",
    "issues": ["array of strings"]
}}

EXAMPLES:
{few_shot_block}

INPUT:
{llm_text}
"""

        payload = {
            "model": "llama3.1:70b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }

        resp = requests.post(url, json=payload, timeout=30)
        data = resp.json()
        raw_model_output = data.get("response", "")

        start_json = raw_model_output.find("{")
        end_json = raw_model_output.rfind("}")
        if start_json == -1 or end_json == -1:
            results.append({"score": torch.tensor(0.0), "issues": []})
            continue

        extracted = raw_model_output[start_json:end_json + 1]
        try:
            parsed = json.loads(extracted)
            parsed["score"] = torch.tensor(parsed.get("score", 0.0))
            results.append(parsed)
        except:
            results.append({"score": torch.tensor(0.0), "issues": []})

    return results
