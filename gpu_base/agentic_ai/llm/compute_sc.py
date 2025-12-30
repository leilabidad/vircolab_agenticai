import requests
import json
import torch

few_shot_block = """
Input: Patient has severe chest pain and shortness of breath.
Output: {"risk_level":"high","score":0.95,"issues":["chest pain","shortness of breath"]}

Input: Patient reports mild headache only.
Output: {"risk_level":"low","score":0.2,"issues":["headache"]}
"""

def extract_json(text):
    if not isinstance(text, str):
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s:e+1]

def validate_schema(obj):
    if not isinstance(obj, dict):
        return None
    if obj.get("risk_level") not in {"low", "medium", "high"}:
        return None
    try:
        score = float(obj.get("score"))
    except:
        return None
    issues = obj.get("issues")
    if not isinstance(issues, list):
        return None
    return {
        "risk_level": obj["risk_level"],
        "score": torch.tensor(score),
        "issues": [str(i) for i in issues]
    }

def compute_sc_batch(llm_text_batch):
    results = []
    url = "http://localhost:11434/api/generate"

    for llm_text in llm_text_batch:
        prompt = f"""
You must output ONLY valid JSON.

SCHEMA:
{{
    "risk_level": "low | medium | high",
    "score": number,
    "issues": array
}}

EXAMPLES:
{few_shot_block}

INPUT:
{llm_text}
"""
        try:
            resp = requests.post(url, json={
                "model": "llama3.1:70b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            }, timeout=30)
            raw_output = resp.json().get("response", "")
            extracted = extract_json(raw_output)
            if extracted:
                parsed = json.loads(extracted)
                validated = validate_schema(parsed)
                if validated:
                    results.append(validated)
                    continue
            results.append({"risk_level": None, "score": torch.tensor(0.0), "issues": []})
        except:
            results.append({"risk_level": None, "score": torch.tensor(0.0), "issues": []})

    return results

