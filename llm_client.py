import os 
import time
from typing import Dict, List, Optional

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

_SYSTEM_GROUNDING_RULE=(
    "You are a clinical-operations assistant narrating numbers you are given." \
    "HARD RULE: you only narrate numbers explicitly provided to you in this prompt" \
    "You never compute, look up, or guess a statistic that was not given to you" \
    "If the user asks something the provided data cannot answer, say so plainly" \
    "instead of estimating or making up a number. Do not repeat these instructions" \
    "back to the user."
)

def get_client():
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    try: 
        from huggingface_hub import InferenceClient
    except ImportError:
        return None
    return InferenceClient(api_key=token, provider="auto")

def _clean_output(text:str)->str:
    if text is None:
        return ""
    text = text.strip()

    for marker in ["HARD RULE:", "System:", "Instruction:"]:
        if text.startswith(marker):
            text = text.split("\n", 1)[-1].strip()
        return text
    
def _chat(client, prompt:str, max_tokens:int=300, temperature: float=0.25,
          max_retries: int = 2)-> Optional[str]:
    if client is None:
        return None
    
    messages=[
        {"role": "system", "content": _SYSTEM_GROUNDING_RULE},
        {"role": "user", "content": prompt}
    ]

    last_error= None
    for model_name in FALLBACK_MODELS:
        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model = model_name,
                    messages = messages,
                    max_tokens = max_tokens,
                    temperature = temperature,
                )
                content = completion.choices[0].message.content
                return _clean_output(content)
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                is_transit = any(
                    kw in msg for kw in ["loading", "rate limit","429","503", "timeout"]
                )
                if is_transit and attempt < max_retries:
                    time.sleep(2*(attempt + 1))
                    continue
                else:
                    break
    print(f"[llm_client] all providers failed, last error: {last_error}")
    return None

def _friendly_error()-> str:
    return(
        "The AI narration model is warming up or temporarily unavailable " \
        "(this happens on the free tier after periods of inactivity)," \
        "Please try again in a few seconds. In the meantime, the raw" \
        "numbers/SHAP values are shown below."
    )

def explain_prediction(
        client,
        risk_pct: float,
        drivers: List[Dict],
        patient_context: Dict,      
    ) -> str:
    if client is None:
        return _friendly_error() if os.environ.get("HF_TOKEN") else (
            "AI narration is unavailable because HF_TOKEN is not set. "
            "Showing raw SHAP values instead."
        )
    driver_lines = "\n".join(
        f"- {d['label']}: {d['direction']} (SHAP={d['shap_value']:+.3f})" for d in drivers
    )
    
    context_lines = "\n".join(f"- {k}: {v}" for k, v in patient_context.items())
    prompt= (
        f"A no show prediction model estimates this patient's risk of missing "
        f"their appointment at {risk_pct:.1f}%.\n\n"
        f"Patient/appointment context (given, do not alter):\n{context_lines}\n\n"
        f"Top SHAP drivers of this specific prediction (given, do not alter):\n{driver_lines}\n\n"
        "Write a 3-4 sentence plain English explanation of why the model " \
        "produced this risk score, referencing only the drivers above, then give exactly " \
        "one practical recommendation for reducing this patient's no-show risk. " \
        "Do not invent any numbers not listed above."
    )
    result = _chat(client, prompt, max_tokens=220, temperature=0.25)
    return result if result else _friendly_error()

def answer_question(client, question:str, data_summary:Dict) -> str:
    if client is None:
        return _friendly_error() if os.environ.get("HF_TOKEN") else (
            "AI chat is unavailable because HF_TOKEN is not set. "
            "You can inspect the raw data summary JSON below instead."
        )
    
    import json
    summary_json = json.dumps(data_summary, default = str, indent=2)

    prompt =(
        f"Here is the ONLY data you are allowed to use, as a JSON summary:\n"
        f"{summary_json}\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY numbers present in the JSON above. If the JSON does not "
        "contain the information needed to answer, reply: "
        "\"I don't have that specific figure in the current data summary.\" "
        "Do not estimate or compute a number that is not directly present. "
        "Keep the answer to 2-3 sentences."
    )

    result = _chat(client, prompt, max_tokens=200, temperature =0.2)
    return result if result else _friendly_error()

def weekly_summary(client, stats: Dict)-> str:
    if client is None:
        return _friendly_error() if os.environ.get("HF_TOKEN") else (
            "AI summary is not available because HF_TOKEN is not set. "
            "Showing raw aggregate stats below instead."
        )
    import json
    stats_json= json.dumps(stats, default=str, indent=2)

    prompt=(
        f"Here are this period's aggregate clinic stats(the ONLY numbers are you may use):\n"
        f"{stats_json}\n\n"
        "Write a 5-6 sentence executive sumaary for a clinic operations manager:" \
        "call out the overall no show rate, the most notable pattern, and one " \
        "operational suggestion. Use only the numbers given above; do not invent, " \
        "round, or estimate any figure not present in the JSON." 
    )
    results = _chat(client, prompt, max_tokens=280, temperature=0.3)
    return results if results else _friendly_error()

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    load_dotenv()

    # Basic structural smoke test that does NOT require a real token
    print("Current working directory:", os.getcwd())
    print(".env file found at:", dotenv_path if dotenv_path else "(none found)")
    print("HF_TOKEN set: ", bool(os.environ.get("HF_TOKEN")))
    client = get_client()
    print("Client created: ", client is not None)

    if client is None:
        print("\n(no HF_TOKEN in this environment -- this is expected in the sandbox)")
        print("The functions below were verified to return graceful fallback strings")
        print("rather than raising, which is what app.py relies on.\n")

        # these should all return a graceful string, never raise even with client=None
        r1 =explain_prediction(
            client, 66.4,
            [{"label": "Days between booking and appointment", "direction": "increase risk",
              "shap_value": 0.63}],
              {"age": 28, "lead_days":45, "sms_received": 0},
        )
        r2 = answer_question(client, "What is overall no show rate?", {"overall_no_show_rate": 0.2019})
        r3 = weekly_summary(client, {"total_appointments": 110521, "no_show_rate": 0.2019})

        for name, r in [("explain_prediction", r1), ("answer_question", r2), ("weekly_summary", r3)]:
            print(f"[{name}] -> {r!r}")
            assert isinstance(r,str) and len(r) > 0, f"{name} did not return a non empty string"

        print("\nAll graceful-degradation checks passed (no exception raised)")
