"""Step 5 - Grounded AI explanations with Groq.

Grounding: the LLM receives ONLY the model's own facts (probability, threshold, amount,
hour, and the top SHAP contributions) and is told not to invent anything else.
If there is no API key, or Groq errors / rate-limits, we fall back to a deterministic
template built from the same SHAP facts - so the pipeline never breaks."""
from __future__ import annotations

import json
import time

from src import config

SYSTEM_PROMPT = (
    "You are a fraud-analyst assistant. In at most 2 short plain-English sentences, explain "
    "why a machine-learning model flagged a card transaction, for a non-technical reviewer. "
    "Use ONLY the facts provided. Name the top risk drivers and say whether each value was "
    "unusually high or low. Features V1..V28 are anonymized behaviour signals (PCA "
    "components): call them 'behaviour signal V14' and do not guess what they mean. Do not "
    "quote SHAP or contribution numbers, and round the probability to a whole percent. Do not "
    "invent merchants, locations, customers or any other fact. No preamble, no bullet points."
)


def template_explanation(prob: float, amount: float, top_features: list[dict]) -> str:
    """Deterministic fallback, built from the same SHAP facts the LLM would get."""
    parts = [f"{f['description']} (value {f['value']:.2f})" for f in top_features[:3]
             if f["contribution"] > 0]
    drivers = ", ".join(parts) if parts else "an unusual combination of signals"
    return (f"Flagged with {prob:.0%} fraud probability on a ${amount:,.2f} transaction; "
            f"the strongest risk drivers were {drivers}.")


class GroqExplainer:
    def __init__(self, api_key: str = config.GROQ_API_KEY, model: str = config.GROQ_MODEL):
        self.model = model
        self.client = None
        if api_key:
            from groq import Groq
            self.client = Groq(api_key=api_key, max_retries=0)  # we handle retries ourselves

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def explain(self, prob: float, threshold: float, amount: float, hour: float,
                top_features: list[dict], max_attempts: int = 3) -> tuple[str, str]:
        """Returns (text, source) where source is 'llm' or 'template'."""
        if not self.enabled:
            return template_explanation(prob, amount, top_features), "template"

        facts = {
            "fraud_probability": round(prob, 4),
            "decision_threshold": round(threshold, 4),
            "amount": amount,
            "hour_of_day": int(hour),
            "top_risk_drivers": [
                {"feature": f["feature"], "meaning": f["description"], "value": f["value"],
                 "direction": "increases fraud risk" if f["contribution"] > 0
                 else "decreases fraud risk", "shap_contribution": f["contribution"]}
                for f in top_features
            ],
        }
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                extra = {}
                if "gpt-oss" in self.model:
                    # gpt-oss "thinks" before answering; keep that short so the
                    # token budget is left for the actual explanation
                    extra["reasoning_effort"] = "low"
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_completion_tokens=500,   # room for reasoning + a 1-2 sentence answer
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": json.dumps(facts)}],
                    **extra,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text, "llm"
            except Exception as e:  # groq.RateLimitError, APIConnectionError, model retired...
                status = getattr(e, "status_code", None)
                if status in (401, 403, 404):
                    # Bad key or retired model: retrying won't help, so switch to templates
                    # for the rest of the run instead of failing on every alert.
                    hint = ("check GROQ_API_KEY in .env (no quotes/spaces), then restart"
                            if status in (401, 403) else
                            f"model '{self.model}' not found - update GROQ_MODEL in .env")
                    print(f"[groq] {status}: {hint}. Using template explanations from now on.")
                    self.client = None
                    break
                print(f"[groq] attempt {attempt} failed ({status or type(e).__name__}): {e}")
                if status == 429 and attempt < max_attempts:   # rate limited -> back off
                    time.sleep(delay)
                    delay *= 2
                    continue
                break
        return template_explanation(prob, amount, top_features), "template"
