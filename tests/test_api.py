"""API tests. Run:  pytest -v"""
from src.explain.llm import GroqExplainer, template_explanation


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert 0 < body["threshold"] < 1


def test_score_returns_probability_and_flag(client, legit_tx):
    r = client.post("/score", json={**legit_tx, "transaction_id": "t-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "t-1"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] is False
    assert body["top_features"] == []          # fast path: no SHAP for legit tx


def test_flagged_tx_gets_shap_features(client, fraud_tx):
    body = client.post("/score", json=fraud_tx).json()
    assert body["is_fraud"] is True
    assert len(body["top_features"]) == 3
    assert body["top_features"][0]["contribution"] > 0


def test_explain_param_forces_shap(client, legit_tx):
    body = client.post("/score?explain=true", json=legit_tx).json()
    assert len(body["top_features"]) == 3


def test_fraud_scores_higher_than_legit(client, legit_tx, fraud_tx):
    p_legit = client.post("/score", json=legit_tx).json()["fraud_probability"]
    p_fraud = client.post("/score", json=fraud_tx).json()["fraud_probability"]
    assert p_fraud > p_legit


def test_missing_field_is_rejected(client, legit_tx):
    tx = dict(legit_tx)
    tx.pop("V14")
    assert client.post("/score", json=tx).status_code == 422


def test_negative_amount_is_rejected(client, legit_tx):
    assert client.post("/score", json={**legit_tx, "Amount": -10}).status_code == 422


def test_unknown_field_is_rejected(client, legit_tx):
    assert client.post("/score", json={**legit_tx, "Class": 1}).status_code == 422


def test_template_explanation_is_grounded():
    feats = [{"feature": "V14", "description": "anonymized behaviour signal V14",
              "value": -7.5, "contribution": 2.1}]
    text = template_explanation(0.97, 250.0, feats)
    assert "V14" in text and "97%" in text


def test_bad_key_disables_llm_and_falls_back():
    class _AuthError(Exception):
        status_code = 401

    class _FakeCompletions:
        def create(self, **_):
            raise _AuthError("Invalid API Key")

    ex = GroqExplainer(api_key="")
    ex.client = type("C", (), {"chat": type("Ch", (), {"completions": _FakeCompletions()})()})()
    text, source = ex.explain(0.9, 0.5, 10.0, 3, [])
    assert source == "template" and ex.enabled is False   # stops calling Groq after a 401


def test_explainer_without_key_falls_back_to_template():
    text, source = GroqExplainer(api_key="").explain(0.9, 0.5, 10.0, 3, [])
    assert source == "template" and text
