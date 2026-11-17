# FraudPulse: Step-by-Step Guide (Windows + VS Code)

Do one step at a time. Each step has **Run**, **Test it** (how you know it worked), and **Talk about it** (interview notes).

All commands go in the **VS Code terminal (PowerShell)**, run from the project folder (`fraudpulse`).

---

## Step 1: Setup

**Run**
```powershell
cd path\to\fraudpulse
python --version                      # 3.11 or 3.12 recommended (3.13 also works)
python -m venv .venv                  # creates an isolated environment for this project
.\.venv\Scripts\Activate.ps1          # you should now see (.venv) at the start of the prompt
```
If activation fails with "running scripts is disabled", run this once and then activate again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Then install everything and create your `.env`:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
code .env                             # paste your Groq key after GROQ_API_KEY=
```
In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick `.venv`.

**Test it**
```powershell
python -c "import xgboost, fastapi, streamlit, groq; print('all good')"
```

**Files**
- `requirements.txt`: every library the project needs. `requirements-api.txt` is a slimmer list used only by Docker.
- `.env` / `.env.example`: secrets and settings. `.env` is git-ignored, so your key never reaches GitHub. `.env.example` is the template you commit.
- `src/config.py`: reads `.env` in one place, so nothing is hard-coded. The Groq model name lives here because Groq retires models.

---

## Step 2: Data and model training

**Get the data** (Kaggle "Credit Card Fraud Detection", 284,807 transactions, 0.172% fraud)

*Option A, browser (easiest):* sign in at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, click **Download**, unzip, and put `creditcard.csv` in the `data\` folder.

*Option B, Kaggle API:*
```powershell
pip install kaggle
# Kaggle > Settings > API > "Create New Token" downloads kaggle.json
mkdir $env:USERPROFILE\.kaggle -Force
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
python scripts\download_data.py
```

**Optional smoke test first** (fake data, about 5 seconds):
```powershell
python -m src.train --synthetic
```

**Train on the real data** (about 1–3 minutes):
```powershell
python -m src.train
```

**Test it.** You should see a table like this:
```
model                      thr    prec  recall      F1  PR-AUC     acc
logistic_regression        ...
xgboost                    ...
Naive 'always legit' accuracy: 0.9983  <- why accuracy is misleading
```
`models\` should now contain `xgb_model.json`, `model_meta.json` and `metrics.json`, and `data\stream.csv` should exist.

**Files**
- `src/features.py`: feature engineering (`log_amount`, `hour`). Training and the API both use this function, which prevents training/serving skew.
- `src/train.py`: stratified 60/20/20 split, then a logistic regression baseline and XGBoost. It picks the threshold on validation, evaluates on test, and saves everything. It also saves the test set as `stream.csv`, so the live stream only uses transactions the model has never seen.
- `src/synthetic.py`: makes fake data with the same columns, used by tests.

**Talk about it**
- *Imbalance:* "Only 0.17% of transactions are fraud. A model that always says 'legit' gets 99.8% accuracy and catches zero fraud. So I used class weighting (`scale_pos_weight` ≈ 578) and judged models by PR-AUC, precision and recall."
- *Why PR-AUC over ROC-AUC:* "ROC-AUC looks great on imbalanced data because there are so many true negatives. PR-AUC focuses on the positive class."
- *Threshold:* "Class weighting inflates probabilities, so 0.5 means nothing. I picked the threshold that maximizes F1 on the validation set and reported test metrics once. In a real bank you'd set it from the cost of a missed fraud vs. a blocked customer. That's why it can be overridden with `FRAUD_THRESHOLD`."
- *Early stopping:* "XGBoost stops adding trees when validation PR-AUC stops improving, and I save only the best trees."

---

## Step 3: Scoring API (FastAPI)

**Run** (Terminal 1, leave it running):
```powershell
uvicorn src.api.main:app --port 8000
```

**Test it**
- Open http://127.0.0.1:8000/docs. This is interactive Swagger documentation generated automatically. Try `/health`.
- In a **second terminal** (activate the venv first with `.\.venv\Scripts\Activate.ps1`):
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
python scripts\try_api.py      # sends a legit tx, a fraud tx, and an invalid one (gets 422)
```

**Files**
- `src/api/schemas.py`: Pydantic models. They reject missing fields, negative amounts, NaN and unknown fields with a 422 error before the model ever runs.
- `src/api/model_service.py`: loads the model once and scores. It uses XGBoost's built-in **TreeSHAP** (`pred_contribs=True`) to return the top 3 features that pushed the transaction toward fraud.
- `src/api/main.py`: the endpoints. The model loads at startup (`lifespan`), not per request. `/score` is a plain `def`, so FastAPI runs it in a thread pool, which suits CPU-bound work.

**Talk about it**: "The API only scores. It never waits on the LLM, so p95 latency stays low even if Groq is slow or rate-limited."

---

## Step 4: Real-time stream simulator

**Run** (Terminal 2, with the API still running in Terminal 1):
```powershell
python -m src.stream.simulator --rate 5 --limit 500 --fraud-ratio 0.02 --reset-db
```
- `--rate`: transactions per second
- `--fraud-ratio 0.02`: demo mode that enriches the stream to 2% fraud so alerts show up quickly. Use `0` for the real 0.17% rate.
- `--loop`: keep going forever (stop with `Ctrl+C`)

**Test it.** You'll see `ALERT ...` lines and progress counts, and `data\transactions.db` gets created.

**Files**
- `src/stream/sources.py`: the `TransactionSource` interface. `CsvReplaySource` replays `stream.csv`, and `KafkaSource` is a documented stub.
- `src/stream/simulator.py`: paces requests at a fixed rate, stores every result in SQLite, and sends flagged ones to the explanation worker.
- `src/db.py`: SQLite in WAL mode, so the dashboard can read while the simulator writes.

**Talk about it**: "I skipped Kafka to keep setup simple, but the simulator depends on an abstract source. Adding Kafka means one new class that yields messages from a topic, and the rest of the pipeline doesn't change."

---

## Step 5: AI explanations (Groq)

These are already wired into Step 4. Check that `GROQ_API_KEY` is set in `.env`, then rerun the simulator. The first line should say `LLM explanations: Groq <model>`.

**Test it**
```powershell
python -c "import sqlite3; c=sqlite3.connect('data/transactions.db'); [print(r) for r in c.execute('select explanation_source, explanation from transactions where is_flagged=1 limit 5')]"
```
To check the key on its own, without the simulator:
```powershell
python -c "from groq import Groq; from src import config; print(Groq(api_key=config.GROQ_API_KEY).chat.completions.create(model=config.GROQ_MODEL, messages=[{'role':'user','content':'Say OK'}]).choices[0].message.content)"
```
It should print something like `OK`.

`explanation_source` should be `llm`. If it says `template`, the key is missing, or the model name is retired (check https://console.groq.com/docs/models and update `GROQ_MODEL`).

**Files**
- `src/explain/llm.py`: builds a facts-only prompt (probability, threshold, amount, hour, top SHAP drivers) and tells the LLM to use only those facts. If anything fails, it falls back to a deterministic template.
- `src/explain/worker.py`: a background thread with a **bounded queue** and a **rate limiter** (`GROQ_MAX_RPM`). Only flagged transactions are queued. If the queue fills up, the overflow gets an instant template explanation (backpressure).

**Talk about it**: "To avoid hallucinated reasons, the LLM never sees raw data or gets to guess. It only rewrites the model's own SHAP contributions in plain English. It runs asynchronously and is rate-limited for the free tier."

---

## Step 6: Live dashboard

**Run** (Terminal 3):
```powershell
streamlit run src/dashboard/app.py
```
It opens http://localhost:8501. Keep the simulator running (`--loop`) to watch it update.

**What's on it**: KPIs (scored, flagged, live precision/recall, p95 latency), fraud rate over time (model vs. actual), the live feed, alert cards with AI explanations and SHAP drivers, test-set metrics with the PR curve and feature importance, and load-test results.

**File**: `src/dashboard/app.py`. `@st.fragment(run_every=...)` re-renders only the live section every few seconds.

---

## Step 7: Load test

Stop the simulator first. Restart the API **without** `--reload`, with access logs off (printing one log line per request to the Windows console slows the server down noticeably) and with several worker processes so it uses more than one CPU core:
```powershell
# Terminal 1
uvicorn src.api.main:app --port 8000 --workers 4 --no-access-log
# Terminal 2
python -m src.loadtest --requests 2000 --concurrency 20
```
Compare against `--workers 1` to show the scaling in interviews. Every run is saved to `results\loadtest_history.jsonl`, and the dashboard shows them all in an "All runs" table.
**Test it.** It prints throughput (req/s) and latency (p50/p95/p99), and saves `results\loadtest_latest.json`, which the dashboard displays.

Try `--concurrency 1`, `10` and `50` to see how throughput and latency trade off.

**File**: `src/loadtest.py`, an asyncio + httpx client with N concurrent workers. It runs a warm-up first and then measures only successful requests.

**Resume line example**: *"Served XGBoost fraud scoring via FastAPI at ~X req/s with p95 latency of Y ms (2,000 requests, 20 concurrent clients, single worker)."* Use your real numbers.

---

## Step 8: Tests, Docker, README

**Tests** (self-contained: they train a tiny model on synthetic data):
```powershell
python -m pytest -v
```
These cover health, a valid score, fraud scoring higher than legit, 422s for a missing field, a negative amount and an unknown field, and the explanation fallback.

**Docker** (optional; needs Docker Desktop, and you must train first so `models\` exists):
```powershell
docker build -t fraudpulse-api .
docker run -p 8000:8000 fraudpulse-api
```

**README results**:
```powershell
python -m src.report
```
Paste the printed tables into the Results section of `README.md`.

**Push to GitHub**:
```powershell
git init
git add .
git commit -m "FraudPulse: real-time AI fraud detection"
```
`.gitignore` already keeps `.env`, the dataset and the models out of git.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: src` | Run commands from the project root, using `python -m ...` |
| Simulator says "Waiting for API" | Start the API in another terminal first |
| Explanations say `template` | Check `GROQ_API_KEY`, then `GROQ_MODEL` against Groq's current model list |
| Groq `401 Invalid API Key` | In `.env` the line must be exactly `GROQ_API_KEY=gsk_...` with no quotes, spaces or `<>`. Save the file, then restart the simulator. Test the key with the one-liner in Step 5. |
| Groq 429 errors | Lower `GROQ_MAX_RPM` in `.env` |
| Port 8000 busy | `uvicorn src.api.main:app --port 8001` and set `API_URL=http://127.0.0.1:8001` in `.env` |
