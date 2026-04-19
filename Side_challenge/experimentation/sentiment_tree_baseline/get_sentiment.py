import boto3
import json
import pandas as pd
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
MAX_THREADS = 12  # Ein kleiner Tick mehr als 10, um Latenzen zu füllen
OUTPUT_PATH = "hybrid_sentiments_v2.csv"
# WECHSEL AUF HAIKU (Massiver Speed-Boost!)
MODEL_ID = 'anthropic.claude-3-5-haiku-20241022-v1:0' 

# Boto3 Client mit optimierter Retry-Logik
bedrock = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-west-2'
)

BULLISH_WORDS = [
    'profit', 'growth', 'upgraded', 'beat', 'partnership', 'expansion', 'buy', 
    'dividend', 'success', 'secures', 'contract', 'record', 'outperform', 
    'acquisition', 'buyback', 'surge', 'agreement', 'approval', 'positive',
    'increase', 'exceeds', 'revenue', 'initiative', 'strategic'
]

BEARISH_WORDS = [
    'loss', 'warning', 'downgraded', 'miss', 'debt', 'investigation', 'scandal', 
    'lawsuit', 'cut', 'decline', 'unexpected', 'underperform', 'bankruptcy', 
    'slashes', 'drop', 'negative', 'suspended', 'disruption', 'recall',
    'concerns', 'unfavorable', 'delay', 'class action', 'probe'
]
def get_keyword_score(text):
    if not text: return 0.0
    text = text.lower()
    pos = sum(1 for word in BULLISH_WORDS if word in text)
    neg = sum(1 for word in BEARISH_WORDS if word in text)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0

def get_claude_json(text):
    if not text or len(str(text)) < 10:
        return {"sentiment": 0.0, "confidence": 0.0}
    
    # Fokus auf nachhaltigen Trend statt kurzen Schock
    prompt = f"""Analysiere die Auswirkung dieser News auf den Aktienkurs bis zum Ende des Handelstages.
Antworte NUR mit JSON: {{'sentiment': float, 'confidence': float}}
Sentiment: -1.0 (sehr negativ) bis 1.0 (sehr positiv).
Confidence: 0.0 bis 1.0 (wie eindeutig ist die News?).
Headlines: {text}"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 40,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    })

    try:
        response = bedrock.invoke_model(modelId=MODEL_ID, body=body)
        score_text = json.loads(response.get('body').read())['content'][0]['text'].strip()
        score_text = re.sub(r"```json\n?|```", "", score_text).strip()
        return json.loads(score_text)
    except Exception:
        return {"sentiment": 0.0, "confidence": 0.0}

def process_session(row):
    c_json = get_claude_json(row['headline'])
    k_score = get_keyword_score(row['headline'])
    return {
        'session': row['session'], 
        'claude_sentiment': c_json.get('sentiment', 0.0),
        'claude_confidence': c_json.get('confidence', 0.0),
        'keyword_sentiment': k_score
    }

# --- DATEN LADEN ---
base_dir = "/home/participant/datathon-hrt-challenge-1/hrt-eth-zurich-datathon-2026/data"
files = ["headlines_seen_train.parquet", "headlines_seen_public_test.parquet", "headlines_seen_private_test.parquet"]
dfs = [pd.read_parquet(os.path.join(base_dir, f)) for f in files if os.path.exists(os.path.join(base_dir, f))]
df_h = pd.concat(dfs, ignore_index=True)
df_recent = df_h[df_h['bar_ix'] >= 29].copy()
# Gruppiere jetzt nur noch die frischen News!
grouped = df_recent.groupby('session')['headline'].apply(lambda x: " | ".join(x)).reset_index()

# --- RESUME LOGIK ---
results = []
already_done = set()
if os.path.exists(OUTPUT_PATH):
    old_df = pd.read_csv(OUTPUT_PATH)
    results = old_df.to_dict('records')
    already_done = set(old_df['session'].tolist())

todo = grouped[~grouped['session'].isin(already_done)].to_dict('records')

print(f"🚀 Turbo-Haiku: Analysiere {len(todo)} Sessions...")

# --- PROCESSING ---
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = {executor.submit(process_session, row): row for row in todo}
    for i, future in enumerate(as_completed(futures)):
        results.append(future.result())
        # Seltener speichern (alle 200 statt 100), spart I/O Zeit
        if (i + 1) % 200 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"Check: {i+1}/{len(todo)} fertig...")

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print("✅ FERTIG!")