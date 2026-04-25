from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_OUTPUT_PATH = Path("hybrid_sentiments_v2.csv")
DEFAULT_MAX_THREADS = 12
RECENT_HEADLINE_START_BAR = 29

BULLISH_WORDS = (
    "profit",
    "growth",
    "upgraded",
    "beat",
    "partnership",
    "expansion",
    "buy",
    "dividend",
    "success",
    "secures",
    "contract",
    "record",
    "outperform",
    "acquisition",
    "buyback",
    "surge",
    "agreement",
    "approval",
    "positive",
    "increase",
    "exceeds",
    "revenue",
    "initiative",
    "strategic",
)

BEARISH_WORDS = (
    "loss",
    "warning",
    "downgraded",
    "miss",
    "debt",
    "investigation",
    "scandal",
    "lawsuit",
    "cut",
    "decline",
    "unexpected",
    "underperform",
    "bankruptcy",
    "slashes",
    "drop",
    "negative",
    "suspended",
    "disruption",
    "recall",
    "concerns",
    "unfavorable",
    "delay",
    "class action",
    "probe",
)


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "hrt-eth-zurich-datathon-2026" / "data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build session-level headline sentiment features with AWS Bedrock Claude and keyword scores."
    )
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-threads", type=int, default=DEFAULT_MAX_THREADS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--recent-start-bar", type=int, default=RECENT_HEADLINE_START_BAR)
    return parser.parse_args()


def keyword_score(text: str) -> float:
    if not text:
        return 0.0
    normalized = text.lower()
    positive = sum(1 for word in BULLISH_WORDS if word in normalized)
    negative = sum(1 for word in BEARISH_WORDS if word in normalized)
    total = positive + negative
    return (positive - negative) / total if total > 0 else 0.0


def claude_sentiment(client: Any, model_id: str, text: str) -> dict[str, float]:
    if not text or len(str(text)) < 10:
        return {"sentiment": 0.0, "confidence": 0.0}

    prompt = f"""Analyze the likely end-of-session stock-price impact of these headlines.
Return only JSON: {{"sentiment": float, "confidence": float}}
Sentiment ranges from -1.0 for very negative to 1.0 for very positive.
Confidence ranges from 0.0 to 1.0.
Headlines: {text}"""

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 40,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
    )

    try:
        response = client.invoke_model(modelId=model_id, body=body)
        score_text = json.loads(response.get("body").read())["content"][0]["text"].strip()
        score_text = re.sub(r"```json\n?|```", "", score_text).strip()
        parsed = json.loads(score_text)
        return {
            "sentiment": float(parsed.get("sentiment", 0.0)),
            "confidence": float(parsed.get("confidence", 0.0)),
        }
    except Exception:
        return {"sentiment": 0.0, "confidence": 0.0}


def load_recent_headline_docs(data_dir: Path, recent_start_bar: int) -> pd.DataFrame:
    input_files = (
        "headlines_seen_train.parquet",
        "headlines_seen_public_test.parquet",
        "headlines_seen_private_test.parquet",
    )
    frames = [pd.read_parquet(data_dir / filename) for filename in input_files if (data_dir / filename).exists()]
    if not frames:
        raise FileNotFoundError(f"No headline parquet files found in {data_dir}")

    headlines = pd.concat(frames, ignore_index=True)
    recent = headlines.loc[headlines["bar_ix"] >= recent_start_bar].copy()
    return recent.groupby("session", sort=True)["headline"].apply(" | ".join).reset_index()


def process_session(client: Any, model_id: str, row: dict[str, Any]) -> dict[str, float | int]:
    text = str(row["headline"])
    claude = claude_sentiment(client, model_id, text)
    return {
        "session": int(row["session"]),
        "claude_sentiment": claude["sentiment"],
        "claude_confidence": claude["confidence"],
        "keyword_sentiment": keyword_score(text),
    }


def write_results(results: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["session", "claude_sentiment", "claude_confidence", "keyword_sentiment"]
    frame = pd.DataFrame(results, columns=columns)
    if not frame.empty:
        frame = frame.sort_values("session")
    frame.to_csv(output_path, index=False)


def build_sentiment_features(args: argparse.Namespace) -> None:
    import boto3

    client = boto3.client(service_name="bedrock-runtime", region_name=args.region)
    grouped = load_recent_headline_docs(args.data_dir, args.recent_start_bar)

    results: list[dict[str, float | int]] = []
    completed_sessions: set[int] = set()
    if args.output.exists():
        existing = pd.read_csv(args.output)
        results = existing.to_dict("records")
        completed_sessions = set(existing["session"].astype(int).tolist())

    todo = grouped.loc[~grouped["session"].isin(completed_sessions)].to_dict("records")
    print(f"Building sentiment features for {len(todo)} remaining sessions.")

    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {executor.submit(process_session, client, args.model_id, row): row for row in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 200 == 0:
                write_results(results, args.output)
                print(f"Processed {index}/{len(todo)} sessions.")

    write_results(results, args.output)
    print(f"Wrote {args.output}")


def main() -> None:
    build_sentiment_features(_parse_args())


if __name__ == "__main__":
    main()
