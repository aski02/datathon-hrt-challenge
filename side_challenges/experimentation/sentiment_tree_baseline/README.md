# Sentiment Tree Baseline

This folder contains side-challenge experimentation files from the hackathon.
It is separate from the main HRT submission pipeline, but the scripts are kept
as runnable experiment code.

Files:

- `build_sentiment_features.py`
  - script for generating session-level sentiment signals from headline text
  - uses Bedrock / Claude and keyword sentiment
- `model_risk_utils.py`
  - shared risk-feature and base-strategy utilities used by the side-challenge model
- `extra_trees_bad_tail_probability_sizer.py`
  - ExtraTrees long-only bad-tail sizing strategy built on the local utilities

Notes:

- These files were normalized from the original local names:
  - `get_sentiment2.py` -> `build_sentiment_features.py`
  - `model_risk_utils2.py`
  - `extra_trees_bad_tail_probability_sizer-3.py`
- The strategy import was adjusted to use the local `model_risk_utils.py`
  so the folder is self-contained.
- `build_sentiment_features.py` expects AWS Bedrock access and `boto3`, which is not part
  of the main repo requirements.
- Use `--data-dir /path/to/hrt-eth-zurich-datathon-2026/data` to run it
  against a non-default data location.
