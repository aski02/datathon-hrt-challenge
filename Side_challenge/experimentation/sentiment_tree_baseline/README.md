# Sentiment Tree Baseline

This folder contains side-challenge experimentation files copied from local
working drafts and grouped into one place.

Files:

- `get_sentiment.py`
  - script for generating session-level sentiment signals from headline text
  - uses Bedrock / Claude and keyword sentiment
- `model_risk_utils.py`
  - shared risk-feature and base-strategy utilities used by the side-challenge model
- `extra_trees_bad_tail_probability_sizer.py`
  - ExtraTrees long-only bad-tail sizing strategy built on the local utilities

Notes:

- These files were normalized from the original local names:
  - `get_sentiment2.py`
  - `model_risk_utils2.py`
  - `extra_trees_bad_tail_probability_sizer-3.py`
- The strategy import was adjusted to use the local `model_risk_utils.py`
  so the folder is self-contained.
- `get_sentiment.py` expects AWS Bedrock access and `boto3`, which is not part
  of the main repo requirements.
