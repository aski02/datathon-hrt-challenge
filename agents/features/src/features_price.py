"""Price-only session feature engineering for Zurich Datathon 2026.

This module builds leakage-safe, deterministic, one-row-per-session features
from the seen half of each session.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_BAR_COLUMNS: tuple[str, ...] = ("session", "bar_ix", "open", "high", "low", "close")
DEFAULT_SEEN_BARS = 50
DEFAULT_WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 50)
DEFAULT_DCT_COEFFS = 10
EPSILON = 1e-12


def build_price_features(
    seen_bars: pd.DataFrame,
    *,
    expected_bars: int = DEFAULT_SEEN_BARS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    dct_coeffs: int = DEFAULT_DCT_COEFFS,
    eps: float = EPSILON,
) -> pd.DataFrame:
    """Build deterministic session-level features from seen bars only.

    Parameters
    ----------
    seen_bars:
        Bar-level input with columns: session, bar_ix, open, high, low, close.
    expected_bars:
        Fixed bar width for feature blocks (50 in this challenge).
    windows:
        Window lengths for aggregate summary features.
    dct_coeffs:
        Number of DCT coefficients from normalized log-close path.
    eps:
        Small epsilon for numerical safety.

    Returns
    -------
    pd.DataFrame
        Session-indexed feature table (one row per session).
    """
    bars = _prepare_seen_bars(seen_bars)
    sessions = pd.Index(sorted(bars["session"].unique()), name="session")
    bar_index = pd.Index(range(expected_bars), name="bar_ix")

    open_m = _pivot_matrix(bars, sessions, bar_index, "open")
    high_m = _pivot_matrix(bars, sessions, bar_index, "high")
    low_m = _pivot_matrix(bars, sessions, bar_index, "low")
    close_m = _pivot_matrix(bars, sessions, bar_index, "close")

    log_open = _safe_log(open_m, eps)
    log_high = _safe_log(high_m, eps)
    log_low = _safe_log(low_m, eps)
    log_close = _safe_log(close_m, eps)

    ret_cc = np.diff(log_close, axis=1)
    ret_oc = log_close - log_open
    range_hl = log_high - log_low

    prev_close = np.concatenate([open_m[:, :1], close_m[:, :-1]], axis=1)
    gap_prev_close = _safe_log(open_m, eps) - _safe_log(prev_close, eps)
    gap_prev_close[:, 0] = 0.0

    bar_range = (high_m - low_m) + eps
    candle_loc = (close_m - low_m) / bar_range
    body_norm = (close_m - open_m) / bar_range

    close_norm = log_close - log_close[:, :1]
    close_norm = _finite(close_norm)
    dct_close = _dct_type_ii(close_norm, n_coeff=dct_coeffs)

    frames: list[pd.DataFrame] = []
    frames.append(_matrix_to_frame(ret_cc, "ret_cc", start_ix=1, index=sessions))
    frames.append(_matrix_to_frame(ret_oc, "ret_oc", start_ix=0, index=sessions))
    frames.append(_matrix_to_frame(range_hl, "range_hl", start_ix=0, index=sessions))
    frames.append(_matrix_to_frame(gap_prev_close, "gap_prev_close", start_ix=0, index=sessions))
    frames.append(_matrix_to_frame(candle_loc, "candle_loc", start_ix=0, index=sessions))
    frames.append(_matrix_to_frame(body_norm, "body_norm", start_ix=0, index=sessions))

    summary_frame = _window_summary_features(
        close_m=close_m,
        ret_cc=ret_cc,
        range_hl=range_hl,
        body_norm=body_norm,
        windows=tuple(sorted(set(int(w) for w in windows))),
        eps=eps,
        index=sessions,
    )
    frames.append(summary_frame)

    close_mean = _row_mean(close_m)
    close_std = _row_std(close_m)
    close_last = _row_last(close_m)
    close_min = _row_min(close_m)
    close_max = _row_max(close_m)

    global_frame = pd.DataFrame(
        {
            "last_close_zscore": _finite((close_last - close_mean) / (close_std + eps)),
            "last_close_pos_in_seen_range": _finite((close_last - close_min) / ((close_max - close_min) + eps)),
        },
        index=sessions,
    )
    frames.append(global_frame)

    # Raw normalized close path block for later train-only compression (e.g. PCA).
    frames.append(_matrix_to_frame(close_norm, "close_norm", start_ix=0, index=sessions))
    frames.append(_matrix_to_frame(dct_close, "dct_close", start_ix=0, index=sessions))

    features = pd.concat(frames, axis=1)
    features = features.sort_index()
    features = features.astype(float)
    return features


def build_close_path_block(
    seen_bars: pd.DataFrame,
    *,
    expected_bars: int = DEFAULT_SEEN_BARS,
    eps: float = EPSILON,
) -> pd.DataFrame:
    """Return a fixed-width normalized close path block suitable for PCA."""
    bars = _prepare_seen_bars(seen_bars)
    sessions = pd.Index(sorted(bars["session"].unique()), name="session")
    bar_index = pd.Index(range(expected_bars), name="bar_ix")

    close_m = _pivot_matrix(bars, sessions, bar_index, "close")
    close_norm = _safe_log(close_m, eps) - _safe_log(close_m[:, :1], eps)
    return _matrix_to_frame(close_norm, "close_norm", start_ix=0, index=sessions)


def build_train_target(seen_train: pd.DataFrame, unseen_train: pd.DataFrame, *, eps: float = EPSILON) -> pd.Series:
    """Compute y = close_end / close_halfway - 1 for each training session."""
    seen = _prepare_seen_bars(seen_train)
    unseen = _prepare_seen_bars(unseen_train)

    close_halfway = seen.groupby("session", sort=True)["close"].last()
    close_end = unseen.groupby("session", sort=True)["close"].last()

    missing_unseen = close_halfway.index.difference(close_end.index)
    missing_seen = close_end.index.difference(close_halfway.index)
    if len(missing_unseen) > 0 or len(missing_seen) > 0:
        raise ValueError(
            "Session mismatch between seen and unseen train bars "
            f"(missing_unseen={len(missing_unseen)}, missing_seen={len(missing_seen)})."
        )

    target = close_end / np.maximum(close_halfway, eps) - 1.0
    target = target.rename("target_return")
    return target.astype(float)


def build_train_set(
    seen_train: pd.DataFrame,
    unseen_train: pd.DataFrame,
    *,
    expected_bars: int = DEFAULT_SEEN_BARS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    dct_coeffs: int = DEFAULT_DCT_COEFFS,
    eps: float = EPSILON,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build train features and target aligned by session index."""
    x_train = build_price_features(
        seen_train,
        expected_bars=expected_bars,
        windows=windows,
        dct_coeffs=dct_coeffs,
        eps=eps,
    )
    y_train = build_train_target(seen_train, unseen_train, eps=eps).reindex(x_train.index)

    if y_train.isna().any():
        missing = int(y_train.isna().sum())
        raise ValueError(f"Missing train targets after alignment (missing={missing}).")

    return x_train, y_train


def build_test_set(
    seen_test: pd.DataFrame,
    *,
    expected_bars: int = DEFAULT_SEEN_BARS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    dct_coeffs: int = DEFAULT_DCT_COEFFS,
    eps: float = EPSILON,
) -> pd.DataFrame:
    """Build leakage-safe session-level test features from seen test bars."""
    return build_price_features(
        seen_test,
        expected_bars=expected_bars,
        windows=windows,
        dct_coeffs=dct_coeffs,
        eps=eps,
    )


def _prepare_seen_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in REQUIRED_BAR_COLUMNS if col not in bars.columns]
    if missing_cols:
        raise ValueError(f"Missing required bar columns: {missing_cols}")

    out = bars.loc[:, REQUIRED_BAR_COLUMNS].copy()
    out = out.sort_values(["session", "bar_ix"], kind="mergesort").reset_index(drop=True)

    dupes = out.duplicated(["session", "bar_ix"]).sum()
    if dupes:
        raise ValueError(f"Found duplicate (session, bar_ix) rows: {int(dupes)}")

    return out


def _pivot_matrix(
    bars: pd.DataFrame,
    sessions: pd.Index,
    bar_index: pd.Index,
    value_col: str,
) -> np.ndarray:
    pivoted = bars.pivot(index="session", columns="bar_ix", values=value_col)
    pivoted = pivoted.reindex(index=sessions, columns=bar_index)
    return pivoted.to_numpy(dtype=float)


def _safe_log(arr: np.ndarray, eps: float) -> np.ndarray:
    return np.log(np.maximum(arr, eps))


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _matrix_to_frame(matrix: np.ndarray, prefix: str, start_ix: int, index: pd.Index) -> pd.DataFrame:
    matrix = _finite(matrix)
    width = matrix.shape[1]
    max_ix = start_ix + width - 1
    digits = max(2, len(str(max_ix)))
    cols = [f"{prefix}_{ix:0{digits}d}" for ix in range(start_ix, start_ix + width)]
    return pd.DataFrame(matrix, index=index, columns=cols)


def _window_summary_features(
    close_m: np.ndarray,
    ret_cc: np.ndarray,
    range_hl: np.ndarray,
    body_norm: np.ndarray,
    windows: tuple[int, ...],
    eps: float,
    index: pd.Index,
) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    seen_width = close_m.shape[1]

    for window in windows:
        if window > seen_width or window < 2:
            continue

        close_w = close_m[:, -window:]
        ret_w = ret_cc[:, -(window - 1) :]
        range_w = range_hl[:, -window:]
        body_w = body_norm[:, -window:]

        cumret_log = _safe_log(close_w[:, -1], eps) - _safe_log(close_w[:, 0], eps)
        mean_ret = _row_mean(ret_w)
        vol_ret = _row_std(ret_w)
        slope_log_close = _row_slope(_safe_log(close_w, eps), eps=eps)
        lastk_ret = _finite(close_w[:, -1] / np.maximum(close_w[:, 0], eps) - 1.0)
        max_drawdown = _max_drawdown(close_w, eps=eps)
        max_runup = _max_runup(close_w, eps=eps)
        frac_pos = _row_mean((ret_w > 0).astype(float))
        mean_range = _row_mean(range_w)
        mean_abs_body = _row_mean(np.abs(body_w))
        acf1 = _lag1_autocorr(ret_w, eps=eps)
        skew = _row_skew(ret_w, eps=eps)
        kurt = _row_kurtosis(ret_w, eps=eps)

        out[f"cumret_log_w{window}"] = _finite(cumret_log)
        out[f"mean_ret_w{window}"] = mean_ret
        out[f"vol_ret_w{window}"] = vol_ret
        out[f"slope_log_close_w{window}"] = slope_log_close
        out[f"lastk_ret_w{window}"] = lastk_ret
        out[f"max_drawdown_w{window}"] = max_drawdown
        out[f"max_runup_w{window}"] = max_runup
        out[f"frac_pos_ret_w{window}"] = frac_pos
        out[f"mean_range_hl_w{window}"] = mean_range
        out[f"mean_abs_body_w{window}"] = mean_abs_body
        out[f"acf1_ret_w{window}"] = acf1
        out[f"skew_ret_w{window}"] = skew
        out[f"kurt_ret_w{window}"] = kurt

    return out


def _row_mean(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    with np.errstate(all="ignore"):
        return _finite(np.nanmean(values, axis=1))


def _row_std(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    with np.errstate(all="ignore"):
        return _finite(np.nanstd(values, axis=1, ddof=0))


def _row_min(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return _finite(np.nanmin(values, axis=1))


def _row_max(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return _finite(np.nanmax(values, axis=1))


def _row_last(values: np.ndarray) -> np.ndarray:
    return _finite(values[:, -1])


def _row_slope(values: np.ndarray, eps: float) -> np.ndarray:
    """OLS slope of each row against index positions, ignoring NaNs."""
    n_rows, n_cols = values.shape
    x = np.arange(n_cols, dtype=float)
    mask = np.isfinite(values)
    y = np.where(mask, values, 0.0)

    count = mask.sum(axis=1).astype(float)
    x_sum = (mask * x).sum(axis=1)
    y_sum = y.sum(axis=1)
    xx_sum = (mask * (x**2)).sum(axis=1)
    xy_sum = (y * x).sum(axis=1)

    denom = count * xx_sum - x_sum**2
    numer = count * xy_sum - x_sum * y_sum

    slope = np.zeros(n_rows, dtype=float)
    valid = (count >= 2.0) & (np.abs(denom) > eps)
    slope[valid] = numer[valid] / denom[valid]
    return _finite(slope)


def _max_drawdown(close_w: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros(close_w.shape[0], dtype=float)
    for i, row in enumerate(close_w):
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        running_max = np.maximum.accumulate(finite)
        drawdowns = finite / np.maximum(running_max, eps) - 1.0
        out[i] = float(np.min(drawdowns))
    return _finite(out)


def _max_runup(close_w: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros(close_w.shape[0], dtype=float)
    for i, row in enumerate(close_w):
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        running_min = np.minimum.accumulate(finite)
        runups = finite / np.maximum(running_min, eps) - 1.0
        out[i] = float(np.max(runups))
    return _finite(out)


def _lag1_autocorr(values: np.ndarray, eps: float) -> np.ndarray:
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=float)

    x = values[:, :-1]
    y = values[:, 1:]
    mask = np.isfinite(x) & np.isfinite(y)

    n = mask.sum(axis=1).astype(float)
    xz = np.where(mask, x, 0.0)
    yz = np.where(mask, y, 0.0)

    mx = xz.sum(axis=1) / np.maximum(n, 1.0)
    my = yz.sum(axis=1) / np.maximum(n, 1.0)

    x_center = (xz - mx[:, None]) * mask
    y_center = (yz - my[:, None]) * mask

    cov = (x_center * y_center).sum(axis=1) / np.maximum(n, 1.0)
    var_x = (x_center**2).sum(axis=1) / np.maximum(n, 1.0)
    var_y = (y_center**2).sum(axis=1) / np.maximum(n, 1.0)

    denom = np.sqrt(var_x * var_y) + eps
    corr = cov / denom

    invalid = (n < 2.0) | (var_x <= eps) | (var_y <= eps)
    corr[invalid] = 0.0
    return _finite(corr)


def _row_skew(values: np.ndarray, eps: float) -> np.ndarray:
    mask = np.isfinite(values)
    n = mask.sum(axis=1).astype(float)
    vz = np.where(mask, values, 0.0)

    mean = vz.sum(axis=1) / np.maximum(n, 1.0)
    centered = (vz - mean[:, None]) * mask

    m2 = (centered**2).sum(axis=1) / np.maximum(n, 1.0)
    m3 = (centered**3).sum(axis=1) / np.maximum(n, 1.0)

    skew = m3 / np.power(m2 + eps, 1.5)
    invalid = (n < 3.0) | (m2 <= eps)
    skew[invalid] = 0.0
    return _finite(skew)


def _row_kurtosis(values: np.ndarray, eps: float) -> np.ndarray:
    mask = np.isfinite(values)
    n = mask.sum(axis=1).astype(float)
    vz = np.where(mask, values, 0.0)

    mean = vz.sum(axis=1) / np.maximum(n, 1.0)
    centered = (vz - mean[:, None]) * mask

    m2 = (centered**2).sum(axis=1) / np.maximum(n, 1.0)
    m4 = (centered**4).sum(axis=1) / np.maximum(n, 1.0)

    kurt = m4 / np.power(m2 + eps, 2.0) - 3.0
    invalid = (n < 4.0) | (m2 <= eps)
    kurt[invalid] = 0.0
    return _finite(kurt)


def _dct_type_ii(values: np.ndarray, n_coeff: int) -> np.ndarray:
    """Orthonormal DCT-II along axis=1 without scipy dependency."""
    n_rows, n_cols = values.shape
    coeff_count = max(1, min(int(n_coeff), n_cols))

    n = np.arange(n_cols, dtype=float)
    k = np.arange(coeff_count, dtype=float)[:, None]

    basis = np.cos(np.pi * (n + 0.5) * k / float(n_cols))
    basis[0, :] *= np.sqrt(1.0 / n_cols)
    if coeff_count > 1:
        basis[1:, :] *= np.sqrt(2.0 / n_cols)

    safe_values = _finite(values)
    return safe_values @ basis.T
