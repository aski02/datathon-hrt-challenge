from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pipeline.types import SplitInput

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_STORE_DIR = ROOT / "agents" / "features" / "feature_store"

AggregationMode = Literal["prob_avg", "rank_avg"]


@dataclass(frozen=True)
class MappingTemplate:
    name: str
    bucket_fracs: tuple[float, ...]
    positions: tuple[float, ...]


@dataclass(frozen=True)
class HeadSpec:
    subspace_name: str
    columns: tuple[str, ...]
    q_bad: float
    c_value: float
    row_subsample_ratio: float
    weight: float
    seed: int


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    n_heads: int
    aggregation: AggregationMode
    feature_subset_ratio: float
    heads: tuple[HeadSpec, ...]


@dataclass(frozen=True)
class SearchResult:
    candidate: CandidateConfig
    mapping_template: MappingTemplate
    mean_sharpe: float
    std_sharpe: float
    robust_score: float
    per_repeat_sharpes: tuple[float, ...]
    fold_sharpes: tuple[float, ...]
    unique_subspaces: tuple[str, ...]


@dataclass
class TrainedHead:
    spec: HeadSpec
    scaler: StandardScaler | None
    model: LogisticRegression | None
    constant_prob: float | None
    prob_center: float
    prob_scale: float


class SubspaceBaggedDownsideRankerStrategy:
    """Default stochastic subspace-bagged downside ranker."""

    name = "subspace-bagged-downside-ranker"

    def __init__(
        self,
        feature_store_dir: Path = DEFAULT_FEATURE_STORE_DIR,
        cv_folds: int = 5,
        cv_repeats: int = 3,
        random_state: int = 42,
        robust_std_penalty: float = 0.35,
        max_iter: int = 2500,
    ) -> None:
        self.feature_store_dir = feature_store_dir
        self.cv_folds = int(cv_folds)
        self.cv_repeats = int(cv_repeats)
        self.random_state = int(random_state)
        self.robust_std_penalty = float(robust_std_penalty)
        self.max_iter = int(max_iter)

        self.q_bad_values: tuple[float, ...] = (0.25, 0.30, 0.35)
        self.c_grid: tuple[float, ...] = (0.3, 1.0, 3.0)
        self.n_heads_grid: tuple[int, ...] = (12, 18, 24)
        self.row_subsample_grid: tuple[float, ...] = (1.0, 0.85, 0.70)
        self.aggregation_grid: tuple[AggregationMode, ...] = ("prob_avg", "rank_avg")

        self.feature_subset_ratio = 0.80
        self.min_features_per_head = 6

        self.q_weight_map: dict[float, float] = {
            0.25: 1.15,
            0.30: 1.00,
            0.35: 0.85,
        }

        self.mapping_templates: tuple[MappingTemplate, ...] = (
            MappingTemplate(
                name="template_a_champion_style_conservative",
                bucket_fracs=(0.10, 0.20, 0.40, 0.20, 0.10),
                positions=(2.25, 1.75, 1.50, 1.00, 0.25),
            ),
            MappingTemplate(
                name="template_b_sharper_downside_protection",
                bucket_fracs=(0.10, 0.20, 0.40, 0.20, 0.10),
                positions=(2.25, 1.75, 1.50, 0.75, 0.00),
            ),
            MappingTemplate(
                name="template_c_severity_aware_tail_cut",
                bucket_fracs=(0.10, 0.20, 0.35, 0.20, 0.15),
                positions=(2.25, 1.75, 1.50, 0.75, 0.00),
            ),
            MappingTemplate(
                name="template_d_drift_preserving_center",
                bucket_fracs=(0.10, 0.20, 0.40, 0.20, 0.10),
                positions=(2.00, 1.75, 1.50, 1.25, 0.50),
            ),
            MappingTemplate(
                name="template_e_soft_tail_guard",
                bucket_fracs=(0.10, 0.20, 0.40, 0.20, 0.10),
                positions=(2.20, 1.75, 1.50, 0.85, 0.00),
            ),
        )

        self._table_cache: dict[str, pd.DataFrame] = {}
        self._trained_heads: list[TrainedHead] = []
        self._trained_columns: set[str] = set()

        self.selected_result: SearchResult | None = None

    def fit(self, train_split: SplitInput, train_target_return: pd.Series) -> None:
        x_train = self._load_feature_table("X_train")
        x_train = self._align_rows(x_train, train_split.sessions, "X_train")
        x_train = self._to_numeric_features(x_train, "X_train")

        y_train = self._load_target_series()
        y_train = self._align_rows(y_train, train_split.sessions, "y_train").astype(float)

        runner_target = self._align_rows(train_target_return.astype(float), train_split.sessions, "train_target_return")
        max_abs_diff = float(np.max(np.abs(y_train.to_numpy(dtype=float) - runner_target.to_numpy(dtype=float))))
        if max_abs_diff > 1e-10:
            raise ValueError(
                f"Feature-store target mismatch against runner target returns (max_abs_diff={max_abs_diff:.3e})."
            )

        y_values = y_train.to_numpy(dtype=float)
        cv_folds = self._build_repeated_folds(y_values)

        subspace_library = self._build_subspace_library(x_train)
        candidates = self._build_subspace_candidates(subspace_library)

        all_results = [
            self._evaluate_candidate(
                x_train=x_train,
                y_values=y_values,
                cv_folds=cv_folds,
                candidate=candidate,
                allowed_templates=self.mapping_templates,
            )
            for candidate in candidates
        ]
        if not all_results:
            raise RuntimeError("No subspace-bagged candidates produced valid repeated-CV results.")

        self.selected_result = self._pick_best_result(all_results)

        self._trained_heads = self._fit_heads_full_data(
            x_train=x_train,
            y_values=y_values,
            candidate=self.selected_result.candidate,
        )
        self._trained_columns = {column for head in self._trained_heads for column in head.spec.columns}

        selected = self.selected_result
        print(
            "[subspace-bagged-downside-ranker] "
            f"selected_candidate={selected.candidate.name}, "
            f"n_heads={selected.candidate.n_heads}, "
            f"aggregation={selected.candidate.aggregation}, "
            f"mapping={selected.mapping_template.name}, "
            f"mean_cv_sharpe={selected.mean_sharpe:.4f}, "
            f"std_cv_sharpe={selected.std_sharpe:.4f}, "
            f"robust_score={selected.robust_score:.4f}, "
            f"subspaces={selected.unique_subspaces}"
        )
        print(
            "[subspace-bagged-downside-ranker] "
            f"selected_per_repeat_sharpes={','.join(f'{value:.4f}' for value in selected.per_repeat_sharpes)}"
        )

    def predict(self, split: SplitInput) -> pd.Series:
        if self.selected_result is None or not self._trained_heads:
            raise RuntimeError("Strategy is not fit. Call fit(train_split, train_target_return) before predict(split).")

        table_name = self._table_for_split(split.name)
        x_split = self._load_feature_table(table_name)
        x_split = self._align_rows(x_split, split.sessions, table_name)
        x_split = self._to_numeric_features(x_split, table_name)

        missing_columns = [column for column in self._trained_columns if column not in x_split.columns]
        if missing_columns:
            preview = ", ".join(missing_columns[:8])
            raise ValueError(
                f"{table_name} is missing {len(missing_columns)} trained feature columns. First missing: {preview}"
            )

        downside_score = self._predict_downside_score(
            x_values=x_split,
            heads=self._trained_heads,
            aggregation=self.selected_result.candidate.aggregation,
        )
        positions = self._map_downside_to_rank_buckets(downside_score, self.selected_result.mapping_template)
        return pd.Series(positions, index=split.sessions, name="target_position")

    def _table_for_split(self, split_name: str) -> str:
        mapping = {
            "train_seen": "X_train",
            "public_seen": "X_public_test",
            "private_seen": "X_private_test",
        }
        if split_name not in mapping:
            raise ValueError(f"Unsupported split name: {split_name!r}")
        return mapping[split_name]

    def _load_feature_table(self, table_name: str) -> pd.DataFrame:
        if table_name in self._table_cache:
            return self._table_cache[table_name]

        path = self.feature_store_dir / f"{table_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature-store file not found: {path}")

        frame = pd.read_parquet(path)
        if "session" in frame.columns:
            if frame["session"].duplicated().any():
                duplicates = int(frame["session"].duplicated().sum())
                raise ValueError(f"{table_name} has duplicated `session` column values (duplicates={duplicates}).")
            frame = frame.set_index("session", drop=True)

        if frame.index.name != "session":
            raise ValueError(f"{table_name} must be indexed by `session` or contain a `session` column.")
        if frame.index.duplicated().any():
            duplicates = int(frame.index.duplicated().sum())
            raise ValueError(f"{table_name} has duplicated session index values (duplicates={duplicates}).")

        frame = frame.sort_index()
        self._table_cache[table_name] = frame
        return frame

    def _load_target_series(self) -> pd.Series:
        frame = self._load_feature_table("y_train")
        if "target_return" in frame.columns:
            series = frame["target_return"]
        elif len(frame.columns) == 1:
            series = frame.iloc[:, 0].rename("target_return")
        else:
            raise ValueError("y_train must have exactly one target column (expected `target_return`).")
        return series.astype(float)

    def _align_rows(self, frame: pd.DataFrame | pd.Series, sessions: pd.Index, label: str) -> pd.DataFrame | pd.Series:
        if sessions.duplicated().any():
            duplicates = int(sessions.duplicated().sum())
            raise ValueError(f"Requested sessions contain duplicates for {label} (duplicates={duplicates}).")

        missing = sessions.difference(frame.index)
        if len(missing) > 0:
            preview = ", ".join(str(value) for value in missing[:8].tolist())
            raise ValueError(f"{label} is missing {len(missing)} requested sessions. First missing: {preview}")
        return frame.reindex(sessions)

    def _to_numeric_features(self, frame: pd.DataFrame, label: str) -> pd.DataFrame:
        numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {"session", "bar_ix"}
        feature_columns = [column for column in numeric_columns if column not in excluded]
        if not feature_columns:
            raise ValueError(f"{label} contains no numeric feature columns after exclusions.")

        numeric_frame = frame.loc[:, feature_columns]
        cleaned = numeric_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if cleaned.isna().to_numpy().any():
            raise ValueError(f"{label} still contains NaN values after cleaning.")
        return cleaned.astype(float)

    def _build_repeated_folds(self, y_values: np.ndarray) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        y_strat = (y_values <= np.quantile(y_values, 0.30)).astype(int)
        repeated: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for repeat_idx in range(self.cv_repeats):
            repeat_seed = self.random_state + 9973 * (repeat_idx + 1)
            splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=repeat_seed)
            repeated.append(list(splitter.split(np.arange(len(y_values)), y_strat)))
        return repeated

    def _build_subspace_library(self, x_train: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        columns = x_train.columns.tolist()

        summary_lite_names = [
            "lastk_ret_w5",
            "lastk_ret_w10",
            "slope_log_close_w10",
            "vol_ret_w10",
            "vol_ret_w20",
            "vol_ret_w50",
            "max_drawdown_w20",
            "max_drawdown_w50",
            "max_runup_w20",
            "max_runup_w50",
            "frac_pos_ret_w20",
            "mean_range_hl_w20",
            "mean_abs_body_w20",
            "acf1_ret_w20",
            "skew_ret_w20",
            "kurt_ret_w20",
            "cumret_log_w50",
            "mean_ret_w20",
        ]
        summary_lite = [name for name in summary_lite_names if name in columns]

        global_names = [name for name in ("last_close_zscore", "last_close_pos_in_seen_range") if name in columns]

        recent_trend_names = [
            "lastk_ret_w5",
            "lastk_ret_w10",
            "cumret_log_w5",
            "cumret_log_w10",
            "slope_log_close_w5",
            "slope_log_close_w10",
            "vol_ret_w5",
            "vol_ret_w10",
            "mean_range_hl_w5",
            "mean_range_hl_w10",
            "mean_abs_body_w5",
            "mean_abs_body_w10",
            "acf1_ret_w10",
        ]
        recent_trend = [name for name in recent_trend_names if name in columns]

        dct_columns = self._sorted_suffix_columns(columns, "dct_close_")
        ret_cc_columns = self._sorted_suffix_columns(columns, "ret_cc_")
        close_norm_columns = self._sorted_suffix_columns(columns, "close_norm_")
        path_compression = dct_columns + ret_cc_columns[-8:] + close_norm_columns[-8:]

        drawdown_state_names = [
            "max_drawdown_w10",
            "max_drawdown_w20",
            "max_drawdown_w50",
            "max_runup_w10",
            "max_runup_w20",
            "max_runup_w50",
            "frac_pos_ret_w10",
            "frac_pos_ret_w20",
            "frac_pos_ret_w50",
            "mean_range_hl_w50",
            "mean_abs_body_w50",
            "vol_ret_w50",
            "last_close_zscore",
            "last_close_pos_in_seen_range",
        ]
        drawdown_state = [name for name in drawdown_state_names if name in columns]

        library: dict[str, tuple[str, ...]] = {
            "summary_lite": tuple(self._dedupe_preserve_order(summary_lite)),
            "summary_lite_plus_global": tuple(self._dedupe_preserve_order(summary_lite + global_names)),
            "recent_trend": tuple(self._dedupe_preserve_order(recent_trend)),
            "path_compression": tuple(self._dedupe_preserve_order(path_compression)),
            "drawdown_state": tuple(self._dedupe_preserve_order(drawdown_state)),
        }

        for name, values in library.items():
            if len(values) < self.min_features_per_head:
                raise ValueError(
                    f"Subspace `{name}` has only {len(values)} columns; expected at least {self.min_features_per_head}."
                )

        return library

    def _build_subspace_candidates(self, subspace_library: dict[str, tuple[str, ...]]) -> list[CandidateConfig]:
        candidates: list[CandidateConfig] = []

        for n_heads in self.n_heads_grid:
            for row_subsample in self.row_subsample_grid:
                for aggregation in self.aggregation_grid:
                    seed = (
                        self.random_state
                        + 701 * n_heads
                        + int(round(row_subsample * 100.0)) * 19
                        + (0 if aggregation == "prob_avg" else 1) * 1237
                    )
                    rng = np.random.default_rng(seed)
                    heads: list[HeadSpec] = []
                    subspace_names = sorted(subspace_library.keys())

                    for head_idx in range(n_heads):
                        subspace_name = str(rng.choice(subspace_names))
                        all_columns = list(subspace_library[subspace_name])
                        n_base = len(all_columns)
                        jitter = float(rng.uniform(0.75, 1.00))
                        feature_ratio = min(1.0, self.feature_subset_ratio * jitter)
                        n_pick = max(self.min_features_per_head, int(round(n_base * feature_ratio)))
                        n_pick = min(n_pick, n_base)
                        picked = tuple(sorted(rng.choice(all_columns, size=n_pick, replace=False).tolist()))

                        q_bad = float(rng.choice(self.q_bad_values))
                        c_value = float(rng.choice(self.c_grid))
                        heads.append(
                            HeadSpec(
                                subspace_name=subspace_name,
                                columns=picked,
                                q_bad=q_bad,
                                c_value=c_value,
                                row_subsample_ratio=float(row_subsample),
                                weight=float(self.q_weight_map[q_bad]),
                                seed=seed + 10007 * (head_idx + 1),
                            )
                        )

                    candidate_name = f"subspace_h{n_heads}_r{int(round(row_subsample*100)):02d}_{aggregation}"
                    candidates.append(
                        CandidateConfig(
                            name=candidate_name,
                            n_heads=n_heads,
                            aggregation=aggregation,
                            feature_subset_ratio=self.feature_subset_ratio,
                            heads=tuple(heads),
                        )
                    )

        return candidates

    def _evaluate_candidate(
        self,
        x_train: pd.DataFrame,
        y_values: np.ndarray,
        cv_folds: list[list[tuple[np.ndarray, np.ndarray]]],
        candidate: CandidateConfig,
        allowed_templates: tuple[MappingTemplate, ...],
    ) -> SearchResult:
        template_repeat_sharpes: dict[str, list[float]] = {template.name: [] for template in allowed_templates}
        template_fold_sharpes: dict[str, list[float]] = {template.name: [] for template in allowed_templates}

        for repeat_idx, folds in enumerate(cv_folds):
            oof_score = np.zeros(len(y_values), dtype=float)

            for fold_idx, (train_index, valid_index) in enumerate(folds):
                fold_score = self._fit_heads_on_fold_and_predict(
                    x_train=x_train,
                    y_values=y_values,
                    train_index=train_index,
                    valid_index=valid_index,
                    candidate=candidate,
                    repeat_idx=repeat_idx,
                    fold_idx=fold_idx,
                )
                oof_score[valid_index] = fold_score

            for template in allowed_templates:
                positions = self._map_downside_to_rank_buckets(oof_score, template)
                repeat_sharpe = self._sharpe_from_positions(positions, y_values)
                template_repeat_sharpes[template.name].append(float(repeat_sharpe))

                for _, valid_index in folds:
                    fold_sharpe = self._sharpe_from_positions(positions[valid_index], y_values[valid_index])
                    template_fold_sharpes[template.name].append(float(fold_sharpe))

        results: list[SearchResult] = []
        unique_subspaces = tuple(sorted({head.subspace_name for head in candidate.heads}))

        for template in allowed_templates:
            per_repeat = tuple(template_repeat_sharpes[template.name])
            fold_sharpes = tuple(template_fold_sharpes[template.name])
            mean_sharpe = float(np.mean(np.asarray(per_repeat, dtype=float)))
            std_sharpe = float(np.std(np.asarray(per_repeat, dtype=float), ddof=0))
            robust_score = float(mean_sharpe - self.robust_std_penalty * std_sharpe)
            results.append(
                SearchResult(
                    candidate=candidate,
                    mapping_template=template,
                    mean_sharpe=mean_sharpe,
                    std_sharpe=std_sharpe,
                    robust_score=robust_score,
                    per_repeat_sharpes=per_repeat,
                    fold_sharpes=fold_sharpes,
                    unique_subspaces=unique_subspaces,
                )
            )

        return self._pick_best_result(results)

    def _pick_best_result(self, results: list[SearchResult]) -> SearchResult:
        if not results:
            raise ValueError("Expected at least one result to compare.")

        best = results[0]
        for candidate in results[1:]:
            if self._is_better_result(candidate, best):
                best = candidate
        return best

    def _is_better_result(self, candidate: SearchResult, incumbent: SearchResult) -> bool:
        if candidate.robust_score > incumbent.robust_score + 1e-9:
            return True
        if not np.isclose(candidate.robust_score, incumbent.robust_score, atol=1e-9):
            return False

        if candidate.mean_sharpe > incumbent.mean_sharpe + 1e-9:
            return True
        if not np.isclose(candidate.mean_sharpe, incumbent.mean_sharpe, atol=1e-9):
            return False

        if candidate.std_sharpe < incumbent.std_sharpe - 1e-9:
            return True
        if not np.isclose(candidate.std_sharpe, incumbent.std_sharpe, atol=1e-9):
            return False

        if candidate.candidate.n_heads < incumbent.candidate.n_heads:
            return True
        if candidate.candidate.n_heads != incumbent.candidate.n_heads:
            return False

        if len(candidate.unique_subspaces) < len(incumbent.unique_subspaces):
            return True
        if len(candidate.unique_subspaces) != len(incumbent.unique_subspaces):
            return False

        cand_min_ratio = min(head.row_subsample_ratio for head in candidate.candidate.heads)
        inc_min_ratio = min(head.row_subsample_ratio for head in incumbent.candidate.heads)
        if cand_min_ratio > inc_min_ratio + 1e-12:
            return True
        if not np.isclose(cand_min_ratio, inc_min_ratio, atol=1e-12):
            return False

        cand_c_span = max(head.c_value for head in candidate.candidate.heads) - min(
            head.c_value for head in candidate.candidate.heads
        )
        inc_c_span = max(head.c_value for head in incumbent.candidate.heads) - min(
            head.c_value for head in incumbent.candidate.heads
        )
        if cand_c_span < inc_c_span - 1e-12:
            return True
        if not np.isclose(cand_c_span, inc_c_span, atol=1e-12):
            return False

        return self._template_complexity(candidate.mapping_template) < self._template_complexity(
            incumbent.mapping_template
        )

    @staticmethod
    def _template_complexity(template: MappingTemplate) -> float:
        spread = float(np.max(template.positions) - np.min(template.positions))
        hard_tail = 1.0 if float(np.min(template.positions)) <= 0.01 else 0.0
        return spread + 0.05 * hard_tail

    def _fit_heads_on_fold_and_predict(
        self,
        x_train: pd.DataFrame,
        y_values: np.ndarray,
        train_index: np.ndarray,
        valid_index: np.ndarray,
        candidate: CandidateConfig,
        repeat_idx: int,
        fold_idx: int,
    ) -> np.ndarray:
        valid_head_outputs: list[np.ndarray] = []
        head_weights: list[float] = []

        for head in candidate.heads:
            x_train_head = x_train.iloc[train_index].loc[:, list(head.columns)].to_numpy(dtype=float)
            y_train_head = y_values[train_index]
            x_valid_head = x_train.iloc[valid_index].loc[:, list(head.columns)].to_numpy(dtype=float)

            train_prob, valid_prob = self._fit_single_head_and_predict(
                x_train=x_train_head,
                y_train=y_train_head,
                x_eval=x_valid_head,
                head=head,
                seed=head.seed + 4099 * (repeat_idx + 1) + 97 * (fold_idx + 1),
            )

            if candidate.aggregation == "prob_avg":
                mean_prob = float(np.mean(train_prob))
                std_prob = float(np.std(train_prob, ddof=0))
                scale = max(std_prob, 1e-8)
                head_output = (valid_prob - mean_prob) / scale
            elif candidate.aggregation == "rank_avg":
                head_output = self._fractional_rank(valid_prob)
            else:
                raise ValueError(f"Unsupported aggregation mode: {candidate.aggregation}")

            valid_head_outputs.append(head_output)
            head_weights.append(float(head.weight))

        outputs = np.vstack(valid_head_outputs)
        weights = np.asarray(head_weights, dtype=float)
        if np.sum(weights) <= 0.0:
            raise ValueError("Head weights must sum to a positive value.")
        weights = weights / np.sum(weights)
        return np.average(outputs, axis=0, weights=weights)

    def _fit_single_head_and_predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        head: HeadSpec,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_train = x_train.shape[0]

        if not (0.0 < head.row_subsample_ratio <= 1.0):
            raise ValueError(f"Invalid row_subsample_ratio={head.row_subsample_ratio}. Expected (0, 1].")

        if head.row_subsample_ratio < 1.0:
            n_sub = max(64, int(round(n_train * head.row_subsample_ratio)))
            n_sub = min(n_sub, n_train)
            sub_index = rng.choice(n_train, size=n_sub, replace=False)
        else:
            sub_index = np.arange(n_train)

        x_fit = x_train[sub_index]
        y_fit = y_train[sub_index]

        threshold = float(np.quantile(y_fit, head.q_bad))
        y_binary = (y_fit <= threshold).astype(int)

        if np.unique(y_binary).size < 2:
            constant_prob = float(y_binary[0])
            train_prob = np.full(n_train, constant_prob, dtype=float)
            eval_prob = np.full(x_eval.shape[0], constant_prob, dtype=float)
            return train_prob, eval_prob

        scaler = StandardScaler()
        x_fit_scaled = scaler.fit_transform(x_fit)
        x_train_scaled = scaler.transform(x_train)
        x_eval_scaled = scaler.transform(x_eval)

        model = LogisticRegression(
            C=head.c_value,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=seed,
        )
        model.fit(x_fit_scaled, y_binary)

        train_prob = model.predict_proba(x_train_scaled)[:, 1]
        eval_prob = model.predict_proba(x_eval_scaled)[:, 1]
        return train_prob, eval_prob

    def _fit_heads_full_data(
        self, x_train: pd.DataFrame, y_values: np.ndarray, candidate: CandidateConfig
    ) -> list[TrainedHead]:
        trained: list[TrainedHead] = []

        for head in candidate.heads:
            x_values = x_train.loc[:, list(head.columns)].to_numpy(dtype=float)
            n_rows = x_values.shape[0]
            rng = np.random.default_rng(head.seed + 88829)

            if head.row_subsample_ratio < 1.0:
                n_sub = max(64, int(round(n_rows * head.row_subsample_ratio)))
                n_sub = min(n_sub, n_rows)
                sub_index = rng.choice(n_rows, size=n_sub, replace=False)
            else:
                sub_index = np.arange(n_rows)

            x_fit = x_values[sub_index]
            y_fit = y_values[sub_index]

            threshold = float(np.quantile(y_fit, head.q_bad))
            y_binary = (y_fit <= threshold).astype(int)

            if np.unique(y_binary).size < 2:
                constant_prob = float(y_binary[0])
                trained.append(
                    TrainedHead(
                        spec=head,
                        scaler=None,
                        model=None,
                        constant_prob=constant_prob,
                        prob_center=constant_prob,
                        prob_scale=1.0,
                    )
                )
                continue

            scaler = StandardScaler()
            x_fit_scaled = scaler.fit_transform(x_fit)
            x_train_scaled = scaler.transform(x_values)

            model = LogisticRegression(
                C=head.c_value,
                solver="lbfgs",
                max_iter=self.max_iter,
                random_state=head.seed + 91217,
            )
            model.fit(x_fit_scaled, y_binary)

            train_prob = model.predict_proba(x_train_scaled)[:, 1]
            center = float(np.mean(train_prob))
            scale = float(np.std(train_prob, ddof=0))
            trained.append(
                TrainedHead(
                    spec=head,
                    scaler=scaler,
                    model=model,
                    constant_prob=None,
                    prob_center=center,
                    prob_scale=max(scale, 1e-8),
                )
            )

        return trained

    def _predict_downside_score(
        self,
        x_values: pd.DataFrame,
        heads: list[TrainedHead],
        aggregation: AggregationMode,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        weights: list[float] = []

        for head in heads:
            head_x = x_values.loc[:, list(head.spec.columns)].to_numpy(dtype=float)
            if head.constant_prob is not None or head.model is None or head.scaler is None:
                prob = np.full(head_x.shape[0], float(head.constant_prob), dtype=float)
            else:
                head_scaled = head.scaler.transform(head_x)
                prob = head.model.predict_proba(head_scaled)[:, 1]

            if aggregation == "prob_avg":
                output = (prob - head.prob_center) / max(head.prob_scale, 1e-8)
            elif aggregation == "rank_avg":
                output = self._fractional_rank(prob)
            else:
                raise ValueError(f"Unsupported aggregation mode: {aggregation}")

            outputs.append(output)
            weights.append(float(head.spec.weight))

        matrix = np.vstack(outputs)
        weight_arr = np.asarray(weights, dtype=float)
        weight_arr = weight_arr / np.sum(weight_arr)
        return np.average(matrix, axis=0, weights=weight_arr)

    @staticmethod
    def _fractional_rank(values: np.ndarray) -> np.ndarray:
        n = len(values)
        if n == 0:
            return np.asarray([], dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = (np.arange(n, dtype=float) + 0.5) / float(n)
        return ranks

    @staticmethod
    def _map_downside_to_rank_buckets(downside_score: np.ndarray, template: MappingTemplate) -> np.ndarray:
        if len(template.bucket_fracs) != len(template.positions):
            raise ValueError(f"Invalid template `{template.name}`: fraction/position length mismatch.")

        total = float(np.sum(np.asarray(template.bucket_fracs, dtype=float)))
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"Invalid template `{template.name}`: fractions must sum to 1.0, got {total}.")

        n = len(downside_score)
        if n == 0:
            return np.asarray([], dtype=float)

        order = np.argsort(downside_score, kind="mergesort")
        pct = np.empty(n, dtype=float)
        pct[order] = (np.arange(n, dtype=float) + 0.5) / float(n)

        edges = np.cumsum(np.asarray(template.bucket_fracs, dtype=float))
        edges[-1] = 1.0

        positions = np.full(n, float(template.positions[-1]), dtype=float)
        lower = 0.0
        for edge, position in zip(edges, template.positions):
            mask = (pct > lower) & (pct <= edge + 1e-12)
            positions[mask] = float(position)
            lower = float(edge)

        return np.clip(positions, 0.0, float(np.max(template.positions)))

    @staticmethod
    def _sharpe_from_positions(positions: np.ndarray, returns: np.ndarray) -> float:
        pnl = positions * returns
        pnl_std = pnl.std(ddof=0)
        if pnl_std == 0.0:
            return 0.0
        return float(pnl.mean() / pnl_std * 16.0)

    @staticmethod
    def _sorted_suffix_columns(columns: list[str], prefix: str) -> list[str]:
        matched = [column for column in columns if column.startswith(prefix)]

        def key_fn(name: str) -> int:
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
            return -1

        return sorted(matched, key=key_fn)

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped


def build_strategy() -> SubspaceBaggedDownsideRankerStrategy:
    return SubspaceBaggedDownsideRankerStrategy()
