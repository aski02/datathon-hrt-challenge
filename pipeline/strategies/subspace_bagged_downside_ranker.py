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

ReliabilityMode = Literal[
    "equal_rank_avg",
    "robust_weighted_rank_avg",
    "robust_weighted_rank_avg_shrunk",
]


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
    seed: int


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    heads: tuple[HeadSpec, ...]


@dataclass(frozen=True)
class ConsensusConfig:
    name: str
    bottom10_threshold: float | None
    bottom20_threshold: float | None
    top10_threshold: float | None
    top_low_down20_max: float
    disagreement_std_threshold: float | None

    @property
    def override_count(self) -> int:
        count = 0
        if self.bottom10_threshold is not None:
            count += 1
        if self.bottom20_threshold is not None:
            count += 1
        if self.top10_threshold is not None:
            count += 1
        if self.disagreement_std_threshold is not None:
            count += 1
        return count


@dataclass(frozen=True)
class ModelConfig:
    name: str
    head_indices: tuple[int, ...]
    reliability_mode: ReliabilityMode
    consensus: ConsensusConfig


@dataclass(frozen=True)
class SearchResult:
    tag: str
    config: ModelConfig
    mean_sharpe: float
    std_sharpe: float
    robust_score: float
    per_repeat_sharpes: tuple[float, ...]
    fold_sharpes: tuple[float, ...]


@dataclass(frozen=True)
class HeadDiagnostic:
    head_ix: int
    q_bad: float
    subspace_name: str
    mean_sharpe: float
    std_sharpe: float
    robust_score: float
    corr_with_equal_ensemble: float
    per_repeat_sharpes: tuple[float, ...]


@dataclass
class TrainedHead:
    spec: HeadSpec
    scaler: StandardScaler | None
    model: LogisticRegression | None
    constant_prob: float | None


@dataclass(frozen=True)
class RepeatArtifacts:
    oof_head_rank: np.ndarray  # shape: [n_heads, n_sessions]


class SubspaceBaggedDownsideRankerStrategy:
    """Consensus-aware follow-up to the 12-head subspace downside ranker."""

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
        self.feature_subset_ratio = 0.80
        self.min_features_per_head = 6

        self.base_mapping = MappingTemplate(
            name="template_a_champion_style_conservative",
            bucket_fracs=(0.10, 0.20, 0.40, 0.20, 0.10),
            positions=(2.25, 1.75, 1.50, 1.00, 0.25),
        )

        self._table_cache: dict[str, pd.DataFrame] = {}

        self.anchor_candidate: CandidateConfig | None = None
        self.head_diagnostics: tuple[HeadDiagnostic, ...] = ()

        self.baseline_result: SearchResult | None = None
        self.reliability_only_result: SearchResult | None = None
        self.consensus_only_result: SearchResult | None = None
        self.combined_result: SearchResult | None = None
        self.pruned_result: SearchResult | None = None
        self.selected_result: SearchResult | None = None

        self._selected_head_indices: tuple[int, ...] = ()
        self._selected_weights: np.ndarray | None = None
        self._selected_consensus: ConsensusConfig | None = None

        self._trained_heads: list[TrainedHead] = []
        self._trained_columns: set[str] = set()

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
        self.anchor_candidate = self._build_anchor_candidate(subspace_library)
        n_heads = len(self.anchor_candidate.heads)

        cv_artifacts = self._build_cv_artifacts(
            x_train=x_train,
            y_values=y_values,
            candidate=self.anchor_candidate,
            cv_folds=cv_folds,
        )

        self.head_diagnostics = self._build_head_diagnostics(
            candidate=self.anchor_candidate,
            cv_artifacts=cv_artifacts,
            y_values=y_values,
        )

        full_indices = tuple(range(n_heads))
        consensus_grid = self._build_consensus_grid()
        consensus_off = consensus_grid[0]

        baseline_config = ModelConfig(
            name="baseline_equal_rank_avg_template_a",
            head_indices=full_indices,
            reliability_mode="equal_rank_avg",
            consensus=consensus_off,
        )
        self.baseline_result = self._evaluate_config(
            tag="baseline",
            config=baseline_config,
            cv_artifacts=cv_artifacts,
            cv_folds=cv_folds,
            y_values=y_values,
        )

        reliability_only_candidates = [
            ModelConfig(
                name=f"reliability_only_{mode}",
                head_indices=full_indices,
                reliability_mode=mode,
                consensus=consensus_off,
            )
            for mode in ("robust_weighted_rank_avg", "robust_weighted_rank_avg_shrunk")
        ]
        self.reliability_only_result = self._pick_best_result(
            [
                self._evaluate_config(
                    tag="reliability_only",
                    config=config,
                    cv_artifacts=cv_artifacts,
                    cv_folds=cv_folds,
                    y_values=y_values,
                )
                for config in reliability_only_candidates
            ]
        )

        consensus_only_candidates = [
            ModelConfig(
                name=f"consensus_only_{consensus.name}",
                head_indices=full_indices,
                reliability_mode="equal_rank_avg",
                consensus=consensus,
            )
            for consensus in consensus_grid[1:]
        ]
        self.consensus_only_result = self._pick_best_result(
            [
                self._evaluate_config(
                    tag="consensus_only",
                    config=config,
                    cv_artifacts=cv_artifacts,
                    cv_folds=cv_folds,
                    y_values=y_values,
                )
                for config in consensus_only_candidates
            ]
        )

        combined_candidates: list[ModelConfig] = []
        for mode in ("robust_weighted_rank_avg", "robust_weighted_rank_avg_shrunk"):
            for consensus in consensus_grid[1:]:
                combined_candidates.append(
                    ModelConfig(
                        name=f"combined_{mode}_{consensus.name}",
                        head_indices=full_indices,
                        reliability_mode=mode,
                        consensus=consensus,
                    )
                )

        self.combined_result = self._pick_best_result(
            [
                self._evaluate_config(
                    tag="combined",
                    config=config,
                    cv_artifacts=cv_artifacts,
                    cv_folds=cv_folds,
                    y_values=y_values,
                )
                for config in combined_candidates
            ]
        )

        pruned_indices = self._build_pruned_indices(target_count=10)
        self.pruned_result = None
        if pruned_indices is not None:
            pruned_candidates: list[ModelConfig] = []
            for mode in ("robust_weighted_rank_avg", "robust_weighted_rank_avg_shrunk"):
                for consensus in consensus_grid[1:]:
                    pruned_candidates.append(
                        ModelConfig(
                            name=f"pruned_{mode}_{consensus.name}",
                            head_indices=pruned_indices,
                            reliability_mode=mode,
                            consensus=consensus,
                        )
                    )

            self.pruned_result = self._pick_best_result(
                [
                    self._evaluate_config(
                        tag="pruned",
                        config=config,
                        cv_artifacts=cv_artifacts,
                        cv_folds=cv_folds,
                        y_values=y_values,
                    )
                    for config in pruned_candidates
                ]
            )

        candidate_results = [
            self.baseline_result,
            self.reliability_only_result,
            self.consensus_only_result,
            self.combined_result,
        ]
        if self.pruned_result is not None:
            candidate_results.append(self.pruned_result)

        self.selected_result = self._pick_best_result(candidate_results)
        selected = self.selected_result

        self._selected_head_indices = selected.config.head_indices
        self._selected_weights = self._build_head_weights(
            mode=selected.config.reliability_mode,
            head_indices=self._selected_head_indices,
        )
        self._selected_consensus = selected.config.consensus

        selected_heads = tuple(self.anchor_candidate.heads[idx] for idx in self._selected_head_indices)
        self._trained_heads = self._fit_heads_full_data(
            x_train=x_train,
            y_values=y_values,
            head_specs=selected_heads,
        )
        self._trained_columns = {column for head in self._trained_heads for column in head.spec.columns}

        pruned_robust = float("nan") if self.pruned_result is None else float(self.pruned_result.robust_score)

        print(
            "[subspace-bagged-downside-ranker] "
            f"baseline_robust={self.baseline_result.robust_score:.4f}, "
            f"reliability_only_robust={self.reliability_only_result.robust_score:.4f}, "
            f"consensus_only_robust={self.consensus_only_result.robust_score:.4f}, "
            f"combined_robust={self.combined_result.robust_score:.4f}, "
            f"pruned_robust={pruned_robust:.4f}"
        )

        print(
            "[subspace-bagged-downside-ranker] "
            f"selected_tag={selected.tag}, config={selected.config.name}, "
            f"heads={len(selected.config.head_indices)}, "
            f"reliability={selected.config.reliability_mode}, "
            f"consensus={selected.config.consensus.name}, "
            f"mean_cv_sharpe={selected.mean_sharpe:.4f}, "
            f"std_cv_sharpe={selected.std_sharpe:.4f}, "
            f"robust_score={selected.robust_score:.4f}"
        )
        print(
            "[subspace-bagged-downside-ranker] "
            f"selected_per_repeat_sharpes={','.join(f'{value:.4f}' for value in selected.per_repeat_sharpes)}"
        )

    def predict(self, split: SplitInput) -> pd.Series:
        if (
            self.selected_result is None
            or self._selected_weights is None
            or self._selected_consensus is None
            or not self._trained_heads
        ):
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

        head_ranks = self._predict_head_rank_matrix(x_split, self._trained_heads)
        positions = self._positions_from_head_rank_matrix(
            head_ranks=head_ranks,
            weights=self._selected_weights,
            consensus=self._selected_consensus,
            template=self.base_mapping,
        )
        return pd.Series(positions, index=split.sessions, name="target_position")

    def _build_repeated_folds(self, y_values: np.ndarray) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        y_strat = (y_values <= np.quantile(y_values, 0.30)).astype(int)
        repeated: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for repeat_idx in range(self.cv_repeats):
            repeat_seed = self.random_state + 9973 * (repeat_idx + 1)
            splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=repeat_seed)
            repeated.append(list(splitter.split(np.arange(len(y_values)), y_strat)))
        return repeated

    def _build_cv_artifacts(
        self,
        x_train: pd.DataFrame,
        y_values: np.ndarray,
        candidate: CandidateConfig,
        cv_folds: list[list[tuple[np.ndarray, np.ndarray]]],
    ) -> tuple[RepeatArtifacts, ...]:
        n_samples = len(y_values)
        n_heads = len(candidate.heads)
        outputs: list[RepeatArtifacts] = []

        for repeat_idx, folds in enumerate(cv_folds):
            oof_rank = np.zeros((n_heads, n_samples), dtype=float)

            for fold_idx, (train_index, valid_index) in enumerate(folds):
                for head_ix, head in enumerate(candidate.heads):
                    x_fold_train = x_train.iloc[train_index].loc[:, list(head.columns)].to_numpy(dtype=float)
                    y_fold_train = y_values[train_index]
                    x_fold_valid = x_train.iloc[valid_index].loc[:, list(head.columns)].to_numpy(dtype=float)

                    _, valid_prob = self._fit_single_head_and_predict(
                        x_train=x_fold_train,
                        y_train=y_fold_train,
                        x_eval=x_fold_valid,
                        head=head,
                        seed=head.seed + 4099 * (repeat_idx + 1) + 97 * (fold_idx + 1),
                    )
                    oof_rank[head_ix, valid_index] = self._fractional_rank(valid_prob)

            outputs.append(RepeatArtifacts(oof_head_rank=oof_rank))

        return tuple(outputs)

    def _build_head_diagnostics(
        self,
        candidate: CandidateConfig,
        cv_artifacts: tuple[RepeatArtifacts, ...],
        y_values: np.ndarray,
    ) -> tuple[HeadDiagnostic, ...]:
        n_heads = len(candidate.heads)
        equal_weights = np.full(n_heads, 1.0 / float(n_heads), dtype=float)

        equal_ensemble_repeat = [
            np.average(artifact.oof_head_rank, axis=0, weights=equal_weights)
            for artifact in cv_artifacts
        ]

        diagnostics: list[HeadDiagnostic] = []
        for head_ix, head in enumerate(candidate.heads):
            per_repeat: list[float] = []
            for artifact in cv_artifacts:
                head_rank = artifact.oof_head_rank[head_ix]
                positions = self._positions_from_head_rank_matrix(
                    head_ranks=head_rank[None, :],
                    weights=np.asarray([1.0], dtype=float),
                    consensus=self._consensus_off(),
                    template=self.base_mapping,
                )
                sharpe = self._sharpe_from_positions(positions, y_values)
                per_repeat.append(float(sharpe))

            mean_sharpe = float(np.mean(np.asarray(per_repeat, dtype=float)))
            std_sharpe = float(np.std(np.asarray(per_repeat, dtype=float), ddof=0))
            robust_score = float(mean_sharpe - self.robust_std_penalty * std_sharpe)

            head_concat = np.concatenate([artifact.oof_head_rank[head_ix] for artifact in cv_artifacts], axis=0)
            ens_concat = np.concatenate(equal_ensemble_repeat, axis=0)
            corr = self._safe_corr(head_concat, ens_concat)

            diagnostics.append(
                HeadDiagnostic(
                    head_ix=head_ix,
                    q_bad=float(head.q_bad),
                    subspace_name=head.subspace_name,
                    mean_sharpe=mean_sharpe,
                    std_sharpe=std_sharpe,
                    robust_score=robust_score,
                    corr_with_equal_ensemble=corr,
                    per_repeat_sharpes=tuple(per_repeat),
                )
            )

        return tuple(diagnostics)

    def _build_head_weights(self, mode: ReliabilityMode, head_indices: tuple[int, ...]) -> np.ndarray:
        n = len(head_indices)
        if n == 0:
            raise ValueError("Head index set is empty.")

        equal = np.full(n, 1.0 / float(n), dtype=float)
        if mode == "equal_rank_avg":
            return equal

        robust_raw = np.asarray(
            [max(self.head_diagnostics[idx].robust_score, 0.0) + 1e-6 for idx in head_indices],
            dtype=float,
        )
        corr_raw = np.asarray(
            [abs(self.head_diagnostics[idx].corr_with_equal_ensemble) for idx in head_indices],
            dtype=float,
        )

        penalty = np.ones_like(corr_raw)
        high = corr_raw > 0.92
        penalty[high] = np.clip(1.0 - 0.25 * (corr_raw[high] - 0.92) / 0.08, 0.75, 1.0)

        weighted = robust_raw * penalty
        if float(np.sum(weighted)) <= 0.0:
            weighted = np.ones_like(weighted)
        weighted = weighted / float(np.sum(weighted))

        if mode == "robust_weighted_rank_avg":
            return self._cap_weights(weighted)

        if mode == "robust_weighted_rank_avg_shrunk":
            shrunk = 0.50 * equal + 0.50 * weighted
            shrunk = shrunk / float(np.sum(shrunk))
            return self._cap_weights(shrunk)

        raise ValueError(f"Unsupported reliability mode: {mode}")

    @staticmethod
    def _cap_weights(weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("Weights must be 1D.")
        if len(w) == 0:
            raise ValueError("Weights cannot be empty.")

        w = np.clip(w, 0.0, None)
        if float(np.sum(w)) <= 0.0:
            w = np.full(len(w), 1.0 / float(len(w)), dtype=float)
        else:
            w = w / float(np.sum(w))

        max_weight = min(0.35, 2.5 / float(len(w)))

        for _ in range(12):
            over = w > max_weight
            if not np.any(over):
                break

            excess = float(np.sum(w[over] - max_weight))
            w[over] = max_weight

            under = ~over
            if not np.any(under):
                w = np.full(len(w), 1.0 / float(len(w)), dtype=float)
                break

            under_sum = float(np.sum(w[under]))
            if under_sum <= 0.0:
                w = np.full(len(w), 1.0 / float(len(w)), dtype=float)
                break

            w[under] = w[under] + excess * (w[under] / under_sum)

        w = np.clip(w, 0.0, None)
        return w / float(np.sum(w))

    def _evaluate_config(
        self,
        tag: str,
        config: ModelConfig,
        cv_artifacts: tuple[RepeatArtifacts, ...],
        cv_folds: list[list[tuple[np.ndarray, np.ndarray]]],
        y_values: np.ndarray,
    ) -> SearchResult:
        weights = self._build_head_weights(config.reliability_mode, config.head_indices)

        per_repeat: list[float] = []
        fold_sharpes: list[float] = []

        for repeat_idx, artifact in enumerate(cv_artifacts):
            head_ranks = artifact.oof_head_rank[np.asarray(config.head_indices, dtype=int), :]
            positions = self._positions_from_head_rank_matrix(
                head_ranks=head_ranks,
                weights=weights,
                consensus=config.consensus,
                template=self.base_mapping,
            )
            repeat_sharpe = self._sharpe_from_positions(positions, y_values)
            per_repeat.append(float(repeat_sharpe))

            for _, valid_index in cv_folds[repeat_idx]:
                fold_sharpe = self._sharpe_from_positions(positions[valid_index], y_values[valid_index])
                fold_sharpes.append(float(fold_sharpe))

        mean_sharpe = float(np.mean(np.asarray(per_repeat, dtype=float)))
        std_sharpe = float(np.std(np.asarray(per_repeat, dtype=float), ddof=0))
        robust_score = float(mean_sharpe - self.robust_std_penalty * std_sharpe)

        return SearchResult(
            tag=tag,
            config=config,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            robust_score=robust_score,
            per_repeat_sharpes=tuple(per_repeat),
            fold_sharpes=tuple(fold_sharpes),
        )

    def _pick_best_result(self, results: list[SearchResult]) -> SearchResult:
        if not results:
            raise ValueError("Expected at least one result to compare.")

        best = results[0]
        for result in results[1:]:
            if self._is_better_result(result, best):
                best = result
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

        cand_complexity = self._config_complexity(candidate.config)
        inc_complexity = self._config_complexity(incumbent.config)
        if cand_complexity < inc_complexity - 1e-12:
            return True
        if not np.isclose(cand_complexity, inc_complexity, atol=1e-12):
            return False

        return candidate.config.name < incumbent.config.name

    @staticmethod
    def _config_complexity(config: ModelConfig) -> float:
        reliability_cost = {
            "equal_rank_avg": 0.0,
            "robust_weighted_rank_avg": 0.15,
            "robust_weighted_rank_avg_shrunk": 0.08,
        }[config.reliability_mode]
        head_cost = 0.015 * float(len(config.head_indices))
        consensus_cost = 0.10 * float(config.consensus.override_count)
        return reliability_cost + head_cost + consensus_cost

    def _positions_from_head_rank_matrix(
        self,
        head_ranks: np.ndarray,
        weights: np.ndarray,
        consensus: ConsensusConfig,
        template: MappingTemplate,
    ) -> np.ndarray:
        if head_ranks.ndim != 2:
            raise ValueError("head_ranks must be 2D (n_heads, n_sessions).")
        if head_ranks.shape[0] != len(weights):
            raise ValueError("head_ranks and weights dimension mismatch.")

        agg_rank = np.average(head_ranks, axis=0, weights=weights)
        base_positions, bucket_id, _ = self._map_downside_to_rank_buckets_with_meta(agg_rank, template)

        rank_std = np.std(head_ranks, axis=0, ddof=0)
        frac_bottom10 = np.mean(head_ranks >= 0.90, axis=0)
        frac_bottom20 = np.mean(head_ranks >= 0.80, axis=0)
        frac_top10 = np.mean(head_ranks <= 0.10, axis=0)

        positions = base_positions.copy()
        bucket = bucket_id.copy()

        if consensus.disagreement_std_threshold is not None:
            noisy = rank_std >= float(consensus.disagreement_std_threshold)
            center_ix = len(template.positions) // 2
            move_toward_center = np.where(bucket < center_ix, bucket + 1, np.where(bucket > center_ix, bucket - 1, bucket))
            bucket[noisy] = move_toward_center[noisy]
            positions[noisy] = np.asarray(template.positions, dtype=float)[bucket[noisy]]

        worst_ix = len(template.positions) - 1
        second_worst_ix = len(template.positions) - 2

        if consensus.bottom10_threshold is not None:
            hard_tail = (bucket == worst_ix) & (frac_bottom10 >= float(consensus.bottom10_threshold))
            positions[hard_tail] = 0.0

        if consensus.bottom20_threshold is not None:
            soften_second_tail = (bucket == second_worst_ix) & (frac_bottom20 >= float(consensus.bottom20_threshold))
            positions[soften_second_tail] = np.minimum(positions[soften_second_tail], 0.75)

        if consensus.top10_threshold is not None:
            top_ix = 0
            promote_top = (
                (bucket == top_ix)
                & (frac_top10 >= float(consensus.top10_threshold))
                & (frac_bottom20 <= float(consensus.top_low_down20_max))
            )
            positions[promote_top] = np.maximum(positions[promote_top], 2.50)

        return np.clip(positions, 0.0, 2.50)

    def _predict_head_rank_matrix(self, x_split: pd.DataFrame, heads: list[TrainedHead]) -> np.ndarray:
        outputs: list[np.ndarray] = []

        for head in heads:
            x_values = x_split.loc[:, list(head.spec.columns)].to_numpy(dtype=float)
            if head.constant_prob is not None or head.scaler is None or head.model is None:
                prob = np.full(x_values.shape[0], float(head.constant_prob), dtype=float)
            else:
                x_scaled = head.scaler.transform(x_values)
                prob = head.model.predict_proba(x_scaled)[:, 1]
            outputs.append(self._fractional_rank(prob))

        return np.vstack(outputs)

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

    def _fit_heads_full_data(self, x_train: pd.DataFrame, y_values: np.ndarray, head_specs: tuple[HeadSpec, ...]) -> list[TrainedHead]:
        trained: list[TrainedHead] = []

        for head in head_specs:
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
                trained.append(
                    TrainedHead(
                        spec=head,
                        scaler=None,
                        model=None,
                        constant_prob=float(y_binary[0]),
                    )
                )
                continue

            scaler = StandardScaler()
            x_fit_scaled = scaler.fit_transform(x_fit)

            model = LogisticRegression(
                C=head.c_value,
                solver="lbfgs",
                max_iter=self.max_iter,
                random_state=head.seed + 91217,
            )
            model.fit(x_fit_scaled, y_binary)

            trained.append(
                TrainedHead(
                    spec=head,
                    scaler=scaler,
                    model=model,
                    constant_prob=None,
                )
            )

        return trained

    def _build_pruned_indices(self, target_count: int) -> tuple[int, ...] | None:
        if not self.head_diagnostics:
            return None

        target_count = max(8, int(target_count))
        total = len(self.head_diagnostics)
        if target_count >= total:
            return None

        ordered = sorted(self.head_diagnostics, key=lambda d: d.robust_score, reverse=True)
        return tuple(sorted(diag.head_ix for diag in ordered[:target_count]))

    def _build_consensus_grid(self) -> tuple[ConsensusConfig, ...]:
        return (
            self._consensus_off(),
            ConsensusConfig(
                name="tail_harden_b10_060",
                bottom10_threshold=0.60,
                bottom20_threshold=None,
                top10_threshold=None,
                top_low_down20_max=0.20,
                disagreement_std_threshold=None,
            ),
            ConsensusConfig(
                name="tail_harden_b10_075",
                bottom10_threshold=0.75,
                bottom20_threshold=None,
                top10_threshold=None,
                top_low_down20_max=0.20,
                disagreement_std_threshold=None,
            ),
            ConsensusConfig(
                name="tail_refine_b10_060_b20_060",
                bottom10_threshold=0.60,
                bottom20_threshold=0.60,
                top10_threshold=None,
                top_low_down20_max=0.20,
                disagreement_std_threshold=None,
            ),
            ConsensusConfig(
                name="tail_refine_b10_075_b20_075",
                bottom10_threshold=0.75,
                bottom20_threshold=0.75,
                top10_threshold=None,
                top_low_down20_max=0.20,
                disagreement_std_threshold=None,
            ),
            ConsensusConfig(
                name="tail_refine_plus_top",
                bottom10_threshold=0.75,
                bottom20_threshold=0.75,
                top10_threshold=0.75,
                top_low_down20_max=0.10,
                disagreement_std_threshold=None,
            ),
            ConsensusConfig(
                name="tail_refine_plus_disagreement",
                bottom10_threshold=0.75,
                bottom20_threshold=0.75,
                top10_threshold=None,
                top_low_down20_max=0.20,
                disagreement_std_threshold=0.18,
            ),
        )

    @staticmethod
    def _consensus_off() -> ConsensusConfig:
        return ConsensusConfig(
            name="off",
            bottom10_threshold=None,
            bottom20_threshold=None,
            top10_threshold=None,
            top_low_down20_max=0.20,
            disagreement_std_threshold=None,
        )

    def _build_anchor_candidate(self, subspace_library: dict[str, tuple[str, ...]]) -> CandidateConfig:
        n_heads = 12
        row_subsample = 0.70

        seed = (
            self.random_state
            + 701 * n_heads
            + int(round(row_subsample * 100.0)) * 19
            + 1237
        )

        rng = np.random.default_rng(seed)
        subspace_names = sorted(subspace_library.keys())

        heads: list[HeadSpec] = []
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
                    seed=seed + 10007 * (head_idx + 1),
                )
            )

        return CandidateConfig(
            name=f"subspace_h{n_heads}_r{int(round(row_subsample*100)):02d}_rank_avg",
            heads=tuple(heads),
        )

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

    def _map_downside_to_rank_buckets_with_meta(
        self,
        downside_score: np.ndarray,
        template: MappingTemplate,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(template.bucket_fracs) != len(template.positions):
            raise ValueError(f"Invalid template `{template.name}`: fraction/position length mismatch.")

        total = float(np.sum(np.asarray(template.bucket_fracs, dtype=float)))
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"Invalid template `{template.name}`: fractions must sum to 1.0, got {total}.")

        n = len(downside_score)
        if n == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=int), np.asarray([], dtype=float)

        order = np.argsort(downside_score, kind="mergesort")
        pct = np.empty(n, dtype=float)
        pct[order] = (np.arange(n, dtype=float) + 0.5) / float(n)

        edges = np.cumsum(np.asarray(template.bucket_fracs, dtype=float))
        edges[-1] = 1.0

        positions = np.full(n, float(template.positions[-1]), dtype=float)
        bucket_id = np.full(n, len(template.positions) - 1, dtype=int)

        lower = 0.0
        for ix, (edge, position) in enumerate(zip(edges, template.positions)):
            mask = (pct > lower) & (pct <= edge + 1e-12)
            positions[mask] = float(position)
            bucket_id[mask] = int(ix)
            lower = float(edge)

        return positions, bucket_id, pct

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
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        a_std = float(np.std(a, ddof=0))
        b_std = float(np.std(b, ddof=0))
        if a_std <= 1e-12 or b_std <= 1e-12:
            return 0.0
        corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return corr

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
