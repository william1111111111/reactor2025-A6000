"""GT-oracle candidate-pool selection for the CoReSet M3 preflight.

This module is deliberately an *offline oracle*.  Ground truth is used only
to score candidates and choose the final ten; no selector produced here is an
inference method.  It is intended to answer whether existing generators have
enough support for a quality-preserving overgenerate-then-compress system.

The main path is:

    checkpoints -> 10 predictions/checkpoint -> deduplicated pool
                -> quality-only / unconstrained / quality-constrained top-10

The quality-constrained selector uses the official per-prediction FRC/F RD
terms and the official pairwise FRDiv distance.  It starts from the matched
M1 set and applies deterministic 1-swap and bounded 2-swap refinement, so a
strict run can never silently relax the quality constraint.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from coreset_reactor.evaluate import (
    _deterministic_postprocess,
    _stable_rng,
    official_prediction,
)
from coreset_reactor.descriptor import reaction_descriptor
from coreset_reactor.m2_diagnostics import (
    farthest_sum_indices,
    frdiv_from_distances,
    normalized_squared_distances,
)
from coreset_reactor.model import ParallelTCN
from coreset_reactor.paired_data import PairedReactionDataset


ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
if str(MAM_ROOT) not in sys.path:
    sys.path.insert(0, str(MAM_ROOT))

POOL_SIZE = 10
EXPECTED_MAIN = {
    "m1_seed_20260918": {"family": "m1", "seed": 20260918,
                         "route_weight": 0.10},
    "m1_seed_20260919": {"family": "m1", "seed": 20260919,
                         "route_weight": 0.10},
    "m1_seed_20260920": {"family": "m1", "seed": 20260920,
                         "route_weight": 0.10},
    "route_0.005": {"family": "route", "route_weight": 0.005},
    "route_0.010": {"family": "route", "route_weight": 0.010},
    "route_0.015": {"family": "route", "route_weight": 0.015},
    "route_0.020": {"family": "route", "route_weight": 0.020},
    "route_0.050": {"family": "route", "route_weight": 0.050},
}


@dataclass(frozen=True)
class CheckpointSpec:
    """A discovered checkpoint with sanitized provenance metadata."""

    path: Path
    label: str
    family: str
    seed: int | None
    route_weight: float | None
    is_mode_adapter: bool = False
    config: dict = field(default_factory=dict, compare=False)

    @property
    def main_candidate(self) -> bool:
        if self.is_mode_adapter:
            return False
        if self.family == "route":
            return True
        # Only the frozen M1 B3 operating point belongs in the primary pool;
        # descriptor-weight sweeps are useful provenance but would otherwise
        # quietly turn this into another weight sweep.
        return self.family == "m1" and self.route_weight is not None and \
            abs(self.route_weight - 0.10) < 1e-8

    def public_dict(self) -> dict:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "label": self.label,
            "family": self.family,
            "seed": self.seed,
            "route_weight": self.route_weight,
            "mode_adapter": self.is_mode_adapter,
            "main_candidate": self.main_candidate,
            "checkpoint_sha256": digest.hexdigest(),
            "parameter_count": self.config.get("model_parameter_count"),
        }


@dataclass
class Candidate:
    trajectory: torch.Tensor
    canonical_source: str
    canonical_family: str
    origins: list[tuple[str, int]]
    descriptor: torch.Tensor | None = None

    def public_origins(self) -> list[dict]:
        return [{"checkpoint": label, "slot": slot}
                for label, slot in self.origins]


def _nearest_config(checkpoint: Path) -> dict:
    for directory in (checkpoint.parent, *checkpoint.parents):
        config_path = directory / "config.json"
        if config_path.is_file():
            try:
                return json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
        if directory == checkpoint.anchor:
            break
    return {}


def _parse_seed(text: str, config: dict) -> int | None:
    if isinstance(config.get("seed"), int):
        return int(config["seed"])
    match = re.search(r"seed(\d+)", text)
    return int(match.group(1)) if match else None


def _classify_checkpoint(path: Path, config: dict) -> tuple[str, float | None, bool]:
    text = path.as_posix().lower()
    mode_adapter = "mode_adapter" in text
    routing = config.get("routing", {})
    route_weight = routing.get("weight")
    if route_weight is not None:
        route_weight = float(route_weight)
    if "mode_routing" in text or "routing" in text and route_weight is not None:
        return "route", route_weight, mode_adapter
    if mode_adapter:
        return "mode_adapter", route_weight, True
    if re.search(r"m1[_-].*b3", text) or "m1_full" in text:
        # The M1 baseline operating point is B3 with descriptor weight .10.
        descriptor_weight = config.get("loss", {}).get("descriptor_cover")
        return "m1", (float(descriptor_weight)
                       if descriptor_weight is not None else None), False
    return "other", route_weight, mode_adapter


def _m1_label(seed: int | None, route_weight: float | None) -> str:
    if seed is None:
        return "m1_unknown_seed"
    if route_weight is not None and abs(route_weight - 0.1) < 1e-8:
        return f"m1_seed_{seed}"
    return f"m1_seed_{seed}_w{route_weight:g}"


def _route_label(weight: float | None, seed: int | None) -> str:
    if weight is None:
        return f"route_unknown_seed_{seed}"
    return f"route_{weight:.3f}"


def discover_checkpoints(roots: Sequence[Path], explicit: Sequence[Path] = ()) -> list[CheckpointSpec]:
    """Discover final checkpoints without failing when an expected one is absent."""
    paths: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.name.endswith((".pt", ".pth", ".ckpt")):
            paths.add(root.resolve())
        elif root.exists():
            paths.update(p.resolve() for p in root.rglob("final_checkpoint.pt"))
    paths.update(p.expanduser().resolve() for p in explicit if p.exists())

    found: list[CheckpointSpec] = []
    for path in sorted(paths):
        config = _nearest_config(path)
        family, route_weight, mode_adapter = _classify_checkpoint(path, config)
        seed = _parse_seed(path.as_posix(), config)
        if family == "m1":
            label = _m1_label(seed, route_weight)
        elif family == "route":
            label = _route_label(route_weight, seed)
        elif family == "mode_adapter":
            label = f"mode_adapter_seed_{seed or 'unknown'}"
        else:
            label = f"other_seed_{seed or 'unknown'}_{path.parent.parent.name}"
        found.append(CheckpointSpec(path, label, family, seed, route_weight,
                                    mode_adapter, config))

    # Do not let two paths for the same logical checkpoint silently inflate the
    # pool.  Prefer a main model to an analysis-only duplicate, then the first
    # lexicographic path for deterministic provenance.
    chosen: dict[str, CheckpointSpec] = {}
    for spec in found:
        old = chosen.get(spec.label)
        if old is None or (spec.main_candidate and not old.main_candidate) or \
                str(spec.path) < str(old.path):
            chosen[spec.label] = spec
    return sorted(chosen.values(), key=lambda item: item.label)


def missing_expected(specs: Sequence[CheckpointSpec]) -> list[str]:
    available = {spec.label for spec in specs if spec.main_candidate}
    return [label for label in EXPECTED_MAIN if label not in available]


def _stable_model_label(spec: CheckpointSpec) -> str:
    return spec.label + (" [analysis-only]" if not spec.main_candidate else "")


def _load_models(specs: Sequence[CheckpointSpec], device: torch.device) -> dict[str, ParallelTCN]:
    models: dict[str, ParallelTCN] = {}
    for spec in specs:
        checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
        model = ParallelTCN(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model.eval().to(device)
        models[spec.label] = model
    return models


def _raw_target_paths(data_root: Path, sample: dict, seed: int) -> list[Path]:
    facial = data_root / "val" / "facial-attributes"
    paired = facial / sample["clip_id"].replace("speaker/", "listener/", 1)
    paired = paired.with_suffix(".npy")
    pool = sorted((facial / "listener" / sample["session_id"]).glob("*.npy"))
    alternatives = [path for path in pool if path != paired]
    rng = _stable_rng(seed, sample["clip_id"])
    if len(alternatives) >= 9:
        selected = rng.sample(alternatives, 9)
    else:
        selected = rng.choices(alternatives or [paired], k=9)
    return [paired, *selected]


def _trajectory_key(trajectory: torch.Tensor) -> bytes:
    return trajectory.detach().cpu().contiguous().numpy().tobytes()


def _deduplicate(
    predictions: Iterable[tuple[str, int, torch.Tensor, str]],
) -> list[Candidate]:
    by_key: dict[bytes, Candidate] = {}
    for label, slot, trajectory, family in predictions:
        key = _trajectory_key(trajectory)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = Candidate(trajectory.cpu(), label, family,
                                    [(label, slot)])
        else:
            existing.origins.append((label, slot))
            # Make baseline provenance canonical when a route candidate is
            # byte-identical; this keeps the matched baseline recoverable.
            if existing.canonical_family != "m1" and family == "m1":
                existing.canonical_source = label
                existing.canonical_family = family
    return list(by_key.values())


def _frc_contribution(target: torch.Tensor, prediction: torch.Tensor) -> float:
    from framework.metrics.FRC import _func as frc_one
    return float(frc_one(target, prediction[None]))


def _frd_contribution(target: torch.Tensor, prediction: torch.Tensor) -> float:
    from framework.metrics.FRD import _func as frd_one
    return float(frd_one(target, prediction[None]))


def _quality_worker(payload: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    target, prediction = payload
    target_tensor = torch.from_numpy(target)
    prediction_tensor = torch.from_numpy(prediction)
    return (_frc_contribution(target_tensor, prediction_tensor),
            _frd_contribution(target_tensor, prediction_tensor))


def _quality_terms(candidates: Sequence[Candidate], target: torch.Tensor,
                   workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    payloads = [(target.numpy(), item.trajectory.numpy()) for item in candidates]
    if workers <= 1:
        scores = [_quality_worker(payload) for payload in payloads]
    else:
        # tslearn's DTW releases the GIL in its numerical kernels. Threads
        # avoid forking an already-initialized CUDA context and preserve the
        # exact official formulas while parallelizing independent candidates.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_quality_worker, payloads))
    frc = np.asarray([score[0] for score in scores], dtype=np.float64)
    frd = np.asarray([score[1] for score in scores], dtype=np.float64)
    return frc, frd


def _objective(distance: torch.Tensor, selected: Sequence[int]) -> float:
    indices = torch.tensor(list(selected), dtype=torch.long)
    return float(distance.index_select(0, indices).index_select(1, indices).sum())


def _feasible(selected: Sequence[int], frc: np.ndarray, frd: np.ndarray,
              frc_target: float, frd_target: float, tolerance: float = 1e-8) -> bool:
    return (float(frc[list(selected)].sum()) + tolerance >= frc_target and
            float(frd[list(selected)].sum()) <= frd_target + tolerance)


def quality_only_indices(frc: np.ndarray, frd: np.ndarray, count: int) -> list[int]:
    """Select by candidate FRC, with exact-FRD as a deterministic tie-break."""
    order = sorted(range(len(frc)), key=lambda i: (-frc[i], frd[i], i))
    return sorted(order[:count])


def _quality_seed(frc: np.ndarray, frd: np.ndarray, count: int,
                 frc_target: float, frd_target: float) -> list[int] | None:
    """Find a feasible seed when no matched baseline set is available."""
    # Start from high-FRC candidates and repair the FRD side by swaps.  This
    # fallback is not used in the normal experiment, where M1 is in the pool,
    # but makes missing-checkpoint behavior explicit instead of crashing.
    selected = quality_only_indices(frc, frd, count)
    if _feasible(selected, frc, frd, frc_target, frd_target):
        return selected
    for _ in range(2 * len(frc)):
        best = None
        for pos, old in enumerate(selected):
            for new in range(len(frc)):
                if new in selected:
                    continue
                trial = selected[:pos] + [new] + selected[pos + 1:]
                frc_gap = max(0.0, frc_target - float(frc[trial].sum()))
                frd_gap = max(0.0, float(frd[trial].sum()) - frd_target)
                key = (frc_gap + frd_gap / max(1.0, abs(frd_target)),
                       -float(frc[trial].sum()), float(frd[trial].sum()), new)
                if best is None or key < best[0]:
                    best = key, trial
        if best is None:
            break
        selected = sorted(best[1])
        if _feasible(selected, frc, frd, frc_target, frd_target):
            return selected
    return None


def refine_quality_constrained(
    distance: torch.Tensor,
    frc: np.ndarray,
    frd: np.ndarray,
    initial: Sequence[int],
    frc_target: float,
    frd_target: float,
    max_pair_candidates: int = 40,
) -> list[int]:
    """Deterministic 1-swap then bounded 2-swap FRDiv refinement."""
    selected = sorted(set(int(i) for i in initial))
    if len(selected) != POOL_SIZE:
        raise ValueError(f"selector requires exactly {POOL_SIZE} distinct candidates")
    if not _feasible(selected, frc, frd, frc_target, frd_target):
        raise ValueError("initial selector set violates quality constraints")

    def best_one_swap(current: list[int]):
        current_score = _objective(distance, current)
        outside = [i for i in range(len(frc)) if i not in current]
        best: tuple[float, tuple[int, ...]] | None = None
        for pos, old in enumerate(current):
            for new in outside:
                trial = sorted(current[:pos] + [new] + current[pos + 1:])
                if not _feasible(trial, frc, frd, frc_target, frd_target):
                    continue
                score = _objective(distance, trial)
                key = (score, tuple(-i for i in trial))
                if score > current_score + 1e-9 and (best is None or key > best[0:2]):
                    best = (score, tuple(trial))
        return list(best[1]) if best is not None else None

    while True:
        improved = best_one_swap(selected)
        if improved is None:
            break
        selected = improved

    # Exhaustive 2-swap is small for a 40-80 candidate pool, but cap the
    # outside set by marginal diversity for a predictable full-VAL runtime.
    outside = [i for i in range(len(frc)) if i not in selected]
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    marginal = {
        i: float(distance[i, selected_tensor].sum()) for i in outside
    }
    outside = sorted(outside, key=lambda i: (-marginal[i], i))[:max_pair_candidates]
    current_score = _objective(distance, selected)
    best: tuple[float, tuple[int, ...]] | None = None
    for pos_a, pos_b in itertools.combinations(range(POOL_SIZE), 2):
        for new_a, new_b in itertools.combinations(outside, 2):
            trial = selected.copy()
            trial[pos_a], trial[pos_b] = new_a, new_b
            trial = sorted(trial)
            if not _feasible(trial, frc, frd, frc_target, frd_target):
                continue
            score = _objective(distance, trial)
            key = (score, tuple(-i for i in trial))
            if score > current_score + 1e-9 and (best is None or key > (best[0], tuple(-i for i in best[1]))):
                best = score, tuple(trial)
    return list(best[1]) if best is not None else selected


def _baseline_indices(candidates: Sequence[Candidate], baseline_label: str) -> list[int]:
    return sorted(i for i, item in enumerate(candidates)
                  if any(label == baseline_label for label, _ in item.origins))


def _set_metrics(trajectories: Sequence[torch.Tensor], distance: torch.Tensor,
                 selected: Sequence[int]) -> dict:
    from framework.metrics import compute_FRVar
    chosen = [trajectories[i] for i in selected]
    return {
        "FRC": None,
        "exact_FRD": None,
        "FRDiv": frdiv_from_distances(distance, list(selected)),
        "FRVar": float(compute_FRVar([torch.stack(chosen)])),
        "count": len(selected),
    }


def _feature_summary(trajectory: torch.Tensor,
                     descriptor: torch.Tensor | None = None) -> dict:
    """Small, non-identifying behavioral statistics for Phase B attribution."""
    x = trajectory.float()
    au = x[..., :15]
    va = x[..., 15:17]
    expression = x[..., 17:25]
    velocity = x[1:] - x[:-1]
    va0 = va[0]
    displacement = torch.linalg.vector_norm(va - va0, dim=-1)
    threshold = 0.05
    above = torch.where(displacement > threshold)[0]
    if descriptor is None:
        descriptor = reaction_descriptor(x)
    if descriptor.shape != (602,):
        raise ValueError(f"expected a 602-D descriptor, got {descriptor.shape}")
    return {
        "descriptor_width": int(descriptor.numel()),
        "descriptor_l2": float(torch.linalg.vector_norm(descriptor)),
        "descriptor_mean": float(descriptor.mean()),
        "descriptor_std": float(descriptor.std(unbiased=False)),
        "au_mean": float(au.mean()),
        "au_active_fraction": float((au >= 0.5).float().mean()),
        "au_mean_range": float(au.mean(0).max() - au.mean(0).min()),
        "va_mean": [float(value) for value in va.mean(0)],
        "va_std": [float(value) for value in va.std(0, unbiased=False)],
        "expression_mean": [float(value) for value in expression.mean(0)],
        "dominant_expression": int(expression.mean(0).argmax()),
        "velocity_abs_mean": float(velocity.abs().mean()),
        "velocity_std": float(velocity.std(unbiased=False)),
        "reaction_latency_proxy": float(above[0].item() / max(1, len(x) - 1)) if len(above) else 1.0,
    }


def _aggregate_feature_summaries(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"count": 0}
    scalar_keys = ["descriptor_l2", "descriptor_mean", "descriptor_std",
                   "au_mean", "au_active_fraction", "au_mean_range",
                   "velocity_abs_mean", "velocity_std", "reaction_latency_proxy"]
    result = {"count": len(rows)}
    for key in scalar_keys:
        result[key] = float(np.mean([row[key] for row in rows]))
    result["va_mean"] = [float(np.mean([row["va_mean"][j] for row in rows])) for j in range(2)]
    result["va_std"] = [float(np.mean([row["va_std"][j] for row in rows])) for j in range(2)]
    result["dominant_expression_counts"] = dict(
        sorted(Counter(row["dominant_expression"] for row in rows).items()))
    return result


def _quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.quantile(array, .10)), "p90": float(np.quantile(array, .90)),
            "min": float(array.min()), "max": float(array.max())}


def _delta_summary(records: Sequence[dict], method: str, baseline: str = "baseline") -> dict:
    fields = ["FRC", "exact_FRD", "FRDiv", "FRVar"]
    result = {}
    for field in fields:
        deltas = [record["methods"][method][field] - record["methods"][baseline][field]
                  for record in records]
        result[field] = _quantiles(deltas)
    return result


def _markdown_report(summary: dict) -> str:
    lines = [
        "# M3 Oracle Select-10: existing candidate-pool upper bound",
        "",
        "This is a GT-oracle offline diagnostic. Ground truth is used to score and select the final ten; it is not an inference method.",
        "",
        f"- Split/scope: `{summary['split']}`, `{summary['contexts']} contexts`",
        f"- Pool construction: `{summary['pool_size_mean']:.2f}` candidates/context before selection, `{summary['unique_pool_size_mean']:.2f}` after exact deduplication",
        f"- Main candidate checkpoints: `{len(summary['main_checkpoints'])}`; analysis-only mode-adapter checkpoints: `{len(summary['analysis_checkpoints'])}`; ignored non-main checkpoints: `{len(summary['ignored_checkpoints'])}`",
        f"- Baseline: `{summary['baseline_label']}`; selection size: `{POOL_SIZE}`",
        f"- Source SHA: `{summary['source_sha256']}`",
        "",
        "## Available checkpoints",
        "",
        "| label | family | seed | route weight | main pool | SHA-256 (prefix) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["checkpoints"]:
        lines.append(f"| {item['label']} | {item['family']} | {item['seed'] or ''} | "
                     f"{item['route_weight'] if item['route_weight'] is not None else ''} | "
                     f"{'yes' if item['main_candidate'] else 'no'} | "
                     f"`{item['checkpoint_sha256'][:12]}` |")
    if summary["missing_expected"]:
        lines += ["", "Missing expected entries (not treated as an error): " +
                  ", ".join(f"`{value}`" for value in summary["missing_expected"])]
    lines += ["", "## Main selection results", "",
              "Values are means across contexts. `FRC` is higher-is-better; exact `FRD` is lower-is-better.",
              "", "| method | FRC | exact FRD | FRDiv | FRVar | strict feasible contexts |", "|---|---:|---:|---:|---:|---:|"]
    for method, values in summary["method_means"].items():
        feasible = summary["feasible_counts"].get(method, "—")
        lines.append(f"| {method} | {values['FRC']:.6f} | {values['exact_FRD']:.6f} | "
                     f"{values['FRDiv']:.6f} | {values['FRVar']:.6f} | {feasible} |")
    lines += ["", "### Relative to matched M1", "",
              "| method | ΔFRC mean/p10/p90 | ΔFRD mean/p10/p90 | ΔFRDiv mean/p10/p90 |",
              "|---|---:|---:|---:|"]
    for method in summary["delta_summary"]:
        delta = summary["delta_summary"][method]
        lines.append(f"| {method} | {delta['FRC']['mean']:.6f} / {delta['FRC']['p10']:.6f} / {delta['FRC']['p90']:.6f} | "
                     f"{delta['exact_FRD']['mean']:.6f} / {delta['exact_FRD']['p10']:.6f} / {delta['exact_FRD']['p90']:.6f} | "
                     f"{delta['FRDiv']['mean']:.6f} / {delta['FRDiv']['p10']:.6f} / {delta['FRDiv']['p90']:.6f} |")
    lines += ["", "## Full-VAL distribution and pool ceiling", ""]
    for method, metrics in summary["distributions"].items():
        lines.append(f"- `{method}`: " + ", ".join(
            f"{key}={value['mean']:.6f} (p10 {value['p10']:.6f}, p90 {value['p90']:.6f})"
            for key, value in metrics.items()))
    lines += ["", f"- Maximum single pairwise diversity in a context: mean `{summary['pool_max_pairwise']['mean']:.6f}`, p90 `{summary['pool_max_pairwise']['p90']:.6f}`.",
              f"- Maximum 10-set FRDiv available in the pool (unconstrained): mean `{summary['pool_max10_frdiv']['mean']:.6f}`.",
              "", "## Phase B attribution", "", "Selected-candidate source composition (canonical source; byte-identical aliases are retained in JSON):", ""]
    lines.append("| source | selected count | mean FRC contribution | mean exact-FRD contribution | mean min distance to selected peers |")
    lines.append("|---|---:|---:|---:|---:|")
    for source, value in summary["source_attribution"].items():
        lines.append(f"| {source} | {value['count']} | {value['mean_frc']:.6f} | "
                     f"{value['mean_frd']:.6f} | {value['mean_min_selected_distance']:.6f} |")
    lines += ["", "Behavioral statistics of selected candidates by source family:", ""]
    for family, value in summary["feature_attribution"].items():
        lines.append(f"- `{family}`: count `{value['count']}`, AU mean `{value['au_mean']:.4f}`, "
                     f"602-D descriptor L2 `{value['descriptor_l2']:.4f}` (mean `{value['descriptor_mean']:.4f}`, std `{value['descriptor_std']:.4f}`), "
                     f"AU active fraction `{value['au_active_fraction']:.4f}`, "
                     f"VA mean `[{' '.join(f'{x:.4f}' for x in value['va_mean'])}]`, "
                     f"velocity `{value['velocity_abs_mean']:.4f}`, latency proxy `{value['reaction_latency_proxy']:.4f}`.")
    lines += ["", "## Interpretation guardrails", "",
              "- This is an upper-bound diagnostic, not a deployable selector.",
              "- The strict quality-constrained row never relaxes FRC or exact FRD; slack rows are separate diagnostics.",
              "- Checkpoint/data paths, trajectories, and per-context identifiers are intentionally omitted from this tracked report. The complete per-context selection ledger is local-only JSON.",
              ""]
    return "\n".join(lines)


def run_oracle(
    checkpoint_specs: Sequence[CheckpointSpec],
    data_root: Path,
    cache_path: Path,
    device: torch.device,
    max_examples: int,
    eval_seed: int,
    baseline_label: str,
    include_analysis: bool = False,
    max_pair_candidates: int = 40,
    mam_root: Path | None = None,
    metric_workers: int = 1,
) -> dict:
    """Run Phase A/B and return a JSON-serializable sanitized summary."""
    main_specs = [spec for spec in checkpoint_specs if spec.main_candidate]
    analysis_specs = [spec for spec in checkpoint_specs if spec.is_mode_adapter]
    ignored_specs = [spec for spec in checkpoint_specs
                     if not spec.main_candidate and not spec.is_mode_adapter]
    if include_analysis:
        active_specs = [*main_specs, *analysis_specs]
    else:
        active_specs = main_specs
    if not active_specs:
        raise FileNotFoundError("no main M1/route checkpoints were discovered")
    labels = {spec.label for spec in active_specs}
    if baseline_label not in labels:
        raise FileNotFoundError(f"baseline checkpoint {baseline_label!r} is not available")

    mam_root = Path(mam_root) if mam_root is not None else MAM_ROOT
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    scaler_mean = cache["descriptor_mean"]
    scaler_std = cache["descriptor_std"]
    dataset = PairedReactionDataset(
        data_root, "val", 750, crop_mode="center",
        normalization_dir=mam_root / "external" / "FaceVerse")
    selected_indices = sorted(random.Random(eval_seed).sample(
        range(len(dataset)), min(max_examples, len(dataset))))
    if not selected_indices:
        raise ValueError("no VAL contexts selected")

    # Processor is loaded once.  Its target output depends on sequence lengths
    # and raw GTs, not on the identity of the candidate checkpoint.
    from framework.modules.post_processor import Processor
    processor = Processor(cfg_dir=str(mam_root),
                          ckpt_dir=str(mam_root / "pretrained_models" / "post_processor"),
                          clip_len_test=1000, device=device, num_preds=POOL_SIZE)
    models = _load_models(active_specs, device)
    spec_by_label = {spec.label: spec for spec in active_specs}
    records: list[dict] = []
    local_times: defaultdict[str, list[float]] = defaultdict(list)

    for context_number, index in enumerate(selected_indices):
        sample = dataset[index]
        source_length = int(sample["source_lengths"])
        inputs = (sample["speaker_audio"][None, :source_length].to(device),
                  sample["speaker_emotion"][None, :source_length].to(device),
                  sample["speaker_3dmm"][None, :source_length].to(device))
        raw_targets = [torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                       for path in _raw_target_paths(data_root, sample, eval_seed)]
        # Align the target set once using the first active model. All models
        # share the same output length for a context.
        with torch.inference_mode():
            first_model = models[active_specs[0].label]
            first_pred = official_prediction(first_model(*inputs)[0].detach().cpu())
        targets = _deterministic_postprocess(processor, first_pred, raw_targets,
                                             device, eval_seed, sample["clip_id"])
        if targets.ndim != 3 or targets.shape[0] != POOL_SIZE:
            raise ValueError(f"target processor returned {targets.shape}, expected [10,T,25]")

        generated: list[tuple[str, int, torch.Tensor, str]] = []
        for spec in active_specs:
            model = models[spec.label]
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = official_prediction(model(*inputs)[0].detach().cpu())
            local_times[spec.label].append(time.perf_counter() - start)
            for slot in range(prediction.shape[0]):
                generated.append((spec.label, slot, prediction[slot], spec.family))
        candidates = _deduplicate(generated)
        if len(candidates) < POOL_SIZE:
            raise ValueError(f"context {context_number} has only {len(candidates)} unique candidates")
        trajectories = [item.trajectory for item in candidates]
        descriptors = reaction_descriptor(torch.stack(trajectories))
        for candidate, descriptor in zip(candidates, descriptors):
            candidate.descriptor = descriptor.detach().cpu()
        distance = normalized_squared_distances(torch.stack(trajectories))
        frc, frd = _quality_terms(candidates, targets, workers=metric_workers)
        baseline = _baseline_indices(candidates, baseline_label)
        if len(baseline) != POOL_SIZE:
            raise ValueError(f"baseline {baseline_label} has {len(baseline)} unique candidates in context {context_number}")
        baseline_frc = float(frc[baseline].sum())
        baseline_frd = float(frd[baseline].sum())
        baseline_metrics = _set_metrics(trajectories, distance, baseline)
        baseline_metrics["FRC"] = baseline_frc
        baseline_metrics["exact_FRD"] = baseline_frd

        quality = quality_only_indices(frc, frd, POOL_SIZE)
        unconstrained = farthest_sum_indices(distance, POOL_SIZE)
        methods = {
            "baseline": baseline,
            "quality_only": quality,
            "max_FRDiv_unconstrained": unconstrained,
        }
        feasibility = {}
        for name, slack in (("strict_quality_constrained", 0.0),
                            ("quality_constrained_FRC_slack_0.25pct", 0.0025),
                            ("quality_constrained_FRC_slack_0.5pct", 0.005)):
            frc_target = baseline_frc * (1.0 - slack)
            seed = baseline if _feasible(baseline, frc, frd, frc_target, baseline_frd) else \
                _quality_seed(frc, frd, POOL_SIZE, frc_target, baseline_frd)
            if seed is None:
                feasibility[name] = False
                continue
            methods[name] = refine_quality_constrained(
                distance, frc, frd, seed, frc_target, baseline_frd,
                max_pair_candidates=max_pair_candidates)
            feasibility[name] = True

        method_metrics: dict[str, dict] = {}
        for name, selected in methods.items():
            metrics = _set_metrics(trajectories, distance, selected)
            metrics["FRC"] = float(frc[selected].sum())
            metrics["exact_FRD"] = float(frd[selected].sum())
            metrics["selected_sources"] = Counter(candidates[i].canonical_source for i in selected)
            metrics["selected_families"] = Counter(candidates[i].canonical_family for i in selected)
            method_metrics[name] = metrics

        # Phase B uses the strict selector when available, otherwise its most
        # permissive separately-labelled slack-0.5% row.
        attribution_method = "strict_quality_constrained" if feasibility.get(
            "strict_quality_constrained") else "quality_constrained_FRC_slack_0.5pct"
        selected = methods.get(attribution_method, baseline)
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        min_selected_distance = distance[selected_tensor][:, selected_tensor].clone()
        min_selected_distance.fill_diagonal_(float("inf"))
        selected_rows = []
        for position in selected:
            feature = _feature_summary(candidates[position].trajectory,
                                       candidates[position].descriptor)
            feature.update({"source": candidates[position].canonical_source,
                            "family": candidates[position].canonical_family,
                            "frc": float(frc[position]),
                            "frd": float(frd[position]),
                            "min_selected_distance": float(min_selected_distance[
                                selected.index(position)].min())})
            selected_rows.append(feature)
        pool_max_pairwise = float(distance.max())
        pool_max10 = frdiv_from_distances(distance, farthest_sum_indices(distance, POOL_SIZE))
        records.append({
            "context_index": context_number,
            "source_length": source_length,
            "unique_pool_size": len(candidates),
            "raw_pool_size": len(generated),
            "pool_max_pairwise": pool_max_pairwise,
            "pool_max10_frdiv": pool_max10,
            "methods": method_metrics,
            "feasible": feasibility,
            "selected_rows": selected_rows,
            "candidate_descriptor_stats": [
                {
                    "origins": candidate.public_origins(),
                    "source": candidate.canonical_source,
                    "family": candidate.canonical_family,
                    "width": int(candidate.descriptor.numel()),
                    "l2": float(torch.linalg.vector_norm(candidate.descriptor)),
                    "mean": float(candidate.descriptor.mean()),
                    "std": float(candidate.descriptor.std(unbiased=False)),
                }
                for candidate in candidates
            ],
            "candidate_sources": [candidate.public_origins() for candidate in candidates],
        })
        if context_number == 0 or (context_number + 1) % 8 == 0 or context_number + 1 == len(selected_indices):
            print(f"[oracle] processed {context_number + 1}/{len(selected_indices)} contexts",
                  flush=True)

    fields = ["FRC", "exact_FRD", "FRDiv", "FRVar"]
    method_names = ["baseline", "quality_only", "max_FRDiv_unconstrained",
                    "strict_quality_constrained",
                    "quality_constrained_FRC_slack_0.25pct",
                    "quality_constrained_FRC_slack_0.5pct"]
    method_means = {}
    distributions = {}
    feasible_counts = {}
    for method in method_names:
        method_rows = [record["methods"][method] for record in records if method in record["methods"]]
        if not method_rows:
            continue
        method_means[method] = {field: float(np.mean([row[field] for row in method_rows]))
                                for field in fields}
        distributions[method] = {field: _quantiles([row[field] for row in method_rows])
                                 for field in fields}
        feasible_counts[method] = sum(bool(record["feasible"].get(method, False))
                                       for record in records)

    source_rows: defaultdict[str, list[dict]] = defaultdict(list)
    family_rows: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        for row in record["selected_rows"]:
            source_rows[row["source"]].append(row)
            family_rows[row["family"]].append(row)
    source_attribution = {}
    for source, rows in sorted(source_rows.items()):
        source_attribution[source] = {
            "count": len(rows),
            "mean_frc": float(np.mean([row["frc"] for row in rows])),
            "mean_frd": float(np.mean([row["frd"] for row in rows])),
            "mean_min_selected_distance": float(np.mean(
                [row["min_selected_distance"] for row in rows])),
        }
    feature_attribution = {family: _aggregate_feature_summaries(rows)
                           for family, rows in sorted(family_rows.items())}
    source_sha = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        source_sha.update(path.name.encode())
        source_sha.update(path.read_bytes())
    summary = {
        "split": "VAL deterministic same-session pilot; GT-oracle selection, not inference",
        "contexts": len(records),
        "eval_seed": eval_seed,
        "metric_workers": metric_workers,
        "pool_size_mean": float(np.mean([r["raw_pool_size"] for r in records])),
        "unique_pool_size_mean": float(np.mean([r["unique_pool_size"] for r in records])),
        "unique_pool_size_distribution": _quantiles([r["unique_pool_size"] for r in records]),
        "pool_max_pairwise": _quantiles([r["pool_max_pairwise"] for r in records]),
        "pool_max10_frdiv": _quantiles([r["pool_max10_frdiv"] for r in records]),
        "baseline_label": baseline_label,
        "main_checkpoints": [spec.public_dict() for spec in main_specs],
        "analysis_checkpoints": [spec.public_dict() for spec in analysis_specs],
        "ignored_checkpoints": [spec.public_dict() for spec in ignored_specs],
        "checkpoints": [spec.public_dict() for spec in checkpoint_specs],
        "missing_expected": missing_expected(checkpoint_specs),
        "method_means": method_means,
        "distributions": distributions,
        "feasible_counts": feasible_counts,
        "delta_summary": {method: _delta_summary(records, method)
                           for method in method_means if method != "baseline"},
        "source_attribution": source_attribution,
        "feature_attribution": feature_attribution,
        "inference_latency_s_mean": {
            label: float(np.mean(values)) for label, values in local_times.items()
        },
        "source_sha256": source_sha.hexdigest(),
        "per_context": records,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mam-root", type=Path, default=MAM_ROOT,
                        help="Mam-Reactor root containing FaceVerse and post-processor assets")
    parser.add_argument("--search-root", type=Path, action="append", default=[],
                        help="Root containing ignored local reports/checkpoints; repeatable")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="Explicit final_checkpoint.pt; repeatable")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--output", type=Path, required=True,
                        help="Local JSON ledger; do not commit this file")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "coreset_reactor" / "results" / "M3_ORACLE_SELECT10.md")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--include-mode-adapter", action="store_true")
    parser.add_argument("--max-pair-candidates", type=int, default=40)
    parser.add_argument("--metric-workers", type=int, default=1,
                        help="Thread workers for independent official FRC/FRD candidate scores")
    args = parser.parse_args()

    specs = discover_checkpoints(args.search_root, args.checkpoint)
    main_specs = [spec for spec in specs if spec.main_candidate]
    if not main_specs:
        raise SystemExit("No M1/route final checkpoints found. Use --search-root or --checkpoint.")
    baseline_label = args.baseline_label
    if baseline_label is None:
        seed18 = [spec for spec in main_specs if spec.label == "m1_seed_20260918"]
        baseline_label = seed18[0].label if seed18 else sorted(spec.label for spec in main_specs)[0]
    summary = run_oracle(
        specs, args.data_root, args.cache, torch.device(args.device),
        args.max_examples, args.eval_seed, baseline_label,
        include_analysis=args.include_mode_adapter,
        max_pair_candidates=args.max_pair_candidates,
        mam_root=args.mam_root,
        metric_workers=args.metric_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_markdown_report(summary))
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "contexts": summary["contexts"],
        "main_checkpoints": [item["label"] for item in summary["main_checkpoints"]],
        "missing_expected": summary["missing_expected"],
        "method_means": summary["method_means"],
    }, indent=2))


if __name__ == "__main__":
    main()
