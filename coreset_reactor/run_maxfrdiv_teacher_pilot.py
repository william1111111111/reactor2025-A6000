"""Run the matched T0/T1 ConditionalREGNN teacher safety pilot."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def source_snapshot_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path_text in subprocess.check_output(
        ["git", "ls-files", "mam_reactor/regnn", "coreset_reactor"],
        cwd=root, text=True,
    ).splitlines():
        path = root / path_text
        digest.update(path_text.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def initial_model_sha256(seed: int, config: dict) -> str:
    sys.path.insert(0, str(config["repo_root"] / "mam_reactor"))
    from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ConditionalREGNN(ConditionalREGNNConfig(**config["model_args"]))
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def source_schedule_sha256(dataset_size: int, seed: int, epochs: int) -> str:
    digest = hashlib.sha256()
    for epoch in range(epochs):
        order = torch.randperm(
            dataset_size, generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        digest.update(json.dumps(order, separators=(",", ":")).encode())
    return digest.hexdigest()


def run_one(
    root: Path, data_root: Path, code_mam_root: Path, asset_mam_root: Path,
    run_root: Path,
    arm: str, rank: int, gpu: int, manifest: Path,
    session_split_manifest: Path | None,
    args: argparse.Namespace,
) -> None:
    run_dir = run_root / "arms" / arm / f"rank_{rank:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        # Import the frozen source snapshot from this worktree.  The external
        # deployment supplies only FaceVerse/assets through this variable.
        "MAM_REACTOR_ROOT": str(asset_mam_root),
        "OMP_NUM_THREADS": str(args.omp_threads),
        "MKL_NUM_THREADS": str(args.omp_threads),
        "PYTHONPATH": f"{code_mam_root}:{root}",
    })
    command = [
        sys.executable, "-m", "regnn.train_conditional_regnn",
        "--data-dir", str(data_root), "--run-dir", str(run_dir),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--workers", str(args.workers), "--precision", args.precision,
        "--seed", str(args.seed), "--fixed-pair-seed", str(args.seed),
        "--fixed-pair-rank", str(rank), "--fixed-pair-count", "10",
        "--fixed-pair-manifest", str(manifest),
        "--fixed-pair-alignment-policy", "relative_time_masked",
        "--graph-layers", str(args.graph_layers),
        "--max-train-steps", str(args.max_steps),
        "--save-every", str(args.save_every), "--print-every", str(args.print_every),
    ]
    if session_split_manifest is not None:
        command.extend([
            "--session-split-manifest", str(session_split_manifest),
            "--session-split-name", args.session_split_name,
        ])
    log_path = run_dir / "train.log"
    with log_path.open("x") as log:
        subprocess.run(command, cwd=root, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    checkpoint = run_dir / f"conditional-epoch{args.expected_checkpoint_epoch:04d}-seed{args.seed}.pth"
    if not checkpoint.exists():
        raise RuntimeError(f"missing expected pilot checkpoint: {checkpoint}")
    print(f"completed {arm} rank={rank} GPU={gpu}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mam-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument(
        "--session-split-manifest", type=Path,
        help="Optional session allowlist passed to every fixed-pair teacher.",
    )
    parser.add_argument(
        "--session-split-name", choices=("B_fit", "B_cal", "B_confirm"),
        default="B_fit",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("T0", "T1"),
                        default=["T0", "T1"],
                        help="Teacher arms to launch; default runs both.")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 5, 6])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--graph-layers", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    asset_mam_root = args.mam_root.resolve()
    code_mam_root = root / "mam_reactor"
    manifest_root = args.manifest_root.resolve()
    session_split_manifest = (
        args.session_split_manifest.resolve()
        if args.session_split_manifest is not None else None
    )
    run_root = args.run_root.resolve()
    if not args.gpus:
        raise ValueError("at least one GPU is required")
    if args.max_steps <= 0 or args.epochs <= 0:
        raise ValueError("pilot steps and epochs must be positive")
    index = json.loads((manifest_root / "manifest_index.json").read_text())
    source_sha = index["source_manifest_file_sha256"]
    manifest_paths = {
        (arm, rank): manifest_root / arm / f"rank_{rank:02d}.json"
        for arm in args.arms for rank in range(10)
    }
    if any(not path.exists() for path in manifest_paths.values()):
        raise FileNotFoundError("manifest root does not contain all T0/T1 ranks")
    record_count = len(list((data_root / "train" / "facial-attributes").glob("*/*/*.npy")))
    if record_count <= 0:
        raise FileNotFoundError("no TRAIN facial attributes")
    batches_per_epoch = record_count // args.batch_size
    if batches_per_epoch <= 0 or args.max_steps >= args.epochs * batches_per_epoch:
        raise ValueError("pilot max-steps must be below the formal schedule length")
    model_args = {
        "hidden_dim": 256, "temporal_layers": 4, "temporal_heads": 8,
        "graph_dim": 64, "graph_layers": args.graph_layers, "graph_heads": 4,
        "dropout": .1, "max_seq_len": 750, "listener_3dmm": False,
    }
    # The formal sampler drops the last partial batch. Compute the exact epoch
    # containing step 600 instead of assuming a fixed dataset size.
    args.expected_checkpoint_epoch = (args.max_steps - 1) // batches_per_epoch + 1
    provenance = {
        "experiment": "matched ConditionalREGNN teacher T0/T1 safety pilot",
        "git_commit_sha": git_sha(root),
        "source_snapshot_sha256": source_snapshot_sha256(root),
        "data_fingerprint": source_sha,
        "faceverse_stats": {
            name: file_sha256(asset_mam_root / "external" / "FaceVerse" / name)
            for name in ("mean_face.npy", "std_face.npy")
        },
        "frozen_target_manifest_source_sha256": source_sha,
        "target_manifest_sha256": {
            f"{arm}/rank_{rank:02d}": file_sha256(path)
            for (arm, rank), path in manifest_paths.items()
        },
        "arms": args.arms,
        "models_per_arm": 10,
        "seed": args.seed,
        "epochs_formal_schedule": args.epochs,
        "max_train_steps": args.max_steps,
        "batch_size": args.batch_size,
        "dataset_records": record_count,
        "batches_per_epoch": batches_per_epoch,
        "precision": args.precision,
        "fixed_pair_alignment_policy": "relative_time_masked",
        "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4, betas=(0.9,0.99))",
        "scheduler": "CosineAnnealingLR(T_max=epochs, eta_min=lr*0.05)",
        "gradient_clip_norm": 1.0,
        "model_args": model_args,
        "parameter_initialization_sha256": initial_model_sha256(
            args.seed, {"repo_root": root, "model_args": model_args}
        ),
        "source_schedule_sha256": source_schedule_sha256(
            record_count, args.seed, args.epochs
        ),
        "gpus": args.gpus,
        "expected_checkpoint_epoch": args.expected_checkpoint_epoch,
        "session_split_manifest": (
            str(session_split_manifest) if session_split_manifest else None
        ),
        "session_split_name": args.session_split_name if session_split_manifest else None,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "pilot_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    jobs = [(arm, rank, args.gpus[(rank + (0 if arm == "T0" else 1)) % len(args.gpus)], manifest)
            for arm in args.arms for rank in range(10)
            for manifest in [manifest_paths[(arm, rank)]]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(run_one, root, data_root, code_mam_root,
                                   asset_mam_root, run_root,
                                   arm, rank, gpu, manifest,
                                   session_split_manifest, args)
                   for arm, rank, gpu, manifest in jobs]
        for future in futures:
            future.result()
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
