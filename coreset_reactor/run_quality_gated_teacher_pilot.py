"""Run matched multi-arm quality-gated ConditionalREGNN teacher pilots."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import coreset_reactor.run_maxfrdiv_teacher_pilot as base


def run_one(
    root: Path, data_root: Path, code_mam_root: Path, asset_mam_root: Path,
    run_root: Path, arm: str, rank: int, gpu: int, manifest: Path,
    args: argparse.Namespace,
) -> None:
    run_dir = run_root / "arms" / arm / f"rank_{rank:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
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
        "--max-train-steps", str(args.max_steps),
        "--save-every", str(args.save_every),
        "--print-every", str(args.print_every),
    ]
    log_path = run_dir / "train.log"
    with log_path.open("x") as log:
        subprocess.run(command, cwd=root, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    checkpoint = run_dir / (
        f"conditional-epoch{args.expected_checkpoint_epoch:04d}-seed{args.seed}.pth"
    )
    if not checkpoint.exists():
        raise RuntimeError(f"missing expected pilot checkpoint: {checkpoint}")
    print(f"completed {arm} rank={rank} GPU={gpu}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mam-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 5, 6])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--omp-threads", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    asset_mam_root = args.mam_root.resolve()
    code_mam_root = root / "mam_reactor"
    manifest_root = args.manifest_root.resolve()
    run_root = args.run_root.resolve()
    if not args.gpus or not args.arms:
        raise ValueError("at least one GPU and one arm are required")
    index = json.loads((manifest_root / "manifest_index.json").read_text())
    if tuple(args.arms) != tuple(index["arms"]):
        raise ValueError("launcher arms must match manifest index arms")
    manifest_paths = {
        (arm, rank): manifest_root / arm / f"rank_{rank:02d}.json"
        for arm in args.arms for rank in range(10)
    }
    if any(not path.exists() for path in manifest_paths.values()):
        raise FileNotFoundError("manifest root does not contain all requested ranks")
    record_count = len(list((data_root / "train" / "facial-attributes").glob("*/*/*.npy")))
    batches_per_epoch = record_count // args.batch_size
    if record_count <= 0 or batches_per_epoch <= 0:
        raise FileNotFoundError("no TRAIN facial attributes")
    if args.max_steps >= args.epochs * batches_per_epoch:
        raise ValueError("pilot max-steps must be below the formal schedule length")
    model_args = {
        "hidden_dim": 256, "temporal_layers": 4, "temporal_heads": 8,
        "graph_dim": 64, "graph_layers": 2, "graph_heads": 4,
        "dropout": .1, "max_seq_len": 750, "listener_3dmm": False,
    }
    args.expected_checkpoint_epoch = (args.max_steps - 1) // batches_per_epoch + 1
    provenance = {
        "experiment": "matched quality-gated Max-FRDiv teacher pilot",
        "git_commit_sha": base.git_sha(root),
        "source_snapshot_sha256": base.source_snapshot_sha256(root),
        "data_fingerprint": index["source_manifest_file_sha256"],
        "frozen_quality_manifest_sha256": index["source_manifest_file_sha256"],
        "target_manifest_sha256": {
            f"{arm}/rank_{rank:02d}": base.file_sha256(path)
            for (arm, rank), path in manifest_paths.items()
        },
        "arms": list(args.arms),
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
        "parameter_initialization_sha256": base.initial_model_sha256(
            args.seed, {"repo_root": root, "model_args": model_args}
        ),
        "source_schedule_sha256": base.source_schedule_sha256(
            record_count, args.seed, args.epochs
        ),
        "gpus": args.gpus,
        "expected_checkpoint_epoch": args.expected_checkpoint_epoch,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "pilot_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    jobs = [
        (arm, rank, args.gpus[(rank + arm_index) % len(args.gpus)],
         manifest_paths[(arm, rank)])
        for arm_index, arm in enumerate(args.arms)
        for rank in range(10)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(
            run_one, root, data_root, code_mam_root, asset_mam_root,
            run_root, arm, rank, gpu, manifest, args,
        ) for arm, rank, gpu, manifest in jobs]
        for future in futures:
            future.result()
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
