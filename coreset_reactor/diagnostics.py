"""Post-hoc gradient attribution for the frozen Milestone-1 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import default_collate

from coreset_reactor.dataset import TrainSetData
from coreset_reactor.descriptor import reaction_descriptor
from coreset_reactor.losses import (descriptor_distance, descriptor_set_loss,
                                   paired_cost, softmin, unpaired_softdtw_cost)
from coreset_reactor.model import ParallelTCN
from coreset_reactor.sampler import exact_reference_indices
from coreset_reactor.train import fixed_schedule


ROOT = Path(__file__).resolve().parents[1]


def gradient_norm(loss: torch.Tensor, model: torch.nn.Module) -> float:
    gradients = torch.autograd.grad(loss, tuple(model.parameters()),
                                    retain_graph=True, allow_unused=True)
    squared = sum(float(grad.detach().float().square().sum())
                  for grad in gradients if grad is not None)
    return math.sqrt(squared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=("B0", "B1", "B3"), default="B3")
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    data = TrainSetData(Path(config["data_root"]), Path(config["cache_path"]),
                        frames=config["model_frames"],
                        segments=config["descriptor_segments"], seed=config["seed"])
    schedule = fixed_schedule(data, config["steps"], config["batch_size"], config["seed"])
    checkpoint = torch.load(args.run / f"m1_{args.arm}_seed{config['seed']}" /
                            "final_checkpoint.pt", map_location="cpu")
    device = torch.device(args.device)
    model = ParallelTCN(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scaler = data.scaler.to(device)
    probes = sorted({1, max(1, config["steps"] // 4),
                     max(1, config["steps"] // 2),
                     max(1, config["steps"] * 3 // 4), config["steps"]})
    records = []
    for step in probes:
        session, indices = schedule[step - 1]
        data.dataset.set_epoch(step - 1)
        samples = [data.dataset[index] for index in indices]
        batch = default_collate(samples)
        session_indices = data.state["sessions"][session]
        count = 10 if args.arm == "B0" else 18
        ids = [exact_reference_indices(session_indices,
                                       data.paired_index_for(sample), count,
                                       config["seed"], step, sample["clip_id"])
               for sample in samples]
        target = data.state["reactions"][torch.tensor(ids)].float().to(device)
        source_lengths = batch["source_lengths"]
        pred = model(batch["speaker_audio"].to(device),
                     batch["speaker_emotion"].to(device),
                     batch["speaker_3dmm"].to(device),
                     lengths=source_lengths)
        paired = softmin(paired_cost(pred, batch["paired_target"].to(device),
                                     batch["pair_lengths"].to(device)), 1,
                         config["softmin_temperature"]).mean()
        unaligned = softmin(unpaired_softdtw_cost(
            pred, target, frames=config["dtw_frames"],
            gamma=config["dtw_gamma"], band_ratio=config["dtw_band_ratio"],
            prediction_lengths=source_lengths),
            1, config["softmin_temperature"]).mean()
        record = {"step": step, "source_lengths": source_lengths.tolist(),
                  "paired_value": float(paired.detach()),
                  "unaligned_value": float(unaligned.detach()),
                  "paired_gradient_norm": gradient_norm(paired, model),
                  "weighted_unaligned_gradient_norm": gradient_norm(
                      config["loss"]["unaligned_sequence"] * unaligned, model)}
        if args.arm == "B3":
            gt_desc = data.state["descriptors"][session_indices].to(device)
            gt_desc = gt_desc[None].expand(len(samples), -1, -1)
            pred_desc = scaler(torch.cat([
                reaction_descriptor(pred[index:index + 1, :, :int(length)],
                                    config["descriptor_segments"])
                for index, length in enumerate(source_lengths.tolist())
            ], dim=0))
            terms = descriptor_set_loss(
                descriptor_distance(pred_desc, gt_desc),
                config["softmin_temperature"],
                config["loss"]["descriptor_valid"],
                config["loss"]["load_balance"],
                config["loss"]["max_prediction_usage"])
            weighted = config["loss"]["descriptor_cover"] * terms["total"]
            record.update(descriptor_value=float(weighted.detach()),
                          descriptor_gradient_norm=gradient_norm(weighted, model),
                          cover_value=float(terms["cover"].detach()),
                          valid_value=float(terms["valid"].detach()),
                          load_value=float(terms["load"].detach()))
        records.append(record)
        print(record, flush=True)
    output = args.run / f"m1_{args.arm}_seed{config['seed']}" / "gradient_diagnostics.json"
    output.write_text(json.dumps({"arm": args.arm, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
