# reactor2025-A6000

## Research results and process archive — 2026-10-10

The [research publication](docs/research_20261010/README.md) collects the current
method and parameters, REACT2025/2024 aggregate results, full-VAL student
comparisons, component ablations, historical protocols, negative results,
recovery records and evaluation corrections. Start with its
[complete record index](docs/research_20261010/INDEX.md).

This is a sanitized **research-record release**, not a full local source/weight
backup. Private datasets, checkpoints, per-sample records, caches and previously
excluded confidential experiments are not uploaded. Historical statements and
different experimental cohorts remain explicitly distinguished.

Source-only snapshots of Mam-Reactor and CoReSet-Reactor, plus selected sanitized aggregate CoReSet-Reactor results. No REACT2025 data, model checkpoints, pretrained post-processor, FaceVerse statistics, raw training reports, or large caches are included.

The Mam-Reactor code is under [`mam_reactor/`](mam_reactor/). Dependencies are listed in [`requirements.txt`](mam_reactor/requirements.txt) and [`requirements_regnn.txt`](mam_reactor/requirements_regnn.txt).

The independent CoReSet-Reactor Milestone 1 source, configuration, and tests are under [`coreset_reactor/`](coreset_reactor/). See its README for the local data and evaluation requirements, and [`coreset_reactor/results/`](coreset_reactor/results/) for the limited aggregate pilot summaries.
