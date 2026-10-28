# sEMG_AI

Adaptive single-channel sEMG hand-gesture recognition. Train on NinaPro DB6,
test the DB6 model on our own recordings, adapt it with few labels, then shrink
the survivor until it runs on the embedded target.

A run of this project delivers one of two things: a **reproducible artifact**
(config + code + manifest → data, model, table) or a **CONFIG_REQUIRED stop**.
Nothing else counts as progress.

## The product

One gesture label, one confidence score. That is the entire output.

**Out of scope, permanently:** muscle-fatigue analysis, force estimation, any
clinical or diagnostic claim. If a request drifts there, say so and stop.

## Read before starting

| File | Holds |
|---|---|
| `memory.md` | Verified state, decisions, open questions. Read it first; update it after every verified change. |
| `PLAN.md` | Milestones S1–S8 and capture track C1, each with the artifact that proves it done. Read it before starting work, update the status table when one finishes. |
| `README.md` | The command sequence from config to deployment. Kept in step with the code. |

Sibling repositories — reference them, never copy them:

| Repo | Path | Role |
|---|---|---|
| `sEMG` | `../sEMG` | KiCad hardware, ADS1292R front end |
| `sEMG_firmware` | `../sEMG_firmware` | STM32F767 acquisition firmware; **wire protocol source of truth** |
| `sEMG_report` | `../sEMG_report` | XeLaTeX thesis |
| `Convert_material` | `../Convert_material` | RAG corpus + Chroma index for papers and datasheets |

## Hard rules

These are the reason the project exists. Break one and the result is worthless
regardless of accuracy.

1. **No leak.** Split by subject/session/trial *before* windowing. A window from
   a trial that appears in train must never appear in validation or test.
   Random window splits are the failure this rule prevents.
2. **Causal.** Every filter and normalisation must be reproducible on the
   device, sample by sample, with no future context. Offline-only processing
   makes the PC-vs-device comparison meaningless.
3. **Config-first.** Gesture list, channel selection, window, stride, filter
   parameters and label mapping live in `configs/`, never in code. Changing a
   hyperparameter means editing a file, not a branch.
4. **Two problems, two numbers.** DB6 benchmark and local evaluation are
   reported separately, always. Never average them, never merge them into a
   single headline figure.
5. **Fail loud.** Missing gesture names, missing label mapping, missing hardware
   limits → write `TODO: CONFIG_REQUIRED` and stop. A plausible guess is worse
   than a gap, because the gap is visible.
6. **Raw is immutable.** `data/raw/` is read-only. Everything under `interim/`
   and `processed/` is rebuilt from config by a command, or it does not exist.
7. **Report 0-shot alongside adaptation.** Few-shot accuracy without its
   zero-shot baseline hides whether adaptation did anything.

## Layout

```
configs/            gestures.yaml + pipeline config (the only tuning surface)
data/raw/           db6/ and local/ — immutable source recordings
data/interim/       filtered, resampled, aligned
data/processed/     db6/ and local/ window tensors + per-window metadata
src/                preprocessing, models, training, evaluation
scripts/            thin entry points; every step reachable by one command
results/            metrics, confusion matrices, manifests
```

Local recordings carry at minimum: `subject_id`, `session_id`, `block_id`,
`trial_id`, `gesture_label`, `sample_rate`, start/end time, quality flag.

## Pipeline order

Each step names the artifact that proves it happened. A step is done when that
artifact exists and a fresh clone can regenerate it.

1. **Freeze the problem.** Pick the gesture set shared by DB6 and local data,
   the DB6 channel, window length, stride, and the causal filter chain. Define
   the two evaluation protocols explicitly.
   *Evidence:* `configs/` parses, both protocols written down, no value hardcoded.
2. **Preprocess and window.** Read raw + metadata, check dropped samples,
   saturation and missing labels, detrend, filter, window, normalise with
   statistics learned on train only, emit tensors + window metadata.
   *Evidence:* window manifest with label, trial id and split per window.
3. **Baselines.** Features RMS, MAV, WL → LDA and SVM on exactly the same
   windows and split as the CNN.
   *Evidence:* model, scaler, config, metrics, confusion matrix saved.
4. **Tiny 1D-CNN on DB6.** Input `[batch, 1, samples]`. Record parameter count
   and MACs for every architecture tried. Train multiple fixed seeds, keep the
   best checkpoint by validation metric, report seed variance not best-seed.
   *Evidence:* checkpoints + a seed table.
5. **Cross-session evaluation and few-shot on DB6.** Evaluate on held-out
   subject/session/trial. Then adapt with 0/1/3/5 labelled shots.
   *Evidence:* the shot table, with 0-shot as its own row.
6. **Zero-shot on local data.** Run the DB6 model unchanged on our recordings.
   Expect the number to be bad; record it as the floor.
   *Evidence:* local metrics + a written note on what broke (channel, unit,
   sampling rate, electrode placement).
7. **Adapt on local data.** Fine-tune or few-shot with local labels.
   *Evidence:* adaptation table against the step-6 floor.
8. **Compress and deploy.** Pruning → fine-tune → INT8. Verify parity on the
   same input vector: reference on PC, deployed on target, difference quantified.
   *Evidence:* PC-vs-device table with accuracy, latency, RAM, Flash.

## Verification

Every claim about numbers comes with the command that produced it. Before
reporting any result, re-run the producing command on a clean checkout path and
compare to the stored manifest. A number without a manifest entry is a rumour.

Nothing is tunable on test data. If a test number moves, the run is invalid, not
improved.

## Document search

Prefer the local RAG over the web when a paper, review, or datasheet is needed:

```bash
cd ../Convert_material && .venv-rag/bin/python -m rag.query "<question>" --show-hits --collection sEMG_material --corpus corpus/sEMG_material
```

`--collection`/`--corpus` are required: the default `.env` points at `Big_data`,
not at sEMG material. Reach for the web only when the RAG returns nothing
relevant.

## Environment

Python 3.12, NVIDIA RTX 3050 Laptop (4 GiB). The system interpreter carries
numpy, matplotlib and PyYAML but **not** scipy, scikit-learn or torch — a
project-owned virtualenv is required before any pipeline code runs. Its tool
(venv/uv/conda) is an open decision, not an assumption.

Firmware and report toolchains live in their own repos; run them there.

## Conventions

- Commit what regenerates: config, code, manifest, metrics, small figures.
  Never commit `data/raw/`, checkpoints, or window tensors.
- One command per pipeline stage, and that command is the one in `README.md`.
- Keep `memory.md` a record of what is true, not what is planned. Plans belong
  in `README.md` or an issue.
