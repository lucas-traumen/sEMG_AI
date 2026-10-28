# sEMG_AI

Adaptive single-channel sEMG hand-gesture recognition for an embedded target:
train on NinaPro DB6, test the DB6 model on our own recordings, adapt with few
labels, then compress the survivor to run on-device.

Output is a gesture label and a confidence score. Fatigue analysis, force
estimation and clinical claims are out of scope.

## Status

Scaffold only. No code, no data, no environment yet.

| File | Holds |
|---|---|
| `AGENT.md` | Rules that bind the whole project — the seven hard constraints |
| `PLAN.md` | Milestones S1–S8 and C1, each with its completion evidence |
| `memory.md` | Verified facts: wire format, scale factor, environment, decisions |

## Commands

The command sequence from data preparation to deployment — one command per stage,
in exact order — is the contract this repo must meet. It is tracked milestone by
milestone in `PLAN.md`; entries appear here as each stage lands.

Until a command is listed below, treat it as nonexistent. A command that does not
yet exist is not "roughly this" — it is missing.

| Stage | Command |
|---|---|
| — | none yet |

## Before writing code

Read `AGENT.md`. The rules that bite hardest: no trial or session leakage across
splits, causal-only filtering, and gesture/label definitions in `configs/` rather
than in code.

## Layout

```
configs/     gestures.yaml + pipeline config — the only tuning surface
data/raw/    immutable recordings (db6/ and local/)
data/        interim/ and processed/ — regenerated from config, never edited
src/         preprocessing, models, training, evaluation
scripts/     one entry point per stage
results/     metrics, confusion matrices, manifests
```
