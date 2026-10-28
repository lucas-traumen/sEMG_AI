# Plan — sEMG gesture recognition pipeline

Ordered execution plan. Verified facts (wire format, scale factor, environment)
live in `memory.md`; the rules that bind every milestone live in `AGENT.md`.
This file holds only what is to be done, in what order, and how each milestone
proves it is finished.

## Status

- Phase: **S1 — Freeze the problem**. Nothing implemented.
- Repo holds documents only: `AGENT.md`, `memory.md`, `PLAN.md`, `README.md`.
- No code, no environment, no data. `data/`, `src/`, `scripts/`, `results/` do
  not exist yet.
- Hard blockers before S2 can run: DB6 not downloaded, local capture not built.

## Non-negotiables (apply to every milestone)

Stated once here, not repeated per milestone:

1. A milestone is finished when its **evidence** exists and a clean checkout can
   regenerate it from config. Accuracy numbers alone are not evidence.
2. The test set is read-only. If a test number moves, the run is invalid.
3. Anything unknown becomes `TODO: CONFIG_REQUIRED` and stops the milestone. A
   guessed gesture name, label id or hardware limit is a defect.
4. Update the status block here and the affected section of `memory.md` in the
   same commit that produces the evidence.

## Dependency graph

```
S1 ──┬──> S2 ──> S3 ──> S4 ──> S5 ──┐
     │                              ├──> S8
     └──> C1 ─────────────> S6 ──> S7 ┘
```

S5 and S7 converge on S8. C1 runs in parallel with S2–S5 because capturing takes
calendar time, but C1 cannot start before S1 fixes the gesture list.

---

## S1 — Freeze the problem

**Goal.** Turn an open-ended "recognise hand gestures" into a closed specification.

**Work.**
- Fix the gesture set as the intersection of what DB6 provides and what the
  local protocol will capture. Write it to `configs/gestures.yaml` with one entry
  per gesture carrying its DB6 label id and its local capture slot.
- Fix the DB6 channel and the rule that selected it. Write it to
  `configs/pipeline.yaml`.
- Fix window length, stride, and the causal filter chain parameters.
- Write both evaluation protocols as prose: **DB6 benchmark** (train and test
  within DB6) and **local evaluation** (test and adapt on own recordings).

**Done when.** `configs/` parses; every gesture in `gestures.yaml` resolves to
both a DB6 label id and a local capture slot with no gaps; both protocol
definitions exist; no numeric parameter that affects results is hardcoded
anywhere outside `configs/`.

**Evidence.** `configs/gestures.yaml`, `configs/pipeline.yaml`, protocol section
in `README.md`.

**Risk.** DB6's subject count, channel layout, sampling rate and label ids are
currently **unverified** (see `memory.md` open questions). Do not fill these in
from memory — read the database's own documentation first.

---

## C1 — Capture local pilot

**Goal.** Produce the local recordings S6 and S7 depend on. Parallel track.

**Work.**
- Build the host recorder: decode the P0 v1 binary stream per the format in
  `memory.md`, verify CRC, resync on loss, write raw `.bin` plus a decoded
  manifest.
- Commit the protocol definition out of `sEMG_firmware`'s working tree, or record
  its content hash in every dataset manifest. It currently exists only as an
  uncommitted file.
- Flash, capture the planned protocol: 2 subjects × 3 days × 2 blocks × 16
  repetitions per gesture.
- Every recording carries `subject_id`, `session_id`, `block_id`, `trial_id`,
  `gesture_label`, `sample_rate`, start/end time, quality flag.

**Done when.** Every planned session exists; the decoder reports zero CRC
failures and every `drdy_seq`/`read_seq` gap is either explained in the manifest
or the recording is rejected; `data/raw/local/` is byte-identical to what the
board sent.

**Evidence.** `data/raw/local/` + per-session manifest naming the firmware commit
and protocol build id.

---

## S2 — Preprocess and window

**Goal.** Turn immutable raw recordings into windowed tensors that cannot leak.

**Work.** Read raw + metadata; check dropped samples, saturation and missing
labels; detrend; apply the configured causal band-pass and notch; cut fixed
windows; normalise using statistics learned on the training split only; emit
tensors plus per-window metadata.

**Done when.** The split is assigned per subject/session/trial **before** any
windowing, and every window inherits its trial's split; normalisation statistics
are computed from train only and applied to validation/test; every window traces
back to a trial id; regenerating from config reproduces byte-identical output,
confirmed by a checksum in the manifest.

**Evidence.** `data/processed/db6/`, `data/processed/local/`, window manifest with
label, trial id, split and checksum per block.

**Trap.** A random window split is the classic way to reach 95% accuracy that
means nothing. The check is mechanical: no trial id may appear under two splits.

---

## S3 — Baselines

**Goal.** Establish the number the CNN has to beat.

**Work.** Extract RMS, MAV, WL features; train LDA and SVM; evaluate on exactly
the split and windows S2 produced.

**Done when.** Both models consume the same manifest as S4 will; model, scaler,
config and metrics are saved per run; a confusion matrix exists per model; the
manifest checksum appears in each result file so a mismatched comparison is
visible.

**Evidence.** `results/s3_baselines/` with metrics, confusion matrices, saved
models.

---

## S4 — Tiny 1D-CNN on DB6

**Goal.** A model small enough to deploy, evaluated honestly.

**Work.** Input `[batch, 1, samples]`; train several fixed seeds; keep the best
checkpoint by validation metric; record parameter count and MACs per
architecture tried.

**Done when.** Parameter count and MACs are recorded for **every** architecture
attempted, including the rejected ones; every seed's result is stored; results
are reported as mean and spread across seeds, not as the best seed; evaluation is
on held-out subject/session/trial.

**Evidence.** `results/s4_cnn/` with checkpoints, per-seed metrics, and an
architecture table listing params, MACs and outcome.

**Trap.** Reporting the best of ten seeds inflates the number by several points
and hides variance the deployment will actually experience.

---

## S5 — Cross-session and few-shot on DB6

**Goal.** Measure what happens across sessions, and whether a few labels recover
it.

**Work.** Evaluate on held-out session/subject; then adapt with 0, 1, 3 and 5
labelled shots.

**Done when.** The **0-shot row exists as its own baseline** and is never omitted;
the shot grid is complete (every shot count × every seed); adaptation labels come
only from the training side; the same held-out data scores every configuration.

**Evidence.** `results/s5_fewshot/` — the full grid as one table.

**Trap.** A few-shot table without a 0-shot row cannot distinguish adaptation
gain from ordinary run-to-run noise.

---

## S6 — Zero-shot on local data

**Goal.** Find out how badly DB6 transfers to our own hardware before fixing it.

**Work.** Run the S4 model unchanged on `data/processed/local/`. No retraining.

**Done when.** Local metrics are recorded **even if they sit at chance level**;
a written mismatch analysis names what broke and why (channel placement, gain or
unit, sampling rate, electrode type, contraction protocol); no parameter was
adjusted after seeing the local result.

**Evidence.** `results/s6_zeroshot/` plus the mismatch note.

**Trap.** Tuning anything after seeing this number destroys it as the floor S7
must beat.

---

## S7 — Adapt on local data

**Goal.** Recover accuracy with as few local labels as possible.

**Work.** Fine-tune or few-shot adapt using local labels; compare against the S6
floor on the same held-out local split.

**Done when.** The gain over the S6 floor is quantified with the same split and
same seeds; a null or negative result is recorded as such rather than retried
until it turns positive; the number of labels used is stated per configuration.

**Evidence.** `results/s7_adapt/` — adaptation table with the S6 floor as row 0.

---

## S8 — Compress and deploy

**Goal.** The model that actually runs, verified against the model that was
trained.

**Work.** Prune, fine-tune, quantise to INT8, deploy to the target, and compare
PC reference against device output on identical input.

**Done when.** Accuracy, latency, RAM and Flash are tabulated for pre-prune,
post-prune and INT8; the exact input vector used for the parity check is stored
alongside both outputs; the difference is quantified, not described as
"matching"; the device result is produced by the same input that produced the PC
result.

**Evidence.** `results/s8_deploy/` with the model, the quantisation parameters,
the stored input vector, both outputs, and the resource table.

---

## Definition of done — whole project

The flow is complete when, from a clean checkout and `configs/` alone, every
table, model and test vector regenerates; no trial or session leaks across
splits; DB6 and local results are reported separately with no blended figure;
adaptation results are shown against their 0-shot baselines; and the embedded
model has been compared to the PC reference on the same input.

## Milestone status

| Milestone | Status | Blocker |
|---|---|---|
| S1 Freeze the problem | not started | DB6 metadata unverified; gesture set undeclared |
| C1 Capture pilot | not started | needs S1; protocol uncommitted; no host recorder |
| S2 Preprocess | not started | needs DB6 download |
| S3 Baselines | not started | needs S2 |
| S4 Tiny CNN | not started | needs S2 |
| S5 Cross-session + few-shot | not started | needs S4 |
| S6 Zero-shot local | not started | needs C1 + S4 |
| S7 Adapt local | not started | needs S6 |
| S8 Compress + deploy | not started | needs S5 + S7 |

## Environment prerequisite

Not a milestone, but it gates S2: the system interpreter lacks scipy,
scikit-learn and torch. `Convert_material/.venv-rag` has them but belongs to
another repo. Choose a tool (venv / uv / conda) and record the choice in
`memory.md` before writing pipeline code.
