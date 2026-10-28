# sEMG_AI — Memory

Verified state only. If a line here is wrong, the code that depended on it is
wrong too, so correct this file in the same commit that proves otherwise.

`Last updated: 2026-10-27`

## Repository

| Item | Value |
|---|---|
| Path | `/home/lucas/Desktop/Github/sEMG_AI` |
| Remote | `git@github.com:lucas-traumen/sEMG_AI.git` — exists, **0 refs** (created empty) |
| Local state | Not a git repository yet; no commits, no files before this doc |
| Contents | `AGENT.md`, `memory.md` only |

## Capture format — read this before writing any parser

Two formats exist in the firmware repo. They are not interchangeable.

| | Legacy ASCII | P0 v1 binary |
|---|---|---|
| Where | commit `a232244` (HEAD of `sEMG_firmware`) | **uncommitted working tree** |
| Bytes | `$<int>;` / `$<int>,<onset>;` | `A5 5A \| descriptor \| len \| payload \| crc_lo crc_hi` |
| Checksum | none | CRC16-CCITT-FALSE |
| Define | — | `Peripherals/Inc/p0_protocol.h` |

**Target the P0 v1 binary stream.** It is the only format that survives a noisy
921600-baud link: the ASCII stream carries no checksum, so a garbled sample is
indistinguishable from a real one — unacceptable when a mislabelled window
silently corrupts training data.

Uncommitted means it can change. Re-read `p0_protocol.h` at capture time and
record the git commit in the dataset manifest.

### Frame geometry

- Total length = `6 + payload_len`; header `A5 5A | descriptor | payload_len`.
- `descriptor` = version nibble `1` << 4 \| record type. `0x11` sample, `0x12`
  config, `0x13` stats, `0x14` event.
- CRC covers `descriptor || payload_len || payload` — **sync bytes excluded**.
- CRC16-CCITT-FALSE: poly `0x1021`, init `0xFFFF`, no reflect, xorout `0`.
  Check vector: `"123456789"` → `0x29B1`. Any implementation that fails this
  vector is wrong.
- Integers little-endian **except** the ADS status and channel words, which are
  copied verbatim MSB-first.

### Sample record (`0x11`, 20-byte payload)

| Offset | Field | Type |
|---|---|---|
| 0 | `drdy_seq` — DRDY interrupt counter | u32 LE |
| 4 | `read_seq` — SPI read counter | u32 LE |
| 8 | `t_us` — 1 µs timestamp | u32 LE |
| 12 | ADS status word | 3 B MSB-first |
| 15 | CH1 raw | 3 B MSB-first |
| 18 | flags | u16 LE |

`drdy_seq != read_seq` means samples were lost between the interrupt and the
SPI read — treat the affected span as a gap, never interpolate across it.

Payloads: config 48 B, stats 60 B, event 16 B.

Flags: bit0 SPI error, bit1 invalid status, bit2 DRDY FIFO loss, bit3 output
loss, bit4 UART error, bit5 timer wrap, bit6 resync.

### Scale factor

VREF 2.42 V, gain 6, 24-bit two's complement → `LSB = 48.081 nV`.
Full scale `±403.333 mV`.

```
uv = raw * 2 * 2.42 / (6 * 2**24) * 1e6      # 1 count = 0.048081 µV
```

Sanity check: 1 mV ↔ 20798 counts. The config record also reports `vref_uv`
and `gain`, so a parser reads them from the stream instead of a constant.

## Acquisition facts (verified in `sEMG_firmware/sEMG_v1.0`)

| Param | Value | Where |
|---|---|---|
| MCU | STM32F767ZITx, 216 MHz (HSE bypass 8 MHz, PLLN 216) | `.ioc`, `main.c` |
| Sample rate | 2000 SPS (`CONFIG1 0x04`) | `ads1292r.c:377` |
| Active channel | **CH1 only** — CH2 powered down (`0x81`) | `ads1292r.c:378` |
| Front end | ADS1292R, gain 6, internal VREF 2.42 V | `main.c:56-57` |
| SPI1 | 1.69 MHz (`/128`) | `main.c:510` |
| Timestamp | TIM2 `PSC=107` → 1 MHz tick, 32-bit | `main.c:48-51` |
| UART6 | 921600 8N1, TX via DMA | `main.c:588` |

Single channel is structural, not a simplification: CH2 is powered down in
hardware. Every model must accept `[batch, 1, samples]`.

### The firmware does not filter

At HEAD the sample path carries a filter chain (HP 20 Hz → comb notch 50 Hz
harmonics 1–9, Q=8 → LP 480 Hz). The working tree removes it: raw LSB only, no
scaling, no decimation, no clamping (`main.c:1198-1199`).

Consequence for this repo: **the AI pipeline owns the entire front-end filter
chain**, and it must be causal so it can be reimplemented on the device. If the
firmware later re-adds filtering, the two chains must match coefficient for
coefficient or the PC-vs-device comparison is void.

## Data

There is **no sEMG dataset anywhere on this machine.** No `.bin`, no `.csv` of
samples, no `logs/` content (`sEMG_firmware/logs/` is an empty directory, and it
is untracked). The only CSV inside `sEMG/` is `sEMG_v1.csv`, a KiCad BOM.

So: DB6 must be downloaded; local recordings must be captured. Nothing in this
repo can be trained or evaluated until one of those happens.

## Environment

| | System `python3` | `Convert_material/.venv-rag` |
|---|---|---|
| Python | 3.12.3 | 3.12.3 |
| numpy | 2.5.4 | 2.5.1 |
| matplotlib | 3.11.2 | 3.11.1 |
| scipy | **missing** | 1.18.0 |
| scikit-learn | **missing** | 1.9.0 |
| torch | **missing** | 2.13.0+cu130, `cuda.is_available() = True` |
| PyYAML | present | — |

GPU: NVIDIA RTX 3050 Laptop, 4096 MiB, driver 595.99.02.

`.venv-rag` has the right packages but belongs to another repo — do not run this
project's pipeline from it. Which tool creates this project's environment
(venv / uv / conda) is still open.

4 GiB of VRAM is enough for a tiny 1D-CNN and nothing larger. Anything requiring
a transformer on raw signal will not fit alongside the desktop compositor.

## Document RAG

```bash
cd ../Convert_material && .venv-rag/bin/python -m rag.query "<q>" --show-hits \
  --collection sEMG_material --corpus corpus/sEMG_material
```

`--collection` and `--corpus` are mandatory: `Convert_material/.env` sets
`OUTPUT=Big_data`, an unrelated corpus, so the default query silently searches
the wrong material.

Corpus sizes: `sEMG_material` 166 md docs, `Hand_gesture` 21, `Big_data` 28.

## Decisions

| # | Decision | Reason |
|---|---|---|
| 1 | Docs in English | Sharper for model-facing text; user chose it over Vietnamese. |
| 2 | Pipeline targets P0 v1 binary, not ASCII | CRC — see capture format above. |
| 3 | Two evaluation protocols, never merged | A single blended number hides which domain the model actually works in. |
| 4 | Gesture list, channel, window, filters in `configs/` | Re-tuning must never mean editing code, or results stop being comparable. |
| 5 | Split by subject/session/trial before windowing | The failure that inflates accuracy with zero real generalisation. |
| 6 | `data/raw/` immutable, `processed/` regenerated | Provenance; a stale tensor is undetectable without it. |
| 7 | Report 0-shot next to every few-shot result | Otherwise adaptation gains cannot be distinguished from baseline noise. |

## Open questions — do not guess

1. **Gesture set.** Not declared. Belongs in `configs/gestures.yaml`. Must be the
   intersection of what DB6 offers and what the local protocol captures.
2. **DB6 metadata.** Subject count, channel count, sampling rate, number of
   gestures and the exact label mapping were *not* verified here. The local RAG
   holds only a benchmark table citing Zanghieri et al. (2020b) on NinaPro DB6
   (150 ms RMS windows, TCN, 71.3/65.0). Read the database's own description
   before writing a config.
3. **DB6 channel choice.** Which of DB6's channels corresponds to our CH1
   placement is unknown; the rule must be fixed and written in config.
4. **Sampling-rate gap.** Local is 2000 SPS. DB6's rate is unverified. If they
   differ, a documented resampling step is required before any comparison.
5. **Deployment target.** STM32F767 has no NPU. INT8 inference would be
   CMSIS-NN on Cortex-M7. Not started, not sized, not measured.
6. **Environment tool.** venv / uv / conda undecided.
7. **Who filters.** Currently this repo does (firmware streams raw). Confirm the
   firmware will not re-add filtering, or the causal chain must be shared.

## Known hazards

- `sEMG_firmware` has 26 uncommitted files, including `p0_protocol.h`. The
  binary protocol has never been committed. A `git clean` there erases the
  format definition.
- The firmware's `Debug/` build is missing `stm32f7xx_hal_tim.c` and
  `stm32f7xx_hal_tim_ex.c` — a capture build may not link until that is fixed.
- `sEMG_report/report/figures/fig_motor_unit.tex:29` has an invalid TikZ node
  (`below=7mm of $(6.0,0.3)$`), so the thesis does not compile. Unrelated to
  this repo, but it blocks citing final results in the report.
