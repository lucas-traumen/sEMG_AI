"""sEMG DB6 channel + architecture selection pipeline.

Modules are deliberately small and single-purpose:

    config          load the YAML stack, hash it, assert the experiment scope
    audit           append-only record of which files were opened, and why
    inventory       Stage A0 -- names and byte sizes only, no signal
    matfile_source  open one .mat from a zip or a mounted directory
    gestures        the raw-label -> class-id bijection
    windowing       segment scan, fully-inside windows, per-channel scalers
    qc              per (file, channel) statistics and the channel map
    manifest        file table, role assignment, window manifest, leakage checks
    models          the three 1D-CNNs, with analytic MACs and measured FLOPs
    train           one seeded training run
    metrics         per-subject -> per-seed -> across-seed accuracy
    reporting       csv/json/parquet writers, RESULT_JSON
"""

__version__ = "1.0.0"
