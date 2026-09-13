# Clean-room reproduction

Build the pinned Python environment with Docker and run the benchmark module.
It reproduces only train/dev artifacts: the sealed final holdout is never
loaded by this command.

For a deterministic double-run verification, run:

    python scripts/verify-reproduction.py

Generated artifacts persist benchmark configuration, source hashes, dataset
fingerprint, preregistration fingerprint, Python version, paired-inference
seed/count, and raw observations. On another platform, discrepancies must be
reported rather than overwritten.
