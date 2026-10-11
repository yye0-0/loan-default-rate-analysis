# Standard Linux verification evidence (synthetic only)

Successful workflow: https://github.com/yye0-0/loan-default-rate-analysis/actions/runs/38113215709

Verified code commit: `d257fa249e3b62cecea5523f5557ab3b5d82c312`.

This directory preserves selected reports, frozen model choices, dependencies,
25-test log and the two saved native CatBoost models. It is not a complete run
directory: raw generated inputs and Parquet datasets are reproducible from the
project entry point and are not duplicated here. Use a fresh output directory
to reproduce the complete experiment and run `verify_outputs.py`.

All outcomes and model scores are synthetic functional verification. These
results provide no estimate of real-loan generalization or business impact.
