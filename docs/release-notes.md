# Release scope

This is an independent source snapshot of the RISDA implementation. The running
research checkout and its outputs are separate. Observation models and fusion
APIs use `risda.models`, with numerical infrastructure in `appa`.

The public interface includes staged ERA5 AE/prior/observation training, two
resumable calibrators, the six Table 1 method labels, and ERA69/TEMP evaluation.
Defaults enable ATMS, AMSU-A, MHS, MWTS, MWHS and CONVI. A shared YAML timeline
controls calibration and testing. CONVI uses all valid QC surface records.
Calibration preserves centered variance for likelihoods and MSE for priorities.
Method-defining settings are selected by the launcher, and run inputs are frozen.

The reference manuscript was used to verify method names and the distinction
between PPGF and PPGF_SD. It remains a local reference file and is not included
in the code-only source archive. The archive includes documentation,
source provenance and synthetic regression tests.

Prepared observations, upstream QC/gridding, normalization arrays, weights,
experiment results, private environment files, Git history and cluster-specific
environment setup are excluded. Configure matching external assets before a
scientific run. The source candidate does not assert distribution rights for
code, data or weights; retained notices are in `NOTICE.md`.
