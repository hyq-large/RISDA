# Source provenance and license status

This is a locally prepared **release candidate**. The source snapshot did not
contain a root project LICENSE establishing redistribution terms. No MIT,
Apache-2.0 or other project-wide license is granted by this preparation step.
The project owner must choose an appropriate license and verify third-party
redistribution obligations before publishing this candidate as open-source code.
Method names follow the supplied manuscript, *RISDA: Reliability-Informed
Score-based Data Assimilation of Multi-Source Atmospheric Observations*. The
local manuscript is anonymized; final author and publication metadata can be
added by the project owner. The manuscript is excluded from the code archive.

`SOURCE_MANIFEST.json` records original source and prepared-export SHA256 hashes.
Original paths for the renamed model package are omitted; its original source
content hashes are retained. New CLI/config/documentation files are identified
separately. Files come from the actual working tree, including
uncommitted research changes; a Git commit alone would not identify this snapshot.

The existing code carries the following explicit references, retained in place:

| Code | Reference in source |
| --- | --- |
| `appa/soap.py` | Adapted from https://github.com/nikhilvyas/SOAP |
| `appa/grid.py` | GraphCast: https://github.com/google-deepmind/graphcast |
| `appa/sampling.py` | https://github.com/crowsonkb/k-diffusion and https://github.com/francois-rozet/azula |
| `appa/math.py` | https://github.com/probabilists/zuko/ |
| `appa/nn/layers.py` | SIREN: https://github.com/vsitzmann/siren |
| `appa/nn/swin.py` | Swin/GraphCast-related references embedded in the implementation |
| `appa/diffusion.py` | Diffusion/MMPS method references embedded in docstrings |

A reference is evidence for a provenance review, not proof that a particular
upstream license applies to every line. Verify the actual revisions and retain
their required notices when licensing the release. Installed dependencies listed
in `pyproject.toml` retain their own licenses; their source is not vendored here.

Data and trained weights are not included and need their own distribution terms.
Private paths, credentials, agent/editor settings, Git history, experiment outputs
and raw observations are excluded from this code package.

