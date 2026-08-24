# fermi_data/

This directory is where the pipeline expects the processed *Fermi*-LAT
counts and exposure cubes (`fermi_data/totani/processed/counts_ccube_*.fits`,
`expcube_*.fits`) once you have built them.

Nothing under `fermi_data/` other than this README is tracked in the
repository — the raw and processed *Fermi*-LAT products are large (each
counts/exposure/template cube is ~45–50 MB) and are exactly reproducible
from the public archive. See **[docs/DATA_FETCH.md](../docs/DATA_FETCH.md)**
for the exact download query and processing commands (event selection, time
range, IRF version, binning).

Override the location with `FERMI_DATA_ROOT=/path/to/fermi_data` if you keep
it somewhere else (see `shared/paths.py`).
