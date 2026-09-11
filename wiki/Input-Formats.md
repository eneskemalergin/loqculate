# Input formats

Auto-detect reads the first header line, in this order (`loqculate/io/readers.py`):

1. EncyclopeDIA: `numFragments` present.
2. Skyline: `Total Area Fragment`, `Peptide Sequence`, and `File Name` all present.
3. DIA-NN report: `Stripped.Sequence` and `Precursor.Quantity` both present.
4. DIA-NN precursor matrix: `Stripped.Sequence` present without that report pair. Stderr warns to prefer `diann_report.tsv`.
5. Spectronaut: `PEP.StrippedSequence` present.
6. Generic: fallback. Requires columns `peptide`, `concentration`, `area` after lower-casing names.

`--format` / `fmt=`: `auto`, `encyclopedia`, `skyline`, `diann_report`, `diann_matrix`, `spectronaut`, `generic`.

## Concentration map

CLI always takes the path. Columns `filename`, `concentration`. Use basenames.

Join behavior differs:

- Skyline: keep rows whose `File Name` is in the map, outer-merge, drop missing concentration.
- DIA-NN report: inner-merge on `File.Name`.
- EncyclopeDIA, DIA-NN matrix, Spectronaut: wide tables. Filename columns are renamed to concentrations; rows whose concentration is not in the map are dropped.
- Generic: no join. Concentration comes from the data file. The map path is still opened (`validate_concentration_map`); a missing file is a warning, not a raise.

Blank concentration cells in the map warn on stderr.

## Multiplier

`--multiplier_file`: `peptide`, `multiplier`. Applied after the reader. Inner join.

## Demo

- `data/demo/one_protein.csv`
- `data/demo/filename2samplegroup_map.csv`
- `data/demo/multiplier_file.csv`
