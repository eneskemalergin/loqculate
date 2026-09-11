<!-- markdownlint-disable MD010 MD033 MD036 MD041 -->
<p align="center">
  <img src="assets/loqculate-readme-header.svg" alt="loqculate" width="420">
</p>

<p align="center">
  LOD and LOQ calculator for mass-spectrometry calibration curves.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10--3.12-2D7D46?style=flat-square&logo=python&logoColor=white" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/version-0.4.1-8B5CF6?style=flat-square" alt="v0.4.1">
  <img src="https://img.shields.io/badge/status-alpha-C17D10?style=flat-square" alt="Alpha">
  <a href="https://github.com/eneskemalergin/loqculate/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/eneskemalergin/loqculate/ci.yml?branch=main&style=flat-square&logo=github&label=CI" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-MIT-4B9D6E?style=flat-square" alt="MIT">
</p>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-CHANGELOG-E05D44?style=flat-square" alt="Changelog"></a>
  <a href="CITATION.cff"><img src="https://img.shields.io/badge/cite-CITATION.cff-0066CC?style=flat-square" alt="Citation"></a>
  <a href="https://github.com/eneskemalergin/loqculate/wiki"><img src="https://img.shields.io/badge/docs-Wiki-0F766E?style=flat-square" alt="Docs"></a>
</p>

loqculate estimates LOD and LOQ from mass-spectrometry calibration curves. It reads a search-engine export, maps run filenames to concentrations, fits one curve per peptide, and writes `figuresofmerit.csv`.

The default model is `PiecewiseCF`. Default LOQ is bootstrap. `OriginalWLS` and `OriginalCV` are the Pino 2020 scripts as Python classes. Not on PyPI.

## Quick start

The source checkout uses [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
git clone https://github.com/eneskemalergin/loqculate
cd loqculate
uv sync
uv run loqculate fit data/demo/one_protein.csv data/demo/filename2samplegroup_map.csv
```

## Documentation

The [wiki](https://github.com/eneskemalergin/loqculate/wiki) has the details:

- [Getting started](https://github.com/eneskemalergin/loqculate/wiki/Getting-Started): concentration map, demo run, output files
- [CLI reference](https://github.com/eneskemalergin/loqculate/wiki/CLI-Reference): `fit`, `compare`, flags, and which flags are actually forwarded
- [Python API](https://github.com/eneskemalergin/loqculate/wiki/Python-API): public imports and `CalibrationData`
- [Models](https://github.com/eneskemalergin/loqculate/wiki/Models): which model to use, then a page per model
- [Input formats](https://github.com/eneskemalergin/loqculate/wiki/Input-Formats): auto-detect order and map joining
- [Benchmarks](https://github.com/eneskemalergin/loqculate/wiki/Benchmarks): what was measured, on which harness
- [Development](https://github.com/eneskemalergin/loqculate/wiki/Development): tests, layout, pull requests

## Citation

If you use loqculate, cite the Pino 2020 method and this software:

> Pino, L.K. et al. (2020). *Matrix-matched calibration curves for assay characterization in data-independent acquisition proteomics.* Analytical Chemistry. <https://doi.org/10.1021/acs.analchem.9b04826>

```bibtex
@software{ergin_loqculate_2026,
  author    = {Ergin, Enes Kemal},
  title     = {loqculate: Limit of Detection and Quantitation calculator
               for mass-spectrometry calibration curves},
  year      = {2026},
  version   = {0.4.1},
  url       = {https://github.com/eneskemalergin/loqculate},
  license   = {MIT},
}
```

`CITATION.cff` is in the repository root.

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

*Three points hold the line,*  
*Through the sliding window's gaze,*  
*Noise becomes the truth.*

</div>
