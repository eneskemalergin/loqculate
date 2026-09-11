# Development

A passing test can still encode the wrong LOD or LOQ. Coverage does not prove a figure of merit is defensible.

Install as on [Installation](Installation). Then:

```bash
uv run pytest
uv run ruff check loqculate/ tests/
uv run ruff format --check loqculate/ tests/
```

CI runs those checks on Linux, macOS, and Windows.

## Layout

```text
CLI input -> io readers -> model fit -> LOD / LOQ -> CSV / plots
```

- `loqculate/cli.py`: user interaction. It calls readers and models.
- `loqculate/config.py`: defaults.
- `loqculate/io/`: readers, writers, multipliers, `CalibrationData`.
- `loqculate/models/`: fit / predict / lod / loq.
- `loqculate/utils/`: shared math.
- `loqculate/compat/`: paper-reproduction ports of `old/`.
- `loqculate/plotting/`: figures.
- `loqculate/testing/`: synthetic generators for tests and benches.

## Documentation

Wiki pages are Markdown files in `wiki/` in this repository.

## Pull requests

Open pull requests against `dev`. `main` is the last release.
