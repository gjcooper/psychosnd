# Contributing to psychosnd

## Development Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --group dev
```

This creates `.venv/` and installs psychosnd in editable mode automatically.

## Common Tasks

```bash
# Run the CLI
uv run psych-scla <soundfile> <logfile>
uv run psych-scla --help

# Run tests
uv run coverage run -m pytest test/
uv run coverage report

# Update the lockfile after editing pyproject.toml
uv lock

# Build distribution artifacts
uv build

# Publish to PyPI
uv publish
```

## Release Process

1. Bump `version` in `pyproject.toml`.
2. `git commit -am "Bump version to X.Y.Z"`
3. `git tag vX.Y.Z && git push --tags`
4. `uv build`
5. `uv publish`
