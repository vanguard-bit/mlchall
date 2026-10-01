# mlchall

Pipeline for the [Amazon ML Challenge](https://amazonml.io/) business entity resolution task: blocking, pair features, LightGBM matching, and test export.

Code lives under `ber/`. See [ber/README.md](ber/README.md) for the core run path.

## Setup

1. Clone this repo and place the official student resource zip at the repo root (filename in `ber/src/paths.py`, or set `MLCHALL_ZIP`).
2. Create a venv and install deps:

```bash
cd ber
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run from `ber/src` with:

```bash
export MLCHALL_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$MLCHALL_ROOT/ber/src"
```

Dataset artifacts and model checkpoints under `ber/data/` are not in git; regenerate with the scripts in `ber/README.md`.
