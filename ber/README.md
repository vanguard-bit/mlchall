# Business entity resolution pipeline

1. **`eval_blocking.py`** — candidate recall vs K (blocking ceiling).
2. **`build_matcher_dataset.py`** — blocked pairs + features → `data/matcher_sample.npz`.
3. **`train_matcher.py`** — LightGBM pair model + validation F0.5 decode tune.
4. **`export_test.py`** — full test block → LightGBM score → `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

```bash
export MLCHALL_ROOT="$HOME/projects/mlchall"
export PYTHONPATH="$MLCHALL_ROOT/ber/src"
$MLCHALL_ROOT/ber/.venv/bin/python eval_blocking.py
$MLCHALL_ROOT/ber/.venv/bin/python build_matcher_dataset.py
$MLCHALL_ROOT/ber/.venv/bin/python train_matcher.py
$MLCHALL_ROOT/ber/.venv/bin/python export_test.py
```

Run only one full-corpus zip scan at a time (~6GB RAM headroom recommended).

Default zip: `$MLCHALL_ROOT/6ab10eb3b23ba_student_resource.zip`

Final submission package (v7rx, public 0.910): `submission/`. Build the zip with `submission/make_zip.sh <team_name>`.
