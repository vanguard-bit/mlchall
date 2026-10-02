# Business entity resolution: reproduction

This folder rebuilds the final submission (public F0.5 0.910): `output/v7rx/matching_results.tsv` and `output/v7rx/candidate_pairs.tsv`.

The pipeline has five stages:

1. Same-country inverted-index blocking.
2. Extra retrieval channels.
3. LightGBM pair scoring.
4. A per-entity score cutoff.
5. One-owner-per-target exclusivity.

Every path is relative to this folder. `src/paths.py` sets `ROOT` to this folder; override it with `BER_ROOT`.

## Requirements

- Linux, Python 3.12.13, 8 CPU cores. Scans run as 8 processes. Scoring runs as 2.
- About 16 GB RAM and 20 GB free disk. Run one full-corpus stage at a time.
- The official dataset zip `6ab10eb3b23ba_student_resource.zip`.
- No GPU, external API, or external data. The only pretrained model is AI4Bharat IndicXlit (`indicxlit-indic-en-v1.0`, MIT, about 11M parameters). It romanizes the 1,518 unique Indic words found in train Source 2 and 3.

## Environments

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Only for stage 3 (IndicXlit). Build fairseq 0.12.2 from source first.
python3.12 -m venv .venv-xlit
.venv-xlit/bin/pip install -r requirements-xlit.txt
```

## Data

```bash
ZIP=/path/to/6ab10eb3b23ba_student_resource.zip
mkdir -p data/train data/test
unzip -j "$ZIP" 'student_resource/dataset/train/*.tsv' -d data/train
unzip -j "$ZIP" 'student_resource/dataset/test/*.tsv' -d data/test
unzip -j "$ZIP" 'student_resource/utils/validate_submission.py' -d data
```

## Full run (training included)

Run all commands from this folder:

```bash
export PYTHONPATH="$PWD/src"
PY=.venv/bin/python
```

| Step | Command | Writes |
| --- | --- | --- |
| 1. Holdout and training pairs (24k train and 6k validation Source 1 rows) | `$PY -u src/build_scoreboard.py` | `data/scoreboard/pairs.npz`, `meta.json` |
| 2. Base matcher | `$PY -u src/sweep_decoder.py` | `data/scoreboard/lgbm_v2.txt` |
| 3. Indic word cache | `$PY -u src/extract_indic_words.py && $PY -u src/extract_test_indic.py && .venv-xlit/bin/python -u src/xlit_words.py` | `indic_words.jsonl`, `indic_xlit_cache.jsonl` |
| 4. Test blocking, top 25 name and top 25 address per Source 1 | `$PY -u src/export_fast.py` | `data/export_fast/cands_{0,1}.tsv`, `output/v2/` |
| 5. Split candidates by country | `$PY -u src/score_v2.py` | `data/export_fast/{US,India,France}.cands.tsv` |
| 6. 24k more training rows, merged | `$PY -u src/extend_train.py && $PY -u src/merge_train.py` | `extra_pairs.npz`, merged `pairs.npz` |
| 7. v5 matcher, used inside the v6 build | `$PY -u src/train_v5.py` | `data/scoreboard/lgbm_v5.txt` |
| 8. Extra channels; final candidate set | `$PY -u src/score_v6.py` | `output/v6/candidate_pairs.tsv`, `output/v6/extra/` |
| 9. Final matcher | `$PY -u src/train_v7r.py` | `data/scoreboard/lgbm_v7r.txt` |
| 10. Score v6 lists with v7r, cutoff 0.75 | `SCORE_OUT=v7r SCORE_MODEL=lgbm_v7r.txt SCORE_MIN_GAIN=0.75 $PY -u src/score_v7.py` | `output/v7r/` |
| 11. Target exclusivity | `SCORE_SRC=v7r SCORE_OUT=v7rx SCORE_MODEL=lgbm_v7r.txt $PY -u src/apply_exclusive.py` | `output/v7rx/` |

Keep the order in the table. Step 6 overwrites `pairs.npz` with the merged 48k-row set (a backup of the original 30k rows goes to `pairs_30k.npz`). Step 2 trains on the original set. Steps 7 and 9 train on the merged set.

## Short run (shipped models)

`data/scoreboard/` already holds `lgbm_v2.txt`, `lgbm_v5.txt`, `lgbm_v7r.txt`, and the IndicXlit cache. To score the test set without retraining, run steps 4, 5, 8, 10, and 11. Step 3 is also unnecessary, because the cache covers every Indic word in test.

## Check and package

```bash
python3 data/validate_submission.py \
  -m output/v7rx/matching_results.tsv \
  -c output/v7rx/candidate_pairs.tsv \
  -t data/test --check-ids
cp output/v7rx/matching_results.tsv output/v7rx/candidate_pairs.tsv ../../output/
```

The final file has 1,732,544 Source 1 rows, of which 113,169 have no match. It contains 5,295,775 matched IDs. The candidate file has 83,938,629 candidate IDs, a mean of 48.4 per Source 1 row.

## Source layout

Only the scripts in the table produce the submission. The other scripts in `src/` are measured experiments. `ber/BACKLOG.md` in the team repo records their results; none of them ship.

- `normalize.py`, `blocking.py`: tokenization and name/address blocking keys.
- `pair_features.py`, `house_features.py`, `v4_features.py`, `v7_features.py`: pair features.
- `v6_keys.py`, `eval_domain_spell.py`: v6 channel keys (accent-folded name, Indic romanization, consonant skeleton plus house number, website stem).
- `decode.py`: keeps candidates at or above the score cutoff, at most 12 IDs per row.
- `f05.py`: macro F0.5 metric.
