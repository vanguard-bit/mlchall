#!/usr/bin/env bash
# Usage: ./make_zip.sh <team_name>
set -euo pipefail

team=${1:?usage: make_zip.sh <team_name>}
cd "$(dirname "$0")"

for f in output/matching_results.tsv output/candidate_pairs.tsv; do
  [[ -s $f ]] || { echo "missing $f (copy it from output/v7rx)" >&2; exit 1; }
done

code=code/business_entity_resolution
out="${team}_submission.zip"
rm -f "$out"
zip -r "$out" \
  output/matching_results.tsv output/candidate_pairs.tsv \
  Documentation_template.md \
  "$code/README.md" "$code/requirements.txt" "$code/requirements-xlit.txt" "$code/src" \
  "$code/data/scoreboard/lgbm_v2.txt" "$code/data/scoreboard/lgbm_v5.txt" "$code/data/scoreboard/lgbm_v7r.txt" \
  "$code/data/scoreboard/indic_words.jsonl" "$code/data/scoreboard/indic_xlit_cache.jsonl" \
  -x '*/__pycache__/*'
echo "wrote $out"
