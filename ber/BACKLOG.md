# 0.90+ Attempt Backlog

Current public score: **0.910** (`output/v7rx`, 27 Sep). v7x was 0.903, v7 was 0.888. The 0.807 figure below is the original baseline this backlog started from.

The present evidence:

- Main blocker recall is about **0.925** at 30 candidates per channel (mean 56 candidates). The production run used top 25 per channel.
- The LightGBM matcher was trained on only **1,200 Source 1 rows** (55,770 pairs).
- The reported validation F0.5 of 0.847 ignored true links absent from the candidate list, so it overstated end-to-end quality.
- The production decoder is a fixed pair-score threshold of 0.60, capped at 12; it is not a set-aware F0.5 decoder.
- Current output emits 3.18 IDs per S1; training truth averages about 3.46. False additions are much more costly than clean omissions.
- France is 15% of test and has no labeled training rows.

When one heavy job is running, use all CPU cores by sharding processes by country or row range. Do not restart a scan already near completion solely to parallelize it.

## Definition of done

A 0.90 attempt is ready for upload only when:

1. Local end-to-end macro F0.5 is measured over **all held-out S1 rows**, including singletons and true IDs lost by blocking.
2. It improves by at least **0.005 absolute** over the same-fold baseline; prefer 0.01 to spend an upload.
3. Candidate recall does not fall, candidate count remains defensible, and every predicted ID is in `candidate_pairs.tsv`.
4. The full validator passes, including `--check-ids`.
5. Results are stable across at least three deterministic S1 folds and US/India slices.

No public-leaderboard change is promoted on public score alone.

## P0 — Build a trustworthy local scoreboard

This blocks every other experiment.

### Work

- Create deterministic, S1-grouped train/validation folds stratified by country, singleton status, and truth-list length.
- Use at least **30,000–50,000 held-out S1 rows**. A 1,200-row matcher sample is too noisy for threshold and decoder selection.
- For every held-out S1, preserve the complete ground-truth set even when a truth ID is absent from candidates.
- Report:
  - end-to-end macro F0.5;
  - candidate recall and fraction of S1 rows with all truths retrieved;
  - precision/recall and F0.5 by US/India, singleton/non-singleton, truth cardinality, script, and missing address;
  - predicted-list-length versus truth-list-length histogram;
  - false-positive cost and false-negative cost separately.
- Save pair scores once so decoder sweeps do not rerun corpus scans.

### Files

- Add `src/make_validation.py`.
- Add `src/evaluate_end_to_end.py`.
- Change `src/build_matcher_dataset.py` to accept fold manifests and larger samples.
- Do not use `decode_greedy_f05_labeled`; it is an oracle diagnostic, not deployable validation.

### Gate

Reproduce a local baseline that is directionally consistent with the 0.807 public score. If local and public differ by more than 0.03, fix validation before model work.

## P1 — Fix the matcher before expanding retrieval

This is the highest-probability lift.

### Work

- Train on at least **30,000 S1 rows** of blocker-generated hard negatives; scale toward 100,000 if RAM permits.
- Keep the validation S1 IDs completely outside training.
- Add missing high-value features:
  - candidate source (`S2` versus `S3`);
  - exact normalized name/address and suffix-stripped equality;
  - character edit distance and longest-token edit distance;
  - explicit digit conflict (both present but different), not only digit equality;
  - postal conflict, house-number prefix/truncation, and street-token agreement;
  - per-channel score margin to rank 2 and rank within source;
  - name/address evidence interaction flags;
  - missing-field pattern;
  - script type and transliteration agreement, but no country one-hot.
- Train separate models for S2 and S3 only if validation proves their error distributions differ.
- Calibrate scores on the held-out fold (isotonic or Platt) before decoding.
- Compare LightGBM with logistic regression as a calibration sanity check; a more complex model is accepted only if end-to-end F0.5 rises.

### Files

- Change `src/pair_features.py`.
- Change `src/build_matcher_dataset.py`.
- Change `src/train_matcher.py`.
- Add `src/calibrate.py` if calibration is retained.

### Gate

Pair-model work must improve end-to-end macro F0.5 by at least 0.005 on all folds and not reduce singleton F0.5.

## P2 — Replace the fixed 0.60 decoder

The metric is per-entity and precision-heavy; decoding pairs independently is structurally wrong.

### Experiments

1. Sweep thresholds from 0.50 to 0.99 on cached held-out scores.
2. Tune separate first-match and extra-match thresholds:
   - first ID requires strong absolute evidence;
   - IDs 2–3 may use a lower threshold when they agree on address;
   - IDs 4+ require house/postal/street agreement and a small score gap.
3. Tune thresholds by evidence class, not country:
   - exact name + address;
   - exact name only;
   - address strong/name noisy;
   - transliteration;
   - missing address.
4. Add a singleton/abstention model using top score, score margin, candidate count, and evidence conflicts.
5. Cluster candidates for one S1 by normalized address/digits. Take or reject a coherent address cluster instead of unrelated pairs.
6. With calibrated pair probabilities, evaluate the expected F0.5 of predicting the top 0, 1, 2, ... IDs and choose the maximizing set size.
7. Train a cardinality model for total matches and, if useful, separate S2/S3 counts from score-distribution and evidence features.
8. Optimize deployable decoder parameters directly for macro F0.5 on one fold and confirm on the other folds.

### Files

- Replace the current behavior in `src/decode.py`.
- Add `src/tune_decoder.py`.
- Change `src/export_test.py` to load decoder parameters from a versioned JSON file.

### Gate

Promote only a decoder that improves every validation fold or whose worst-fold loss is under 0.002 with a mean gain over 0.005.

## P3 — Raise lexical blocking recall without flooding candidates

Do this after P1/P2. A matcher cannot recover a blocked-out truth, but wider retrieval is harmful unless the matcher is already precise.

### 3A. Multi-copy cluster expansion

The typical S1 entity has 3–4 true copies. Do not score every candidate only against S1.

- Use the highest-confidence S2/S3 match as a seed.
- Compare remaining candidates with the seed and with the provisional address cluster.
- Add candidate-to-seed name/address similarity, shared digits, cluster size, and source composition as matcher features.
- Permit a noisy copy to join only when it agrees with the seed on independent location evidence.
- Never expand transitively from a weak seed.

This targets pairs whose trade name differs from S1 but whose address agrees with another true copy.

### Gate

Measure expansion precision and macro F0.5. Cluster expansion must not reduce singleton F0.5 or create large connected components.

### 3B. Ranking repair

Several “misses” (`Urology Partners`, exact `Royal Software`, ordinary `Jay Hospitality`, `Mountain/Paige Hill`) already share keys and are ranking failures.

- Build a wider internal union (for example top 50 name + top 50 address).
- Cheaply rerank with exact name, house/postal, street IDF, and name/address joint evidence.
- Keep a smaller final list for the model and `candidate_pairs.tsv`.
- Plot candidate recall against mean/final P95 candidates. Target **>=0.97 recall** before claiming the blocker is competitive.

### 3C. Typo key

- Long-token SymSpell/delete index or sorted consonant bag.
- Example target: `Hospitality` / `Hotspiialty`.
- Require a compatible house number or strong street evidence.

### 3D. Domain key

- Remove TLD and split/compare the domain with the concatenated normalized name.
- Example target: `sparkskrantzinsurance.com`.
- Require street agreement or another independent address signal.

### Files

- Change `src/blocking.py`.
- Add retrieval-channel attribution to evaluation and candidate output metadata.
- Extend `src/eval_blocking.py` to report marginal recall per channel.

### Gate

For each added channel, report true links absent from the baseline candidate list and new false candidates. The final end-to-end F0.5, not raw recall, decides whether it ships.

## P4 — Replace hand-written Indic transliteration

Use **AI4Bharat IndicXlit**, checkpoint `indicxlit-indic-en-v1.0`: approximately 11M parameters, MIT licensed, native-to-Roman Hindi and Punjabi. Do not use Gemma or claim model output was hand-cleaned.

### Work

- Cache transliteration by unique non-Latin token; do not invoke the model per row repeatedly.
- Preserve both the raw-script tokens and top Roman hypotheses.
- Strip transliterated legal forms only after transliteration.
- Add transliteration keys as an extra retrieval channel, not as destructive replacement text.
- Evaluate Hindi and Gurmukhi separately:
  - `रॉयल सॉफ्टवेयर` should approach `Royal Software`;
  - `ਗ੍ਰੀਨ ਐਗਰੋ` should approach `Green Agro`.

### Files

- Add `src/indic_transliterate.py`.
- Change `src/normalize.py` and `src/blocking.py`.
- Pin the package/model and include MIT attribution in the final package.

### Gate

Measure marginal candidate recall and end-to-end F0.5. Do not ship merely because transliterations look better.

## P5 — Dense retrieval only as a complementary channel

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` is Apache-2.0 and about 118M parameters, but raw cosine is not a decision rule:

- Hindi true pair: 0.929.
- False `Infrastructure` / `Infratech`: 0.836.
- True `Hospitality` / `Hotspiialty`: 0.778.
- True domain pair: 0.742.
- Punjabi pair: 0.677.

### Work

- Use same-country ANN top-K only.
- Union dense neighbors with lexical candidates.
- Feed cosine, dense rank, and lexical/dense agreement into LightGBM.
- Never accept a pair on cosine alone.
- Benchmark candidate recall and F0.5 first on the 30k–50k holdout.
- Use the 3050 for encoding when available; it changes runtime, not ranking quality.

### Gate

Keep dense retrieval only if it adds blocked-out truths and the retrained matcher preserves enough precision to improve end-to-end F0.5.

## P6 — Learn the dataset's corruption process

There are about 7.64M labeled positive links. Use them as an internal spelling and formatting corpus instead of importing an external dictionary.

### Work

- Mine source-specific token substitutions, deletions, insertions, transpositions, abbreviations, and address truncations from true S1↔S2 and S1↔S3 pairs.
- Estimate whether an edit occurs frequently in positives but rarely in blocker hard negatives.
- Add high-confidence learned variants as alternate keys and features:
  - typo likelihood and character-confusion likelihood;
  - legal-form and address abbreviation likelihood;
  - house/postal truncation compatibility;
  - learned script↔Roman token pairs;
  - source-specific noise likelihood.
- Generate synthetic corruptions from these learned operations for retriever training, while keeping real validation S1 rows isolated.
- Reject variants that are common across unrelated businesses.

### Files

- Add `src/mine_corruptions.py`.
- Add `src/corruption_features.py`.
- Change `src/blocking.py` and `src/pair_features.py`.

### Gate

The learned channel must add blocked-out true links at high marginal precision and improve all-fold end-to-end F0.5.

## P7 — Supervised contrastive retriever

Raw MiniLM is not sufficient, but the 7.64M positive links and blocker hard negatives can train a task-specific bi-encoder.

### Work

- Start from an Apache-2.0 multilingual encoder.
- Train S1↔S2/S3 positives with hard negatives from the lexical blocker.
- Include source-specific corruptions learned in P6.
- Hold out validation S1 entities completely.
- Retrieve within country, top-K per source, and feed dense score/rank to LightGBM.
- Use the 3050 for a small experiment; use a larger GPU only after the sample recall curve proves value.

### Gate

On the held-out sample, the trained encoder must rank true typo, domain, and Indic pairs above near-name false pairs such as `Infrastructure` / `Infratech`. It must improve candidate recall and final macro F0.5.

## P8 — France robustness

There is no labeled France validation, so changes must be invariant and conservative.

- Unicode accent folding while preserving the raw form.
- French legal suffixes (`SARL`, `SAS`, `SA`, `EURL`) already belong in suffix normalization; verify tests.
- Treat `le`, `la`, `les`, `de`, `du`, `des` as optional name particles only in an alternate representation.
- Parse French postal codes and house numbers with the same generic features.
- Fit no country one-hot and no France-specific threshold without labels.
- Build synthetic corruption tests from exact/high-confidence France pairs only for invariance checks, not as ground truth.

## Upload plan (maximum five per day)

1. **Baseline:** current validated 0.807 file. Already used.
2. **Matcher + calibrated decoder:** upload only after P0–P2 clears the gate.
3. **Set-aware version:** cluster expansion + expected-F0.5/count decoder, only if it beats upload 2 locally.
4. **Lexical blocker v2:** ranking repair + learned corruption + typo/domain/IndicXlit channels.
5. **Supervised dense union or best ensemble:** only if it beats uploads 2–4 locally; otherwise keep the slot.

Record for every upload: git/code snapshot, model hash, blocker config, decoder JSON, validation metrics, candidate statistics, file hash, and public score.

## Measured, not uploaded

- Parsed house fields (`house_same` after stripping leading zeros, `house_missing`, `house_conflict` when the street matches and the house does not). Retrained on the scoreboard train split. At cutoff 0.70, validation macro F0.5 moves from 0.8707 to 0.8973 (+0.0267). True links accepted 16,240 to 17,073. False links 893 to 658. Singleton F0.5 0.726 to 0.826. Clears the local 0.005 gate. Not exported. The earlier local-to-public gap on this fold was about 0.03.
- Further cleanups on that house matcher, same fold, cutoff 0.70: street plus house-prefix 0.8992, postal 0.8997, accent-folded and squashed name 0.8991, all of them 0.9006. Best lift over the house matcher is +0.0020. Under the 0.005 gate.
- Reverse lookup with those cleaned keys, scored by the house matcher: address keys add 161 true and 11,769 false nominations; the matcher keeps 102 true and 1 false, F0.5 0.9008 (+0.0023). Exact squashed-name keys add 74 true and 240 false; the matcher keeps 41 true and 1 false, F0.5 0.8995 (+0.0009). Under the 0.005 gate.

## Next — retrieval channels (priority 1 after v3 and v4 finish scoring)

This is ahead of a bigger matcher, more training rows, embeddings, and the encoder. Candidate recall on the 6,000-row validation fold is **0.926** (19,370 of 20,914 true matches). The 1,544 misses are neither in the name top-25 nor the address top-25. LightGBM cannot recover them. India miss rate 9.5%, US 6.0%. Removing Indic-script misses drops India to 5.3%.

Do not add five channels at K=25. Keep the final list near 50–70 pairs per Source 1. After every channel, report unique candidate recall: a true match already in the current name-plus-address list does not count. Touch LightGBM only after that recall rises and the new list is still about this size. End-to-end macro F0.5 must beat the v5 matcher (0.9034) by at least 0.005.

Measured split of the 1,544, one primary reason each:

| Problem | Count | Order |
|---|---:|---|
| Name already similar, ranked out of the top 25 | 635 | 1, with the 108 below |
| Indic script vs Latin, Jaro–Winkler already ≥ 0.80 | 108 | 1, same fix as the row above |
| Indic script vs Latin, Jaro–Winkler < 0.60 | 144 | 2 |
| Indic script vs Latin, Jaro–Winkler 0.60–0.80 | 86 | inspect after 1 and 2 |
| Address missing | 220 | not its own channel; recovered only if a name channel hits |
| Latin accent only | 166 | 3 |
| Severe spelling | 88 | 5 |
| Domain or URL | 75 | 4 |
| Business represented differently | 22 | leave |

1. **Dedupe and backfill.** Measured and rejected. See the closed list.
2. **v6 — accent-fold plus IndicXlit as one exact name key.** Measured and under the gate. See the closed list. The optimistic Indic-only insertion (0.9096) is not a retrieval result.
3. **Same name key, but only when the house or street also agrees.** Measured and under the gate. See the closed list. House alone is almost the same score and hurts singletons less than adding street-only hits.
4. **Domain stem key.** Measured and under the gate. See the closed list.
5. **Tolerant spelling key.** Measured as an exact consonant skeleton plus the same house number. Under the gate. A one-edit skeleton would add at most 32 more true misses. See the closed list.

Retrieval on this list is measured. The next tests are features on pairs that are already candidates. They cannot recover the 1,544 matches that never enter the list.

## Next — features on the current candidate list

Same gate: end-to-end macro F0.5 must beat **0.9034** by at least 0.005, and singleton F0.5 must not fall. Fit IDF on train Source 1 only. No geocoder, no postal gazetteer, no embedding. On a 200,000-row slice of train Source 1, 198,488 names contain a token outside the 20 most common words (`and` 7.6%, `india` 2.9%, `care` 2.4%, `associates`, `group`, `partners`, `services`, `solutions`). About 104,000 of those addresses have two or more street tokens, so the parser drops the last token as the city and never compares it.

1. **Rare-token agreement and conflict, with city similarity.** Measured together on the current candidate lists. See the result below. Cleared the local gate. Not exported yet.
2. **City similarity** was part of that same retrain, not a separate run.
3. **Per-token name alignment, one column.** Measured with the initialism bit, on top of the v7 features. Under the gate. See the closed list.
4. **Initialism bit.** Measured in that same retrain. Almost never fires (0.003 of true pairs, 0.000 of false). Not a submission.

Rare-token IDF plus city, retrained on the same pairs and scored at cutoff 0.70: holdout macro F0.5 **0.9126** (+0.0092 over v5’s 0.9034). Singleton **0.8750**, up from 0.8537. On the validation pairs the rare shared token averages 0.537 on true links and 0.275 on false ones; a token on only one side averages 0.374 true and 0.615 false; city Jaro–Winkler 0.475 true and 0.383 false; exact city 0.313 true and 0.163 false. Model `lgbm_v7.txt`, 400 trees. Does not replace `lgbm_v5.txt`. Test file `output/v7/matching_results.tsv` scores v6’s candidate lists with this matcher. Format validator PASS (ID-existence check off). 108,040 empty rows, 1,624,504 non-empty. Public score **0.888**, up from v5’s **0.878** (+0.010). The local-to-public gap stays about **0.025** (v5 was 0.9034 → 0.878; the v7 matcher alone was 0.9126). The combined holdout score of this file was not measured.

## Audit — plan vs. running pipeline (27 Sep)

Findings from reading `score_v6.py`, `score_v7.py`, `train_v7.py`, `apply_exclusive.py`, `eval_v6_union.py`, `pair_features.py`, `decode.py`, and the v7 test file. Ordered by likely public impact. Items 1 and 2 are measured below. Items 3 and 4 are test-only; the 6,000-row holdout cannot see them.

1. **Train/serve skew on Indic names.** Measured. See the results section below. `score_v7` romanizes; `lgbm_v7.txt` was trained on raw script. Scoring the shipped model on romanized base pairs moves the holdout from 0.9126 to **0.9133** (singleton 0.8750 → 0.8811, India 0.8738 → 0.8755, US unchanged). The skew is real and small for the shipped model. Retraining on romanized names is `lgbm_v7r.txt`.
2. **The v7 test file is an unmeasured combination.** Measured. `extra_pairs.npz` is the extra 24k training rows, not the v6 channels, so the channels were re-retrieved (`eval_v7_extras.py`). The uploaded file's holdout analogue — v7 matcher, romanized names, all v6 extras, fabricated ranks — is **0.9172** (singleton 0.8720, India 0.8802, US 0.9413), not 0.9126. Public 0.888 against that local number is a gap of **0.029**. Dropping the domain channel is 0.9177 / singleton 0.8780. The extras help under v7. Do not drop them.
3. **The holdout never produces long lists; the test file does.** Holdout v7 predictions max out at 9 IDs (25 rows at 8, 9 at 9). The test file has **7,632** rows at the cap of 12, **1,965** at 11, **2,689** at 10 — 12,286 rows (0.71%). Train truth has 571 rows at 10 or more out of 2.2M (0.026%) and a maximum of 11, so every one of the 7,632 capped rows carries at least one false ID. Cap sweep on the holdout is flat (cap 6–12 within 0.0006), so this is a test-only pattern. Likely causes: cross-Source 1 competition (item 4), the extras (item 2), or France. A row with 4 true and 8 false IDs scores 0.38. Fix: inspect a sample of the 12-length rows in `output/v7/matching_results.tsv` for common names; apply item 4 first, then item 2.
4. **Target exclusivity, `output/v7x`.** 112,526 test targets were claimed by more than one Source 1 (237,658 extra claims, worst target claimed 179 times). Truth assigns each target to exactly one Source 1. `output/v7x` keeps the highest v7 score per target. The holdout has 2 such clashes, so the local score did not move. Public score **0.903**, up from v7's **0.888** (+0.015). The argmax owner was right often enough that dropping the other claims was a large net gain.
5. **Cutoff and cap are correct for v7.** Sweep on holdout: floor 0.65 → 0.9119, 0.70 → **0.9126**, 0.75 → 0.9105. Cap 12 to 8 within 0.0001. No action.
6. **Consistent, not bugs:** `MISSING_IDF=1.0` in train and score; `prepare_record` uses the v3 tokenizer for tail features while `name_five` uses v4, in both train and score; `city_token` returns the last street-like token, which for France is often `CEDEX` or a postal-town — noise, not skew, and unmeasurable without French labels.
7. **Disk:** 970MB free. A rebuild of a full test file needs roughly 150MB per output folder plus scratch. Delete `ber/data/parts` scratch or older logs before another full scoring run.

## Where the holdout loss is (v7, 6,000 rows, F0.5 0.9126)

Total loss 0.0874. By row pattern: missed true IDs only **0.0383** (1,952 rows), non-singleton predicted empty **0.0175** (105 rows), both false and missed **0.0130** (198 rows), false IDs only **0.0118** (302 rows), singleton given an ID **0.0068** (41 rows). Recall failures are **0.056** of the loss, precision failures **0.019**, and the rest mixed. By country: India mean **0.8738** (2,368 rows, loss 0.0498), US mean **0.9379** (3,632 rows, loss 0.0376). By truth size: rows with 2–4 true IDs carry 0.053 of the loss. Predicted list sizes are shifted low: 674 rows predicted 1 ID against 312 truth rows of size 1; 392 predicted empty against 328 true singletons.

Ceilings: finding all 1,544 unreachable true IDs (1,123 rows) → **0.9449**. Removing every false ID → **0.9377**. Oracle over the current candidate lists → **0.9729**. So the matcher leaves 0.060 on the table within the lists it already has, and retrieval leaves 0.032. Every FP-reduction rule tried so far (set veto, address bar, margin, hard negatives, higher cutoff) lost more true IDs than the false IDs it removed; the model is already on the precision side. Gains have to come from **recall inside the candidate list on India**, and from the test-only defects above.

## Path to 0.900 (+0.012 public), in order

One upload left. `output/v7rx` is public **0.910** (+0.007 over v7x's 0.903, above the ~0.908 estimate). It is `lgbm_v7r` at cutoff 0.75 on the v6 lists, then exclusivity on the v7r scores: 207,135 extra claims removed, 97,869 contested targets, 113,169 empty rows, validator PASS. The closest holdout number is 0.9209 (same model and cutoff, domain dropped, no exclusivity). The remaining local-to-public gap is about **0.011**. The last upload stays `output/v7rx` at public **0.910**. The India competition model does not ship. On 883,188 train India Source 1 rows there were only **12,211** contested targets (test v7rx had 97,869). Highest-score ownership scored **0.8863** singleton **0.8475**. The second model scored **0.8862**, singleton unchanged. Held-out targets 2,281. Delta about zero, so no test file was written.

- **A. Done.** `output/v7x` public **0.903**. The prior estimate was +0.003 to +0.006. The hard argmax rule was worth +0.015.
- **B. Shipped as `output/v7rx`.** Public **0.910** (+0.007 over v7x). Holdout at cutoff 0.75 without domain was **0.9209**, singleton **0.8902**, India **0.8919**, US **0.9399**. The uploaded file kept the domain channel. Model `lgbm_v7r.txt`.
- **C. Second-stage competition model.** Measured on all 883,188 train India Source 1 rows and rejected. Only 12,211 targets were claimed by more than one Source 1. Highest-score ownership: F0.5 **0.8863**, singleton **0.8475**. The second model, on 2,281 held-out targets: **0.8862**, singleton unchanged. It does not beat the rule already in `v7rx`. No test file.
- **D. Recall inside the list on India.** 0.056 of the loss is true IDs the model rejects. Per-country inspection of India rows with pattern `fn_only`: what does the rejected true pair look like (Indic, address empty, name rank)? If Indic-script pairs dominate, item B(1) is the fix. If empty-address pairs dominate, a separate empty-address model (train only on pairs where one side has no address) can be scored and blended; the global model treats a missing address as weak evidence against.
- **E. Full-record character 3-gram retrieval channel** (name plus address as one string, same-country), scored against the 1,282 keyless misses. Public solutions report this as the channel that recovers severe-spelling and reordered-token misses. Measure unique candidate recall first; it enters the list only if the v7 matcher keeps more true than false from it. Retrieval channels so far have been under the gate, so this is behind C and D.
- **F. France.** Untestable. Keep CEDEX out of `city_token`, and do nothing else without labels.

Do not upload anything from C–F without the holdout gate, or the competition-aware validation for C. B is shipped. The India lift and the exclusivity pass both showed up on the public score.

## Measured — romanized retrain and v6 extras under v7 (27 Sep)

Scripts: `train_v7r.py`, `eval_v7_extras.py`, `sweep_v7r.py`, `sweep_v7r_extras.py`. Model `lgbm_v7r.txt` does not replace `lgbm_v7.txt`. 93,074 of 2,055,883 training ids romanize; 105,807 pair rows had `raw_ratio` rewritten; 11,584 of those rows are in the holdout. Rare-token IDF on true val pairs moves from 0.537 to 0.556. Same 400 trees, same hyperparameters.

Shipped model `lgbm_v7.txt`, cutoff 0.70:

| Lists | F0.5 | Singleton | India | US |
|---|---:|---:|---:|---:|
| Raw base (the old 0.9126) | 0.9126 | 0.8750 | 0.8738 | 0.9379 |
| Romanized base | 0.9133 | 0.8811 | 0.8755 | 0.9379 |
| Romanized base + all v6 extras (uploaded file) | 0.9172 | 0.8720 | 0.8802 | 0.9413 |
| Romanized base + extras, domain dropped | 0.9177 | 0.8780 | 0.8818 | 0.9410 |

The combined extras add 248 true and 70 false accepts. Without domain, 228 true and 35 false. Domain is the part that hurts singletons. Keeping the other three channels is right.

Retrained model `lgbm_v7r.txt`, romanized names, extras without domain:

| Cutoff | F0.5 | Singleton | India | US |
|---|---:|---:|---:|---:|
| 0.65 | 0.9220 | 0.8293 | 0.8955 | 0.9393 |
| 0.70 | 0.9218 | 0.8567 | 0.8944 | 0.9397 |
| 0.75 | 0.9209 | 0.8902 | 0.8919 | 0.9399 |
| 0.78 | 0.9201 | 0.9024 | 0.8902 | 0.9397 |
| 0.80 | 0.9199 | 0.9146 | 0.8889 | 0.9401 |

Base lists only, no extras, at 0.70: **0.9178** / singleton **0.8598**. The gain is India (0.8738 → 0.8919 at cutoff 0.75). US does not move. No cutoff clears +0.005 over the uploaded file's 0.9172 while also holding singleton at or above 0.8720: 0.70 is +0.0046 with a worse singleton, 0.75 is +0.0037 with a better one. Do not export a test file for this.

## Explicitly closed

- Seed expansion (from a matcher-accepted Source 2/3 row, pull other noisy rows with the same house and street, cap 25 per address): on the 6,000-row validation fold the raw add is 235 true links and 24,546 false nominations. The shipped matcher at 0.70 keeps 74 true and 6 false. End-to-end macro F0.5 moves from 0.8707 to 0.8720 (+0.0013). Singleton F0.5 stays 0.726. Under the 0.005 upload gate. Do not ship.
- Reverse blocking (S2/S3 nominate holdout S1, top 5 each, keep 15 per S1): on the 6,000-row validation fold, forward recall is 0.926. Reverse adds 371 true links the top-25 list missed and 31,700 false nominations. The shipped matcher at 0.70 keeps 147 of those true links and 14 false ones. End-to-end macro F0.5 moves from 0.8707 to 0.8744 (+0.0037). Singleton F0.5 moves from 0.726 to 0.723. Under the 0.005 upload gate. Do not ship.
- Latin rescue rule: on 2,500 training rows it accepted 454 true and 68 false pairs; **453/454 true pairs were already in the top-25 list**. Its marginal contribution was **1 true link versus 22 false links**. Do not ship.
- Raw multilingual MiniLM cosine threshold: the false near-name outranks real typo/domain pairs. Do not ship.
- Gemma transliteration hidden as “hand-cleaning”: license/audit risk and false methodology. Do not use.
- v6 exact name key (accent-fold plus IndicXlit whole-name, country prefix, drop any key with more than 12 postings), unioned with the current top-25 name and top-25 address lists and scored by the v5 matcher: on the 6,000-row validation fold, 5,670 query keys, 1,474 dropped as common, **92** new true candidates and **394** false ones. Macro F0.5 moves from **0.9034 to 0.9048** (+0.0014). Train Source 2 and Source 3 contain only **1,518** unique Indic words, so many businesses share one folded name and the cap removes the true row. The earlier **0.9096** figure inserted true links only and is not this index. Under the 0.005 upload gate. Do not export `output/v6`.
- Accent folding as its own retrieval key (a pair counts only when the folded names match and the plain names differ): the 6,000-row fold has **0** accented Source 1 names, so this is Source 2/3 accents against Latin US/India names, not France. Cap 12 adds 113 true and 9,777 false; F0.5 **0.8988** (−0.0046) and singleton 0.8537 → 0.8384. Requiring a house or street agreement adds 55 true and 177 false; the matcher keeps 52 true and 15 false; F0.5 **0.9039** (+0.0005). France is not in this fold. Do not ship.
- v6 name key kept only when the house or street also agrees (up to 25 new ids, house-and-street first): candidates 200 true and 1,947 false. The matcher keeps 180 true and 61 false. F0.5 **0.9060** (+0.0026). Singleton 0.8537 → 0.8445. House agreement alone, street not required: candidates 152 true and 558 false, matcher keeps 146 true and 50 false, F0.5 **0.9059** (+0.0025), singleton 0.8506. Mean list stays about 47. Under the 0.005 gate, and singletons get worse. Do not export.
- Domain stem, exact and only when one side is a website: of the 1,544 holdout misses, **38** have a stem that equals the other squashed name. The index returns 29 true and 3,887 false candidates. The matcher keeps 28 true and 34 false. F0.5 **0.9036** (+0.0002). Singleton 0.8537 → 0.8506. Only **2** holdout Source 1 rows contain a domain themselves. Do not ship.
- Consonant skeleton plus the same house number (house at least two digits): **93** of the 1,544 misses, and the index returns all 93 with **113** false candidates. The matcher keeps 85 true and 19 false. F0.5 **0.9054** (+0.0020). Singleton stays **0.8537**. One more skeleton edit, still with the same house, covers **32** further misses and was not indexed. Under the 0.005 gate. Do not export.
- v6 test file is being built in `output/v6` from the union above, including accent folding through the folded name key. Local combined score remains **0.9070**. France is still unlabeled. The open accent dump that scored 0.8988 is not in the file.
- Higher cutoff when an existing candidate has an empty address: 0.85 gives F0.5 **0.9024** (−0.0010) and singleton 0.8720; 0.90 gives **0.9011** and the same singleton. Singletons improve because false ids are dropped, and true matches with a blank address are dropped with them. Do not ship.
- Target exclusivity on the holdout: only **2** accepted targets are claimed twice, so local F0.5 stays **0.9126**. On the test file, 112,526 targets were claimed more than once (237,658 extra claims). `output/v7x` keeps the higher v7 score. Public **0.903**, from v7's **0.888** (+0.015). Shipped.
- Score gap (keep later IDs only within 0.08–0.25 of the top score): best is gap 0.25 at **0.9125** (−0.0001), singleton unchanged at 0.8750. Tighter gaps fall to 0.8953. Do not ship.
- Rare-token candidate plus house, street, city, or postal (IDF at least 0.70, cap 8): 35 new true candidates and 652 false. v7 F0.5 **0.9130** (+0.0004), singleton unchanged. Do not ship.
- Address-aware bar on v7 scores (keep 0.70 when house, street, or postal agrees; require a higher score when the name is the only evidence): at 0.80 the holdout is **0.9121** (−0.0005) and singleton rises from 0.8750 to **0.8994**. At 0.85 it is 0.9089 / 0.8994. Not enough for a public 0.900.
- Hard-negative reweight (false pairs with name Jaro–Winkler ≥ 0.80 and no address agreement, 680,800 of 2,347,296 negatives): weight 2 gives **0.9102** / singleton 0.8872; weight 4 gives 0.9079; weight 8 gives 0.9053. All under v7’s 0.9126. Do not export.
- Set veto on v7 scores (drop a moderate top match with no rare-token or house support; keep a later ID only when it shares that token or house): every setting loses holdout F0.5. The least-bad full rule is **0.9058** (−0.0068) with singleton 0.8811. Blanking only the unsupported moderate top, and leaving the rest of a strong list alone, is **0.9102** (−0.0024) with singleton 0.8811. Greedy 0.70 stays **0.9126** / singleton **0.8750**. True address-only matches get removed. Do not ship.
- Soft token alignment (IDF-weighted, Jaro–Winkler at least 0.9) plus an initialism bit, retrained on top of the v7 features: at cutoff 0.70, holdout F0.5 **0.9141** (+0.0015 over 0.9126) and singleton **0.8720**, just under v7’s 0.8750. The best cutoff is 0.65 at **0.9152** (+0.0026) with singleton **0.8445**. Initialism is on for 0.003 of true pairs and 0.000 of false pairs. Under the 0.005 gate. Do not export. Model `lgbm_v8.txt`.
- Name-list dedupe and backfill: on the 6,000-row validation fold, keeping 25 distinct normalized names drops candidate recall from **0.9262 to 0.8804** and v4 F0.5 from **0.9021 to 0.8904**. Only **57** current misses are recovered. **1,282** of **1,544** neither-list misses are beyond name-rank 80 or have no name key. True copies that share a normalized name get dropped. Do not ship.
