# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We use same-country lexical blocking: an inverted index over name and address keys, plus four narrow extra channels. A LightGBM pair classifier scores the candidates, and a score cutoff tuned for holdout macro F0.5 picks each Source 1 entity's match list. A final exclusivity pass gives every Source 2/3 record to at most one Source 1 entity. The three changes with the largest measured lift were parsed house and street fields, rare-token IDF and city features, and target exclusivity, which alone added +0.015 public. The final file scores **0.910** on the public leaderboard.

---

## 2. Methodology

### 2.1 Problem Analysis

- Train has 2.2M Source 1 entities. Source 2 and Source 3 together hold 10.3M records. Truth lists average about 3.5 IDs per Source 1 entity.
- Names vary by legal suffix, abbreviation, word order, typos, website-style names (`sparkskrantzinsurance.com`), and Indic script in Source 2/3 (Hindi, Punjabi) against Latin-script Source 1 names.
- Addresses vary by abbreviation, dropped PIN or state, landmark text, and house numbers that are truncated or zero-padded. The house number plus street is often the strongest independent evidence.
- In truth, each Source 2/3 record belongs to exactly one Source 1 entity. A first test file claimed 112,526 targets for more than one Source 1 entity.
- France is about 15% of test and has no labelled rows, so nothing is tuned per country. Accent folding and French name particles (`le`, `la`, `de`, ...) appear only as alternate features.
- The metric counts singletons, and F0.5 weights precision, so a wrong merge on a singleton costs a full 1.0.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier + set decoder + exclusivity post-process.  
**Core Innovation:** Parsed house and street agreement, rare-token IDF, and city features feed a gradient-boosted pair model. Indic names are romanized once per unique word with IndicXlit, and the same romanized text is used in training and at test time. A one-owner-per-target rule keeps the highest-scoring Source 1 claim on each contested record.

All decisions were gated on a fixed 6,000-entity Source 1 holdout. A change shipped only if end-to-end macro F0.5 improved by at least 0.005 and singleton F0.5 did not drop. That holdout score counts true IDs that blocking lost and also counts singletons.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Retrieval runs within the same country only.
  - **Name channel:** exact normalized phrase, content tokens, sorted-token key, character 4-grams of the squashed name, and website domain stems.
  - **Address channel:** street tokens, postal codes (5 or more digits), and house numbers.
  - **Ranking and limits:** Keys are IDF-weighted. Postings are capped at 2,500 per key, with at most 14 name keys and 12 address keys per record. The top 25 by name score and the top 25 by address score are unioned.
  - **Extra channels:**
    - Accent-folded plus IndicXlit-romanized exact name key. Common keys are kept only when the house or a street token also agrees.
    - Consonant skeleton plus the same house number.
    - Exact website stem.
- **Candidate pairs generated:** 83,938,629 for 1,732,544 test Source 1 entities, a mean of 48.4 per entity.
- **How you ensured true matches were not lost:** Each channel was measured by unique recall on the holdout. Only true matches missing from the existing list counted, and each new false candidate was charged against the channel. Name-plus-address recall at 25+25 is 0.926 (19,370 of 20,914 true links). The remaining misses are mostly Indic-script names, severe misspellings, and records with no shared key. A channel was kept only when the matcher accepted more true links from it than false ones.

---

## 4. Matching Model

**Features used:**
- **Name features:**
  - Jaro-Winkler, Levenshtein ratio, token Jaccard, sorted-token equality, and character 3-gram cosine.
  - The same similarities on accent-folded and squashed names.
  - A French-particle-stripped alternate name.
  - Maximum IDF of a shared token and of a one-sided token (rare-token agreement and conflict), with IDF fitted on train Source 1 names.
- **Address features:**
  - Jaro-Winkler, ratio, token Jaccard, and 3-gram cosine.
  - Postal equality and postal conflict.
  - Parsed house-number flags: same, missing, conflict when the street agrees but the house does not, and prefix or truncation.
  - Street-token Jaccard.
  - City token Jaro-Winkler and equality.
  - Empty-address flags.
- **Other:**
  - Name and address retrieval ranks and block scores, and which channel produced the candidate.
  - Country equality. There is no country one-hot.
  - Indic Source 2/3 names pass through IndicXlit (`indicxlit-indic-en-v1.0`, MIT, about 11M parameters, the only pretrained model). Romanization runs on the 1,518 unique Indic words and is cached. The same romanized text is shown to the model in training and at test time.

**Model type:** LightGBM binary classifier (learning rate 0.05, 63 leaves, up to 400 trees with early stopping). It trains on blocker-generated pairs for 48,000 train Source 1 entities, about 2.5M pairs.  
**Threshold selection method:** For each Source 1 entity, candidates are sorted by model score. Every ID scoring at least 0.75 is kept, up to 12 IDs. The cutoff was swept from 0.65 to 0.80 on holdout end-to-end macro F0.5, with singleton F0.5 tracked separately (Appendix B). Set-aware alternatives scored lower on the holdout and were not used: score-gap rules, set vetoes, and evidence-class bars. Exclusivity runs after decoding: when several Source 1 entities claim the same record, only the highest model score keeps it. This removed 207,135 extra claims across 97,869 contested test targets.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Public leaderboard: 0.910.
  - Holdout: 0.9209 (6,000 Source 1 entities, same matcher and cutoff, before exclusivity). The holdout contains almost no contested targets.
  - Holdout by segment: singleton 0.890, India 0.892, US 0.940.
- **Common false positives (wrong merges):**
  - Near-identical names of different businesses with no address support, such as `Infrastructure` / `Infratech`.
  - One noisy record claimed by several similar Source 1 entities. Exclusivity fixes this.
  - Singletons given a single plausible match.

  Every rule-based precision filter we tried removed more true links than false ones: set vetoes, address-strength bars, score-gap cuts, and hard-negative reweighting.
- **Common false negatives (missed matches):**
  - On the holdout, recall failures are about 0.056 of the total F0.5 loss, while precision failures are about 0.019.
  - Predicted lists are shorter than true lists for entities with 2 to 4 true copies.
  - Most of the gap is on India.
  - About 7% of true links never reach the candidate list: Indic script, heavy typos, reordered tokens, or no shared key.

---

## 6. Conclusion

A lexical blocker with a small number of measured, gated channels, plus a well-featured LightGBM matcher, took us from 0.807 to 0.910 public. Accepting a change only when the holdout end-to-end score rose, singletons included, kept the upload budget for real lifts. The two largest late gains came from data structure rather than model capacity: each record has one owner, and Indic names must be romanized the same way in training and at test time.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains:

- `src/`: all source code.
- `data/scoreboard/`: the three LightGBM models and the IndicXlit cache.
- `README.md`: the exact 11-step reproduction.
- `requirements.txt`: the main environment.
- `requirements-xlit.txt`: the IndicXlit environment.

Entry points, in order:

1. `build_scoreboard.py`
2. `sweep_decoder.py`
3. `extract_indic_words.py`, `extract_test_indic.py`, `xlit_words.py`
4. `export_fast.py`, then `score_v2.py`: test blocking and country shards.
5. `extend_train.py`, `merge_train.py`, `train_v5.py`
6. `score_v6.py`: extra channels; writes `candidate_pairs.tsv`.
7. `train_v7r.py`
8. `score_v7.py` with `SCORE_MODEL=lgbm_v7r.txt SCORE_MIN_GAIN=0.75`
9. `apply_exclusive.py`: writes the final `matching_results.tsv`.

With the shipped models, only steps 4, 6, 8, and 9 are needed.

### B. Additional Results

| Version | Change | Holdout F0.5 | Public |
| --- | --- | --- | --- |
| baseline | name + address blocking, small LightGBM | — | 0.807 |
| v5 | house/street fields, 48k training entities, French particles | 0.9034 | 0.878 |
| v7 | rare-token IDF and city features on v6 candidate lists | 0.9126 | 0.888 |
| v7x | v7 + target exclusivity | 0.9126 | 0.903 |
| v7rx | matcher retrained on romanized names, cutoff 0.75, exclusivity | 0.9209 | **0.910** |

The holdout contains almost no contested targets, so exclusivity does not change the holdout score; the v7x gain shows up only on public. Holdout F0.5 at each decoder cutoff (v7r):

| Cutoff | Macro | Singleton | India | US |
| --- | --- | --- | --- | --- |
| 0.65 | 0.9220 | 0.8293 | 0.8955 | 0.9393 |
| 0.70 | 0.9218 | 0.8567 | 0.8944 | 0.9397 |
| 0.75 | 0.9209 | 0.8902 | 0.8919 | 0.9399 |
| 0.80 | 0.9199 | 0.9146 | 0.8889 | 0.9401 |

We chose 0.75 because it gave up 0.001 macro F0.5 for +0.034 singleton F0.5, and 15% of test is unlabelled France.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
