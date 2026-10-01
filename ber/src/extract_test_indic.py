"""Append Indic words from test Source 2 and Source 3 that are not cached yet."""

from __future__ import annotations

import json

from paths import DATA_DIR
from v6_keys import indic_token, lang_of
from normalize import tokens

WORDS = DATA_DIR / "scoreboard" / "indic_words.jsonl"
CACHE = DATA_DIR / "scoreboard" / "indic_xlit_cache.jsonl"
TEST = DATA_DIR / "test"


def main() -> None:
    have: set[tuple[str, str]] = set()
    if CACHE.exists():
        with CACHE.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                have.add((row["w"], row["lang"]))
    if WORDS.exists():
        with WORDS.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                have.add((row["w"], row["lang"]))
    new: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for filename in ("test_source2.tsv", "test_source3.tsv"):
        with (TEST / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                name = line.rstrip("\n").split("\t", 2)[1]
                if not indic_token(name):
                    continue
                for tok in tokens(name):
                    if not indic_token(tok):
                        continue
                    lang = lang_of(tok)
                    pair = (tok, lang)
                    if not lang or pair in have or pair in seen:
                        continue
                    seen.add(pair)
                    new.append(pair)
        print(f"{filename} new words {len(new)}", flush=True)
    with WORDS.open("a", encoding="utf-8") as handle:
        for word, lang in new:
            handle.write(json.dumps({"w": word, "lang": lang}, ensure_ascii=False) + "\n")
    print(f"appended {len(new)}", flush=True)


if __name__ == "__main__":
    main()
