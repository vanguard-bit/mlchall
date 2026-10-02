"""Collect unique Indic words from train Source 2 and Source 3."""

from __future__ import annotations

import json
from pathlib import Path

from normalize import tokens
from paths import DATA_DIR

OUT = DATA_DIR / "scoreboard" / "indic_words.jsonl"


def lang_of(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "hi"
        if 0x0980 <= o <= 0x09FF:
            return "bn"
        if 0x0A00 <= o <= 0x0A7F:
            return "pa"
        if 0x0A80 <= o <= 0x0AFF:
            return "gu"
        if 0x0B00 <= o <= 0x0B7F:
            return "or"
        if 0x0B80 <= o <= 0x0BFF:
            return "ta"
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
        if 0x0C80 <= o <= 0x0CFF:
            return "kn"
        if 0x0D00 <= o <= 0x0D7F:
            return "ml"
    return ""


def indic(text: str) -> bool:
    return any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF for ch in text)


def main() -> None:
    found: dict[tuple[str, str], None] = {}
    root = DATA_DIR / "train"
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        with (root / filename).open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                _eid, name, _addr, _country = line.rstrip("\n").split("\t")
                if not indic(name):
                    continue
                for tok in tokens(name):
                    if not indic(tok):
                        continue
                    lang = lang_of(tok)
                    if lang:
                        found[(tok, lang)] = None
        print(f"{filename} unique words {len(found)}", flush=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for word, lang in found:
            handle.write(json.dumps({"w": word, "lang": lang}, ensure_ascii=False) + "\n")
    print(f"wrote {len(found)}", flush=True)


if __name__ == "__main__":
    main()
