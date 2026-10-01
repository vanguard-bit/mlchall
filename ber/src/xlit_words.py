"""Romanize unique Indic words with IndicXlit. Resumes from the cache file."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

_orig = dataclasses._get_field


def _get_field(cls, a_name, a_type, default_kw_only):
    mod = getattr(cls, "__module__", "")
    if not (mod.startswith("fairseq") or mod.startswith("hydra")):
        return _orig(cls, a_name, a_type, default_kw_only)
    default = getattr(cls, a_name, None)
    target = default.default if isinstance(default, dataclasses.Field) else default
    typ = type(target)
    changed = False
    try:
        if typ.__hash__ is None:
            typ.__hash__ = object.__hash__
            changed = True
    except TypeError:
        pass
    try:
        return _orig(cls, a_name, a_type, default_kw_only)
    finally:
        if changed:
            try:
                typ.__hash__ = None
            except TypeError:
                pass


dataclasses._get_field = _get_field
_load = torch.load


def _load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _load(*args, **kwargs)


torch.load = _load_full

from ai4bharat.transliteration import XlitEngine  # noqa: E402

WORDS = Path("/home/loki/projects/mlchall/ber/data/scoreboard/indic_words.jsonl")
CACHE = Path("/home/loki/projects/mlchall/ber/data/scoreboard/indic_xlit_cache.jsonl")


def load_done() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not CACHE.exists():
        return done
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            done.add((row["w"], row["lang"]))
    return done


def romanize(engine, word: str, lang: str) -> str:
    out = engine.translit_word(word, lang_code=lang, topk=1)
    if isinstance(out, dict):
        values = out.get(word) or next(iter(out.values()), [])
        if isinstance(values, list) and values:
            return str(values[0])
        return str(values)
    if isinstance(out, list) and out:
        return str(out[0])
    return str(out)


def main() -> None:
    done = load_done()
    pending = []
    with WORDS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (row["w"], row["lang"]) not in done:
                pending.append(row)
    print(f"done {len(done)} pending {len(pending)}", flush=True)
    if not pending:
        return
    engine = XlitEngine(src_script_type="indic", beam_width=4, rescore=False)
    with CACHE.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(pending, start=1):
            try:
                roman = romanize(engine, row["w"], row["lang"])
            except Exception as exc:  # noqa: BLE001
                roman = ""
                row["error"] = str(exc)
            row["roman"] = roman
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                handle.flush()
                print(f"  {i}/{len(pending)}", flush=True)
    print(f"xlit finished {len(pending)}", flush=True)


if __name__ == "__main__":
    main()
