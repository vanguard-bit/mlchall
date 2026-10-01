"""Romanize Indic validation misses with IndicXlit. Writes xlit text next to each pair."""

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

ROOT = Path("/home/loki/projects/mlchall/ber/data/scoreboard/indic_misses.json")


def indic_side(row: dict) -> str:
    for ch in row["s1_name"]:
        if ch.isalpha() and 0x0900 <= ord(ch) <= 0x0DFF:
            return row["s1_name"]
    return row["m_name"]


def main() -> None:
    rows = json.loads(ROOT.read_text(encoding="utf-8"))
    engine = XlitEngine(src_script_type="indic", beam_width=4, rescore=False)
    for i, row in enumerate(rows):
        text = indic_side(row)
        try:
            roman = engine.translit_sentence(text, lang_code=row["lang"])
        except Exception as exc:  # noqa: BLE001
            roman = ""
            row["error"] = str(exc)
        row["xlit"] = roman
        if i and i % 40 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    ROOT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
