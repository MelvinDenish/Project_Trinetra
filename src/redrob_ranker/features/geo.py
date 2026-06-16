"""Geography modifier.

The JD is Pune/Noida, hybrid, and explicitly "we don't sponsor work visas",
so non-India candidates are a strong negative (and 24,887 of the pool are
non-India — a deliberate trap population). India is the baseline; candidates in
or near the named hubs (Noida/Pune/Hyderabad/Mumbai/Delhi-NCR/Bengaluru) get a
small bonus; non-India candidates willing to relocate are penalized less than
those who aren't.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..textmatch import contains_any, norm


def _is_india(c: dict) -> bool:
    country = norm(c.get("profile", {}).get("country"))
    return "india" in country


def geo_factor(cands: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.empty(len(cands), dtype=np.float32)
    detail: list[dict] = []
    for i, c in enumerate(cands):
        loc = c.get("profile", {}).get("location", "")
        willing = bool(c.get("redrob_signals", {}).get("willing_to_relocate"))
        if _is_india(c):
            in_hub = contains_any(loc, config.PREFERRED_CITIES)
            factor = config.GEO_INDIA * (config.GEO_CITY_BONUS if in_hub else 1.0)
            d = {"is_india": True, "hub": in_hub, "willing_relocate": willing}
        elif willing:
            factor = config.GEO_NONINDIA_RELOCATE
            d = {"is_india": False, "hub": False, "willing_relocate": True}
        else:
            factor = config.GEO_NONINDIA
            d = {"is_india": False, "hub": False, "willing_relocate": False}
        d["factor"] = float(factor)
        d["country"] = c.get("profile", {}).get("country", "")
        out[i] = factor
        detail.append(d)
    return out, detail
