"""
Pronalazenje tacnih identifikatora datasetova u Copernicus katalogu.
 
Identifikatori se mijenjaju uz nove verzije produkata, pa ih ne upisujemo
rucno nego ih trazimo u zivom katalogu po produktu i po kljucnim rijecima.
Rezultat se kesira, a cijela lista kandidata se ispisuje — ako pravilo
promasi, iz ispisa se odmah vidi sta zaista postoji.
"""
from __future__ import annotations
 
import json
import logging
from pathlib import Path
 
log = logging.getLogger(__name__)
CACHE = Path("data/resolved_datasets.json")
 
 
def _catalogue(product_id: str) -> list:
    """Vraca listu (dataset_id, [varijable]) za dati produkt."""
    import copernicusmarine as cm
 
    obj = None
    for kwargs in ({"product_id": product_id},
                   {"contains": [product_id]},
                   {}):
        try:
            obj = cm.describe(**kwargs)
            break
        except TypeError:
            continue
    if obj is None:
        raise RuntimeError("Ne mogu da procitam Copernicus katalog")
 
    data = obj if isinstance(obj, dict) else getattr(obj, "__dict__", {}) or {}
    if not data:
        try:
            data = json.loads(obj.model_dump_json())   # pydantic v2
        except Exception:
            data = {}
 
    out = []
    for prod in data.get("products", []):
        pid = prod.get("product_id") or prod.get("productId") or ""
        if product_id and pid != product_id:
            continue
        for ds in prod.get("datasets", []):
            did = ds.get("dataset_id") or ds.get("datasetId") or ""
            if did:
                out.append((did, _variables(ds)))
    return out
 
 
def _variables(ds: dict) -> list:
    names = set()
    for ver in ds.get("versions", []):
        for part in ver.get("parts", []):
            for svc in part.get("services", []):
                for v in svc.get("variables", []):
                    n = v.get("short_name") or v.get("standard_name")
                    if n:
                        names.add(n)
    return sorted(names)
 
 
def resolve(key: str, product_id: str, must: list, prefer: list = (),
            avoid: list = ()) -> str:
    """
    Bira dataset iz produkta: mora sadrzati sve iz `must`, poeni za `prefer`,
    kazna za `avoid`. Ispisuje sve kandidate radi provjere.
    """
    cache = _load_cache()
    if key in cache:
        return cache[key]
 
    candidates = _catalogue(product_id)
    log.info("Produkt %s — %d datasetova:", product_id, len(candidates))
    for did, vars_ in candidates:
        log.info("    %s   %s", did, ",".join(vars_[:6]))
 
    def score(did: str) -> float | None:
        low = did.lower()
        if not all(m.lower() in low for m in must):
            return None
        s = sum(1.0 for p in prefer if p.lower() in low)
        s -= sum(2.0 for a in avoid if a.lower() in low)
        return s
 
    scored = [(score(d), d) for d, _ in candidates]
    scored = [(s, d) for s, d in scored if s is not None]
    if not scored:
        raise RuntimeError(
            f"Za '{key}' nijedan dataset u {product_id} ne sadrzi {must}. "
            f"Dostupni: {[d for d, _ in candidates]}"
        )
 
    scored.sort(key=lambda t: (-t[0], len(t[1])))
    chosen = scored[0][1]
    log.info("  -> za '%s' biram: %s", key, chosen)
 
    cache[key] = chosen
    _save_cache(cache)
    return chosen
 
 
def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}
 
 
def _save_cache(d: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, indent=2))
 
