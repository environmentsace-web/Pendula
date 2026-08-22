"""
Pronalazenje tacnih identifikatora datasetova u Copernicus katalogu.
 
Identifikatori se mijenjaju uz nove verzije produkata, pa ih ne upisujemo
rucno nego ih trazimo u zivom katalogu po produktu i po kljucnim rijecima.
Rezultat se kesira, a cijela lista kandidata se ispisuje - ako pravilo
promasi, iz ispisa se odmah vidi sta zaista postoji.
"""
from __future__ import annotations
 
import json
import logging
from pathlib import Path
 
log = logging.getLogger(__name__)
CACHE = Path("data/resolved_datasets.json")
 
 
def _get(obj, *names, default=None):
    """Cita polje bez obzira da li je objekat pydantic model ili rjecnik."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return default
 
 
def _catalogue(product_id: str) -> list:
    """Vraca listu (dataset_id, [varijable]) za dati produkt."""
    import copernicusmarine as cm
 
    cat = None
    last = None
    for kwargs in ({"product_id": product_id},
                   {"contains": [product_id]},
                   {}):
        try:
            cat = cm.describe(**kwargs)
            break
        except TypeError as e:
            last = e
            continue
    if cat is None:
        raise RuntimeError(f"Ne mogu da procitam Copernicus katalog: {last}")
 
    out = []
    for prod in _get(cat, "products", default=[]) or []:
        pid = _get(prod, "product_id", "productId", default="")
        if product_id and pid and pid != product_id:
            continue
        for ds in _get(prod, "datasets", default=[]) or []:
            did = _get(ds, "dataset_id", "datasetId", default="")
            if did:
                out.append((did, _variables(ds)))
    return out
 
 
def _variables(ds) -> list:
    names = set()
    for ver in _get(ds, "versions", default=[]) or []:
        for part in _get(ver, "parts", default=[]) or []:
            for svc in _get(part, "services", default=[]) or []:
                for v in _get(svc, "variables", default=[]) or []:
                    n = _get(v, "short_name", "shortName", "standard_name")
                    if n:
                        names.add(str(n))
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
    log.info("Produkt %s - %d datasetova:", product_id, len(candidates))
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
 
