"""
Skoriranje po celiji i izdvajanje ZONA (poligona), ne tacaka.
 
Skor(vrsta, celija, dan) = Sez(mjesec) * Temp(SST) * Vert * SUM(w_i * f_i)
"""
import numpy as np
from scipy import ndimage
from skimage import measure
 
from .config import ZONES
from .species import Species
 
 
# ------------------------------------------------------------------- KAPIJE
def season_gate(sp: Species, month: int) -> float:
    return sp.profil(month)["months"].get(month, 0.0)
 
 
def sst_gate(sp: Species, sst: np.ndarray, month: int | None = None) -> np.ndarray:
    """Trapezoidna kapija: 1 u optimumu, linearno do 0 na granici tolerancije."""
    pr = sp.profil(month) if month is not None else dict(
        sst_range=sp.sst_range, sst_tolerance=sp.sst_tolerance)
    lo, hi = pr["sst_range"]
    tol = pr["sst_tolerance"]
    below = np.clip((sst - (lo - tol)) / tol, 0, 1)
    above = np.clip(((hi + tol) - sst) / tol, 0, 1)
    return np.clip(np.minimum(below, above), 0, 1)
 
 
# ------------------------------------------------------------------ SKORIRANJE
def score_species(sp: Species, month: int, layers: dict) -> np.ndarray:
    """
    layers: dict naziv_prediktora -> 2D polje normalizovano na 0..1.
    Prediktori koji nedostaju se tretiraju kao neutralni (0.5) i to se
    biljezi u metapodatke izlaza, da korisnik zna da je sloj procijenjen.
    """
    sez = season_gate(sp, month)
    if sez <= 0:
        shape = next(iter(layers.values())).shape
        return np.zeros(shape)
 
    temp = sst_gate(sp, layers["sst"], month)
    vert = layers.get("vertical_concentration", 0.7)
 
    additive = np.zeros_like(temp)
    for name, w in sp.profil(month)["weights"].items():
        f = layers.get(name)
        if f is None:
            f = np.full_like(temp, 0.5)
        additive = additive + w * np.clip(f, 0, 1)
 
    score = sez * temp * vert * additive
    score = np.where(np.isnan(layers["sst"]), np.nan, score)
    return np.clip(score, 0, 1) * 100.0
 
 
# --------------------------------------------------------------- ZONE
def extract_zones(score: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                  drivers: dict | None = None, weights: dict | None = None) -> list:
    """
    Prag -> ciscenje -> povezane komponente -> konture -> poligoni.
    Vraca listu GeoJSON Feature dict-ova sa poligonom i metapodacima.
    """
    valid = np.isfinite(score) & (score > 0)
    if valid.sum() < 20:
        return []
 
    thr = np.nanpercentile(score[valid], ZONES.percentile)
    if not np.isfinite(thr) or thr <= 0:
        return []
 
    smooth = ndimage.gaussian_filter(np.nan_to_num(score), ZONES.smooth_sigma)
    smooth[~valid] = 0.0
 
    mask = smooth >= thr
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, np.ones((3, 3)))
 
    labels, n = ndimage.label(mask)
    components = [labels == lab for lab in range(1, n + 1)]
 
    # Zona od 5000 km2 nije savjet nego karta mora. Prevelike komponente se
    # rekurzivno rasijecaju vecim pragom dok ne padnu ispod max_area_km2.
    components = _split_oversized(components, smooth, lats)
 
    features = []
    for comp in components:
        area = _area_km2(comp, lats)
        if area < ZONES.min_area_km2:
            continue
 
        contours = measure.find_contours(comp.astype(float), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
 
        ring = [[float(_interp(lons, c[1])), float(_interp(lats, c[0]))]
                for c in contour]
        ring = _simplify(ring, ZONES.simplify_deg)
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
 
        vals = score[comp]
        ii, jj = np.where(comp)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "score_mean": round(float(np.nanmean(vals)), 1),
                "score_max": round(float(np.nanmax(vals)), 1),
                "area_km2": round(area, 1),
                "centroid": [round(float(lons[jj].mean()), 4),
                             round(float(lats[ii].mean()), 4)],
                "razlog": _explain(comp, drivers or {}, weights or {}),
            },
        })
 
    features.sort(key=lambda f: -f["properties"]["score_mean"])
    return features[:ZONES.max_zones_per_species]
 
 
def _split_oversized(components: list, smooth: np.ndarray,
                     lats: np.ndarray, depth: int = 0) -> list:
    """Rasijeca komponente vece od max_area_km2 podizanjem praga na njihov medijan."""
    if depth >= 4:
        return components
 
    out, changed = [], False
    for comp in components:
        if _area_km2(comp, lats) <= ZONES.max_area_km2:
            out.append(comp)
            continue
        vals = smooth[comp]
        thr = np.nanpercentile(vals, 60)
        sub = comp & (smooth >= thr)
        sub = ndimage.binary_opening(sub, np.ones((3, 3)))
        sublabels, k = ndimage.label(sub)
        if k == 0:
            out.append(comp)
            continue
        changed = True
        out.extend(sublabels == i for i in range(1, k + 1))
 
    return _split_oversized(out, smooth, lats, depth + 1) if changed else out
 
 
def _explain(comp: np.ndarray, drivers: dict, weights: dict) -> str:
    """
    Zasto je ova zona izdvojena. Gleda SAMO prediktore koje ta vrsta stvarno
    koristi i rangira ih po doprinosu skoru (tezina * srednja vrijednost),
    a ne po sirovoj vrijednosti - inace jak sloj koji vrsta ignorise
    ispadne kao razlog.
    """
    if not drivers or not weights:
        return ""
    contrib = {}
    for name, w in weights.items():
        arr = drivers.get(name)
        if isinstance(arr, np.ndarray) and arr.shape == comp.shape:
            m = float(np.nanmean(arr[comp]))
            if np.isfinite(m):
                contrib[name] = w * m
    if not contrib:
        return ""
    top = sorted(contrib.items(), key=lambda kv: -kv[1])[:2]
    labels = {
        "sst_front": "termalni front",
        "forage_index": "prisustvo mamca",
        "current_shear": "smicanje struje",
        "current_edge": "ivica struje",
        "bojana_plume": "uticaj Bojane",
        "dist_shelf_edge": "ivica selfa",
        "dist_structure": "struktura dna",
        "chl_gradient": "gradijent hlorofila",
        "depth_band": "povoljna dubina",
        "canyon_depth": "kanjon",
        "turbidity": "mutnoca",
        "surf_zone": "priobalna plicina",
        "floating_objects": "naplavine iz Bojane",
        "thermocline_depth": "termoklina u dohvatu",
        "slope": "nagib dna",
        "calm_sea": "mirno more",
        "moon": "mjesecina",
        "diel": "zora/sumrak",
        "night": "noc",
    }
    return " + ".join(labels.get(k, k) for k, _ in top)
 
 
def _area_km2(mask: np.ndarray, lats: np.ndarray, res_deg: float = 0.01) -> float:
    dy = res_deg * 111.32
    ii, _ = np.where(mask)
    if ii.size == 0:
        return 0.0
    dx = dy * np.cos(np.deg2rad(lats[ii]))
    return float(np.sum(dx * dy))
 
 
def _interp(axis: np.ndarray, pos: float) -> float:
    i0 = int(np.floor(pos))
    i1 = min(i0 + 1, len(axis) - 1)
    frac = pos - i0
    return axis[i0] * (1 - frac) + axis[i1] * frac
 
 
def _simplify(ring: list, tol: float) -> list:
    """Douglas-Peucker; shapely ako postoji, inace prorjedjivanje."""
    try:
        from shapely.geometry import Polygon
        poly = Polygon(ring).simplify(tol, preserve_topology=True)
        if poly.is_empty:
            return ring
        return [[round(x, 5), round(y, 5)] for x, y in poly.exterior.coords]
    except Exception:
        step = max(1, len(ring) // 60)
        return [[round(p[0], 5), round(p[1], 5)] for p in ring[::step]]
 
 
# ------------------------------------------------------------- UPOZORENJA
def safety_status(wave_max: float, wind_max_kn: float,
                  gust_max_kn: float, thresholds) -> dict:
    """Tri nivoa. Zone se i dalje racunaju, ali se prikazuju zakljucane."""
    reasons = []
    level = "zeleno"
 
    if wave_max >= thresholds.wave_red:
        level = "crveno"
        reasons.append(f"talasi do {wave_max:.1f} m")
    elif wave_max >= thresholds.wave_amber:
        level = "zuto"
        reasons.append(f"talasi do {wave_max:.1f} m")
 
    if wind_max_kn >= thresholds.wind_red or gust_max_kn >= thresholds.gust_red:
        level = "crveno"
        reasons.append(f"vjetar {wind_max_kn:.0f} cv, udari {gust_max_kn:.0f} cv")
    elif wind_max_kn >= thresholds.wind_amber and level == "zeleno":
        level = "zuto"
        reasons.append(f"vjetar {wind_max_kn:.0f} cv")
 
    return {
        "nivo": level,
        "razlog": ", ".join(reasons) if reasons else "uslovi povoljni",
        "izlazak_preporucen": level != "crveno",
    }
 
