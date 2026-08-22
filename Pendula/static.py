"""
Staticki slojevi — racunaju se jednom iz batimetrije i kesiraju.
 
Batimetrija: EMODnet Bathymetry (bolja uz obalu) ili GEBCO 2024 kao rezerva.
Fajl se skida rucno jednom i stavlja u data/bathymetry.nc — nema smisla
preuzimati ga svaki dan jer se dno ne mijenja.
"""
from __future__ import annotations
 
import logging
from pathlib import Path
 
import numpy as np
from scipy import ndimage
 
from .config import BBOX, GRID_RES, STATIC_CACHE, STRUCTURES
 
log = logging.getLogger(__name__)
EARTH_R = 6371.0
 
 
def analysis_grid():
    """Nativna mreza analize — ista kao satelitski SST (0.01 stepen)."""
    lats = np.arange(BBOX["lat_min"], BBOX["lat_max"], GRID_RES)
    lons = np.arange(BBOX["lon_min"], BBOX["lon_max"], GRID_RES)
    return lats, lons
 
 
def load_bathymetry():
    """
    Dubina na analitickoj mrezi. Preuzima se automatski sa EMODnet-a
    prvi put, poslije se cita iz kesa. Korisnik ne dira nijedan fajl.
    """
    from . import bathymetry
    lats, lons = analysis_grid()
    return bathymetry.to_grid(lats, lons)
 
 
def koridor(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Ogranicava domen na potez Lustica -> Ada Bojana.
 
    Pravougaoni box mora biti sirok da uhvati duboku vodu prema jugozapadu,
    ali time zahvata i albansku obalu ispred Ljezhe. Zato sijecemo:
      - linija kroz vrh Lustice  -> sve sjeverozapadno od nje otpada
      - linija kroz usce Bojane  -> sve jugoistocno od nje otpada
      - unutrasnjost Boke        -> zaliv, nije teren za panulu
    Obje linije idu pod 225 stepeni, sto prati pruzanje obale.
    Budva, Platamuni i Katic ostaju u domenu; dubina prema jugozapadu takodje.
    """
    LON, LAT = np.meshgrid(lons, lats)
 
    def strana(lat0, lon0):
        dx = (LON - lon0) * 111.32 * np.cos(np.deg2rad(LAT))
        dy = (LAT - lat0) * 111.32
        return dx - dy          # >0 jugoistocno od linije, <0 sjeverozapadno
 
    lustica = strana(42.400, 18.530)      # rt Ostro, ulaz u Boku
    bojana = strana(41.852, 19.353)
    boka = (LAT > 42.38) & (LON > 18.60)  # unutrasnjost zaliva
    return (lustica > 0) & (bojana < 0) & ~boka
 
 
def _cell_km(lats):
    dy = GRID_RES * np.pi / 180.0 * EARTH_R
    dx = dy * np.cos(np.deg2rad(lats))
    return dx, dy
 
 
def slope_deg(depth: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Nagib dna u stepenima."""
    dx_km, dy_km = _cell_km(lats)
    filled = np.where(np.isnan(depth), np.nanmax(depth), depth)
    gy = np.gradient(filled, axis=0) / (dy_km * 1000.0)
    gx = np.gradient(filled, axis=1) / (dx_km[:, None] * 1000.0)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    s[np.isnan(depth)] = np.nan
    return s
 
 
def distance_to_isobath(depth: np.ndarray, lats: np.ndarray,
                        target: float = 200.0) -> np.ndarray:
    """
    Rastojanje u km do zadate izobate. Za tunu je izobata 200 m
    (ivica selfa) najjaci pojedinacni prediktor.
    """
    band = np.abs(depth - target) < (target * 0.08)
    if band.sum() == 0:
        return np.full_like(depth, np.nan)
    dx_km, dy_km = _cell_km(lats)
    dist_cells = ndimage.distance_transform_edt(~band)
    return dist_cells * float(np.mean(dx_km) + dy_km) / 2.0
 
 
def distance_to_points(points: list, lats: np.ndarray,
                       lons: np.ndarray) -> np.ndarray:
    """Rastojanje u km do najblize tacke iz liste (seke, olupine, rtovi)."""
    LON, LAT = np.meshgrid(lons, lats)
    best = np.full(LAT.shape, np.inf)
    for p in points:
        dlat = (LAT - p["lat"]) * 111.32
        dlon = (LON - p["lon"]) * 111.32 * np.cos(np.deg2rad(LAT))
        best = np.minimum(best, np.hypot(dlat, dlon))
    return best
 
 
def distance_to_coast(depth: np.ndarray, lats: np.ndarray) -> np.ndarray:
    land = np.isnan(depth)
    dx_km, dy_km = _cell_km(lats)
    d = ndimage.distance_transform_edt(~land)
    out = d * float(np.mean(dx_km) + dy_km) / 2.0
    out[land] = np.nan
    return out
 
 
def build() -> dict:
    """Racuna sve staticke slojeve i kesira ih."""
    import xarray as xr
 
    lats, lons = analysis_grid()
    cache = Path(STATIC_CACHE)
    if cache.exists():
        ds = xr.open_dataset(cache)
        return {v: ds[v].values for v in ds.data_vars}
 
    depth = load_bathymetry()
    depth[~koridor(lats, lons)] = np.nan
    layers = {
        "depth": depth,
        "slope": slope_deg(depth, lats),
        "dist_shelf_edge": distance_to_isobath(depth, lats, 200.0),
        "dist_structure": distance_to_points(STRUCTURES, lats, lons),
        "dist_bojana": distance_to_points(
            [s for s in STRUCTURES if "Bojan" in s["name"]], lats, lons),
        "dist_coast": distance_to_coast(depth, lats),
    }
 
    cache.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {k: (("lat", "lon"), v) for k, v in layers.items()},
        coords={"lat": lats, "lon": lons},
    ).to_netcdf(cache)
    log.info("Staticki slojevi kesirani u %s", cache)
    return layers
 
 
def synthetic(seed: int = 0) -> dict:
    """
    Sinteticki staticki slojevi za testiranje bez batimetrijskog fajla.
    Grubo oponasa stvarnu geometriju: plitko na SI, strmi pad ka JZ.
    """
    lats, lons = analysis_grid()
    LON, LAT = np.meshgrid(lons, lats)
 
    depth = np.clip(
        2600 * ((42.30 - LAT) * 0.35 + (19.50 - LON) * 1.25) - 250, 2, 1400)
    depth[(LAT > 42.16) & (LON < 18.86)] = np.nan     # kopno SZ
    depth[(LAT > 42.02) & (LON > 19.30)] = np.nan     # kopno IZ
    depth[~koridor(lats, lons)] = np.nan              # van poteza
 
    return {
        "depth": depth,
        "slope": slope_deg(depth, lats),
        "dist_shelf_edge": distance_to_isobath(depth, lats, 200.0),
        "dist_structure": distance_to_points(STRUCTURES, lats, lons),
        "dist_bojana": distance_to_points(
            [s for s in STRUCTURES if "Bojan" in s["name"]], lats, lons),
        "dist_coast": distance_to_coast(depth, lats),
    }
 
