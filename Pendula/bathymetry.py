"""
Batimetrija se preuzima automatski - korisnik ne dira nijedan fajl.
 
Izvor: EMODnet Bathymetry WCS (otvoren servis, bez registracije).
Preuzima se jednom i kesira; dno se ne mijenja, pa nema razloga za ponavljanje.
"""
from __future__ import annotations
 
import logging
from pathlib import Path
 
import numpy as np
import requests
 
from .config import BBOX, GRID_RES
 
log = logging.getLogger(__name__)
 
WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE = "emodnet:mean"
CACHE_TIF = Path("data/emodnet_bathymetry.tif")
 
 
def download(force: bool = False) -> Path:
    """Skida isjecak batimetrije za nas domen kao GeoTIFF."""
    if CACHE_TIF.exists() and not force:
        log.info("Batimetrija vec preuzeta: %s (%.1f MB)",
                 CACHE_TIF, CACHE_TIF.stat().st_size / 1e6)
        return CACHE_TIF
 
    CACHE_TIF.parent.mkdir(parents=True, exist_ok=True)
    bbox = (f"{BBOX['lon_min']},{BBOX['lat_min']},"
            f"{BBOX['lon_max']},{BBOX['lat_max']}")
 
    params = {
        "service": "wcs",
        "version": "1.0.0",
        "request": "getcoverage",
        "coverage": COVERAGE,
        "crs": "EPSG:4326",
        "BBOX": bbox,
        "format": "image/tiff",
        "interpolation": "nearest",
        "resx": str(GRID_RES / 2),      # dvostruko finije od analiticke mreze
        "resy": str(GRID_RES / 2),
    }
 
    log.info("Preuzimam batimetriju sa EMODnet-a za domen %s ...", bbox)
    r = requests.get(WCS_URL, params=params, timeout=180)
    r.raise_for_status()
 
    if not r.content[:4] in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(
            "EMODnet nije vratio GeoTIFF nego vjerovatno poruku o gresci:\n"
            + r.text[:400]
        )
 
    CACHE_TIF.write_bytes(r.content)
    log.info("Sacuvano: %s (%.1f MB)", CACHE_TIF, len(r.content) / 1e6)
    return CACHE_TIF
 
 
def to_grid(lats: np.ndarray, lons: np.ndarray, force: bool = False):
    """
    Vraca dubinu u metrima na analitickoj mrezi: pozitivno nadolje,
    NaN nad kopnom.
    """
    import rasterio
    from rasterio.enums import Resampling
 
    path = download(force=force)
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(len(lats), len(lons)),
            resampling=Resampling.bilinear,
        ).astype(float)
        nodata = src.nodata
 
    if nodata is not None:
        data[data == nodata] = np.nan
 
    # EMODnet za dijelove bez podataka zna vratiti vrijednost koja nije
    # prijavljena kao nodata. Bez ove provjere kopno postane "vrlo duboko
    # more" i ulazi u proracun kao stanistite za tunu i sabljarku.
    data[np.abs(data) > 12000] = np.nan
 
    # EMODnet daje elevaciju (more je negativno). Okrecemo u dubinu.
    depth = -data
    depth[depth <= 0] = np.nan
 
    # Najdublja tacka u domenu je oko 1300 m; sve preko toga je greska.
    sumnjivo = depth > 2000
    if sumnjivo.any():
        log.warning("Odbacujem %d celija sa dubinom preko 2000 m - "
                    "vjerovatno oznaka za nedostatak podatka", int(sumnjivo.sum()))
        depth[sumnjivo] = np.nan
 
    # GeoTIFF ide od sjevera nadolje, nasa mreza od juga nagore
    depth = np.flipud(depth)
 
    valid = np.isfinite(depth)
    log.info("Batimetrija na mrezi: %d%% mora, dubine %.0f-%.0f m",
             100 * valid.mean(), np.nanmin(depth), np.nanmax(depth))
    return depth
 
