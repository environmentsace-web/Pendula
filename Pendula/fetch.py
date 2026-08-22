"""
Preuzimanje podataka. Zahtijeva mrezu i Copernicus nalog:
  export COPERNICUSMARINE_SERVICE_USERNAME=...
  export COPERNICUSMARINE_SERVICE_PASSWORD=...
 
Sve funkcije kesiraju na disk. Ako preuzimanje padne, koristi se posljednje
uspjesno polje i to se oznacava u izlazu (polje `zastarjelo`), da korisnik
nikad ne dobije star podatak predstavljen kao svjez.
"""
from __future__ import annotations
 
import datetime as dt
import logging
from pathlib import Path
 
import numpy as np
import requests
import xarray as xr
 
from . import catalog
from .config import (BBOX, DATASETS, OPEN_METEO_FORECAST, OPEN_METEO_FLOOD,
                     BOJANA_GAUGE, FORECAST_DAYS)
 
log = logging.getLogger(__name__)
CACHE = Path("data/cache")
CACHE.mkdir(parents=True, exist_ok=True)
 
 
def _subset(key: str, start: dt.date, end: dt.date, **kw) -> xr.Dataset:
    """Tanak omotac oko copernicusmarine.subset sa kesiranjem."""
    import copernicusmarine as cm
 
    spec = DATASETS[key]
    dataset_id = catalog.resolve(
        key, spec["product"], spec["must"],
        spec.get("prefer", ()), spec.get("avoid", ()))
 
    # Prvi nivo dubine nije 0 m nego oko 1 m, pa ne zadajemo donju granicu.
    # Gornju saljemo samo za 3D slojeve; povrsinski se biraju kasnije.
    if spec.get("max_depth"):
        kw.setdefault("maximum_depth", spec["max_depth"])
 
    out = CACHE / f"{key}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    if not out.exists():
        cm.subset(
            dataset_id=dataset_id,
            variables=spec["variables"],
            minimum_longitude=BBOX["lon_min"], maximum_longitude=BBOX["lon_max"],
            minimum_latitude=BBOX["lat_min"], maximum_latitude=BBOX["lat_max"],
            start_datetime=f"{start}T00:00:00",
            end_datetime=f"{end}T23:59:59",
            output_filename=str(out),
            overwrite=True,
            **kw,
        )
    return xr.open_dataset(out)
 
 
def _subset_safe(key, start, end, **kw):
    """Kao _subset, ali ako imena varijabli ne odgovaraju, skida sve."""
    try:
        return _subset(key, start, end, **kw)
    except Exception as e:
        msg = str(e).lower()
        if "depth" in msg:
            log.warning("Dubina ne odgovara (%s) - pokusavam bez nje", e)
            kw.pop("maximum_depth", None)
            saved = DATASETS[key].pop("max_depth", None)
            try:
                return _subset(key, start, end, **kw)
            finally:
                if saved is not None:
                    DATASETS[key]["max_depth"] = saved
        if "variable" not in msg:
            raise
        log.warning("Imena varijabli ne odgovaraju (%s) - skidam sve", e)
        spec = DATASETS[key]
        saved = spec["variables"]
        spec["variables"] = None
        try:
            ds = _subset(key, start, end, **kw)
        finally:
            spec["variables"] = saved
        log.info("Dostupne varijable u '%s': %s", key, list(ds.data_vars))
        return ds
 
 
def pick(ds, *candidates):
    """Prva varijabla iz dataseta koja postoji medju ponudjenim imenima."""
    for c in candidates:
        if c in ds.data_vars:
            return ds[c]
    lower = {v.lower(): v for v in ds.data_vars}
    for c in candidates:
        if c.lower() in lower:
            return ds[lower[c.lower()]]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise KeyError(f"Nema nijedne od {candidates}; postoji: {list(ds.data_vars)}")
 
 
def latest_chlorophyll(today: dt.date, max_lookback: int = 7,
                       min_coverage: float = 0.35) -> tuple[xr.DataArray, dt.date]:
    """
    Hlorofil objavljen na najskorijem datumu - trazi unazad dok ne nadje
    snimak sa dovoljnim pokrivanjem domena (oblaci prave rupe u L3 OLCI).
    Ako ni jedan dan ne prodje prag, pada na L4 gap-free 1 km.
    """
    for back in range(max_lookback):
        day = today - dt.timedelta(days=back)
        try:
            ds = _subset_safe("chl_sat", day, day)
        except Exception as e:
            log.warning("OLCI %s nedostupan: %s", day, e)
            continue
        chl = ds["CHL"].isel(time=0)
        coverage = float(np.isfinite(chl).mean())
        if coverage >= min_coverage:
            log.info("Hlorofil OLCI 300 m, %s, pokrivanje %.0f%%",
                     day, coverage * 100)
            return chl, day
        log.info("OLCI %s pokrivanje samo %.0f%% - trazim dalje",
                 day, coverage * 100)
 
    log.warning("Prelazim na L4 gap-free (interpolirano)")
    for back in range(max_lookback):
        day = today - dt.timedelta(days=back)
        try:
            ds = _subset_safe("chl_sat_gapfree", day, day)
            return ds["CHL"].isel(time=0), day
        except Exception:
            continue
    raise RuntimeError("Nema dostupnog hlorofila u posljednjih 7 dana")
 
 
def sst_with_history(today: dt.date, lags=(3, 7)):
    """Satelitski SST za danas i za zadate pomake unazad (za tendenciju)."""
    start = today - dt.timedelta(days=max(lags) + 2)
    ds = _subset_safe("sst_sat", start, today)
    sst = pick(ds, "analysed_sst", "adjusted_sea_surface_temperature", "sst")
    if sst.attrs.get("units", "").lower() in ("kelvin", "k"):
        sst = sst - 273.15
    return sst
 
 
def physics_forecast(today: dt.date):
    """Struje, MLD, 3D temperatura i salinitet - danas + prognoza."""
    end = today + dt.timedelta(days=FORECAST_DAYS)
    return {
        "currents": _subset_safe("currents", today, end),
        "mld": _subset_safe("mld", today, end),
        "temp3d": _subset_safe("temp3d", today, end),
        "sal3d": _subset_safe("sal3d", today, end),
    }
 
 
def bgc_forecast(today: dt.date):
    """3D hlorofil i nitrati -> DCM i nutriklina."""
    end = today + dt.timedelta(days=FORECAST_DAYS)
    return {
        "chl3d": _subset_safe("bgc3d", today, end),
        "no3": _subset_safe("nutrients", today, end),
    }
 
 
def waves_forecast(today: dt.date):
    end = today + dt.timedelta(days=FORECAST_DAYS)
    return _subset_safe("waves", today, end)
 
 
def _get_json(url: str, params: dict, tries: int = 3,
              timeout: int = 60) -> dict | None:
    """HTTP upit sa ponovnim pokusajima. Vraca None ako sve propadne."""
    import time
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 5 * (i + 1)
            log.warning("Pokusaj %d/%d za %s nije uspio (%s)",
                        i + 1, tries, url.split("/")[2], e)
            if i < tries - 1:
                time.sleep(wait)
    return None
 
 
def meteo(lat: float, lon: float) -> dict | None:
    """Vjetar, udari, pritisak, padavine - Open-Meteo, bez kljuca."""
    out = _get_json(OPEN_METEO_FORECAST, {
        "latitude": lat, "longitude": lon,
        "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                  "surface_pressure,precipitation,cloud_cover",
        "daily": "sunrise,sunset",
        "wind_speed_unit": "kn",
        "forecast_days": FORECAST_DAYS,
        "timezone": "Europe/Podgorica",
    })
    if out is None:
        log.warning("Meteo nedostupan - upozorenja se racunaju samo iz talasa")
    return out
 
 
def bojana_discharge() -> dict:
    """
    Proticaj Bojane (GloFAS). Kljucno za dvije stvari: jacinu bocatne plume
    (strelka, lica) i izbacivanje drvenih naplavina 2-5 dana nakon kisa
    (lampuga).
    """
    return _get_json(OPEN_METEO_FLOOD, {
        "latitude": BOJANA_GAUGE["lat"], "longitude": BOJANA_GAUGE["lon"],
        "daily": "river_discharge",
        "past_days": 7,
        "forecast_days": FORECAST_DAYS,
    })
 
 
def flotsam_index(discharge: dict | None) -> float:
    """
    Indeks naplavina 0..1: odnos maksimalnog proticaja u prozoru od 2-5 dana
    unazad prema tekucem. Vrhunac naplavina kasni za vrhuncem proticaja.
    """
    if not discharge:
        return 0.0
    q = discharge.get("daily", {}).get("river_discharge", [])
    if len(q) < 8:
        return 0.0
    window = [v for v in q[2:6] if v is not None]
    baseline = np.nanmedian([v for v in q if v is not None])
    if not window or not baseline:
        return 0.0
    return float(np.clip((max(window) / baseline - 1.0) / 2.0, 0, 1))
 
