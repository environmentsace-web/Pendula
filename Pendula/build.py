"""
Orkestrator — jedan dnevni ciklus.
 
    python -m panula.build            # stvarni podaci (traži Copernicus nalog)
    python -m panula.build --sinteticki   # bez mreže, za provjeru logike
 
Izlaz je statički GeoJSON u public/ koji frontend čita bez ikakvog ključa.
"""
from __future__ import annotations
 
import argparse
import datetime as dt
import json
import logging
from pathlib import Path
 
import numpy as np
 
from . import fields, predictors, static, zones
from .config import (FORECAST_DAYS, MAX_TROLL_DEPTH, OUTPUT_DIR, SAFETY,
                     SST_TENDENCY_LAGS)
from .species import SPECIES
 
log = logging.getLogger("panula")
 
 
def run(today: dt.date | None = None, synthetic: bool = False) -> dict:
    today = today or dt.date.today()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
 
    lats, lons = static.analysis_grid()
    stat = static.synthetic() if synthetic else static.build()
 
    days = [today + dt.timedelta(days=i) for i in range(FORECAST_DAYS)]
    sources, written = {}, []
 
    for day in days:
        if synthetic:
            data = _synthetic_day(day, stat, lats, lons)
        else:
            data = _fetch_day(day, today, stat, lats, lons, sources)
 
        # vjetar ide samo u upozorenja, ne u prediktore
        wind_kn = data.pop("wind_kn", 8.0)
        gust_kn = data.pop("gust_kn", 12.0)
 
        preds, vert = predictors.build_predictors(
            lats=lats, lons=lons, static=stat, date=day, **data)
 
        safety = zones.safety_status(
            data["wave_max_m"], wind_kn, gust_kn, SAFETY)
 
        for key, sp in SPECIES.items():
            layers = dict(preds)
            layers["depth_band"] = predictors.depth_band_for(sp, stat["depth"])
 
            score = zones.score_species(sp, day.month, layers)
            feats = zones.extract_zones(score, lats, lons, drivers=layers,
                                        weights=sp.weights)
 
            depth_field = fields.troll_depth_advice(
                sp.troll_depth_m, vert["prey_center"], sp.follows_dcm,
                MAX_TROLL_DEPTH)
            for f in feats:
                f["properties"]["dubina_panule_m"] = _at(
                    depth_field, f["properties"]["centroid"], lats, lons)
 
            fc = {
                "type": "FeatureCollection",
                "properties": {
                    "vrsta": key,
                    "naziv": sp.naziv,
                    "latinski": sp.latinski,
                    "datum": day.isoformat(),
                    "panula": {
                        "dubina_m": [sp.troll_depth_m[0],
                                     min(sp.troll_depth_m[1], MAX_TROLL_DEPTH)],
                        "brzina_kn": list(sp.troll_speed_kn),
                        "granica_dubine_m": MAX_TROLL_DEPTH,
                    },
                    "vertikala": {k: v for k, v in vert.items()
                                  if k != "prey_center"} |
                                 {"objasnjenje": predictors.vertical_note(vert)},
                    "bezbjednost": safety,
                    "napomena": sp.napomena,
                },
                "features": feats,
            }
            path = out / f"{key}_{day.isoformat()}.geojson"
            path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
            written.append(path.name)
 
    index = {
        "generisano": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "domen": {"lon": [float(lons[0]), float(lons[-1])],
                  "lat": [float(lats[0]), float(lats[-1])]},
        "dani": [d.isoformat() for d in days],
        "vrste": list(SPECIES),
        "granica_dubine_panule_m": MAX_TROLL_DEPTH,
        "izvori": sources or {"rezim": "sinteticki test, bez stvarnih podataka"},
        "fajlovi": written,
    }
    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Zapisano %d fajlova u %s", len(written) + 1, out)
    return index
 
 
def _levels(ds) -> np.ndarray:
    for d in ("depth", "deptht", "lev"):
        if d in ds.coords:
            return ds[d].values
    raise KeyError(f"Nema koordinate dubine; postoji: {list(ds.coords)}")
 
 
def _log_res(name, da) -> None:
    names = set(da.dims) | set(da.coords)
    y = "latitude" if "latitude" in names else "lat"
    if y in da.coords and da[y].size > 1:
        step = float(abs(da[y].values[1] - da[y].values[0]))
        log.info("%s: mreza %d tacaka, korak %.4f stepeni (~%.1f km)",
                 name, da[y].size, step, step * 111.32)
 
 
def _at(field: np.ndarray, centroid, lats, lons) -> float:
    j = int(np.argmin(np.abs(lons - centroid[0])))
    i = int(np.argmin(np.abs(lats - centroid[1])))
    v = field[i, j]
    return None if not np.isfinite(v) else round(float(v), 1)
 
 
def _fetch_day(day, today, stat, lats, lons, sources) -> dict:
    """Stvarni podaci. Popunjava `sources` radi izvjestaja o svjezini."""
    from . import fetch
 
    sst_hist = fetch.sst_with_history(today, SST_TENDENCY_LAGS)
    _log_res("SST", sst_hist)
    chl, chl_date = fetch.latest_chlorophyll(today)
    phy = fetch.physics_forecast(today)
    bgc = fetch.bgc_forecast(today)
    wav = fetch.waves_forecast(today)
    met = fetch.meteo(float(np.mean(lats)), float(np.mean(lons)))
    q = fetch.bojana_discharge()
 
    sources.setdefault("sst", {"datum": str(sst_hist.time.values[-1])[:10],
                               "rezolucija_km": 1.1})
    sources.setdefault("hlorofil", {"datum": chl_date.isoformat(),
                                    "senzor": "OLCI 300 m"})
    sources.setdefault("struje", {"rezolucija_km": 4.2})
 
    sel = dict(time=str(day), method="nearest")
 
    def rg(da):
        """Prebacuje na nasu mrezu bez obzira kako se koordinate zovu."""
        names = set(da.dims) | set(da.coords)
        y = "latitude" if "latitude" in names else "lat"
        x = "longitude" if "longitude" in names else "lon"
        return da.interp({y: lats, x: lons}).values
 
    def surf(da):
        """Povrsinski nivo, ako dataset uopste ima dimenziju dubine."""
        for d in ("depth", "deptht", "lev"):
            if d in da.dims:
                return da.isel({d: 0})
        return da
 
    return dict(
        sst=rg(sst_hist.sel(**sel)),
        sst_lag3=rg(sst_hist.sel(time=str(day - dt.timedelta(days=3)),
                                 method="nearest")),
        sst_lag7=rg(sst_hist.sel(time=str(day - dt.timedelta(days=7)),
                                 method="nearest")),
        chl_surf=rg(chl),
        chl3d=rg(bgc["chl3d"]["chl"].sel(**sel)),
        chl_levels=_levels(bgc["chl3d"]),
        theta=rg(phy["temp3d"]["thetao"].sel(**sel)),
        theta_levels=_levels(phy["temp3d"]),
        salinity=rg(surf(phy["sal3d"]["so"].sel(**sel))),
        u=rg(surf(phy["currents"]["uo"].sel(**sel))),
        v=rg(surf(phy["currents"]["vo"].sel(**sel))),
        mld=rg(phy["mld"]["mlotst"].sel(**sel)),
        wave_max_m=float(wav["VHM0"].sel(**sel).max()),
        wind_kn=float(np.nanmax(met["hourly"]["wind_speed_10m"])),
        gust_kn=float(np.nanmax(met["hourly"]["wind_gusts_10m"])),
        flotsam=fetch.flotsam_index(q),
    )
 
 
def _synthetic_day(day, stat, lats, lons) -> dict:
    """Polja za provjeru logike bez mreze."""
    rng = np.random.default_rng(day.toordinal())
    LON, LAT = np.meshgrid(lons, lats)
    depth = stat["depth"]
 
    sst = 23.5 - 2.2 / (1 + np.exp(-(LON - 19.05) * 28)) \
        + rng.normal(0, 0.05, LAT.shape)
    sst[np.isnan(depth)] = np.nan
 
    levels = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200], float)
    theta = np.stack([sst - 7.5 / (1 + np.exp(-(z - 35) / 6.0)) for z in levels])
    chl3d = np.stack([
        (0.18 + 0.8 * np.exp(-((z - 45) ** 2) / 450)) * np.ones_like(sst)
        for z in levels])
 
    sal = 38.3 - 4.0 * np.exp(-stat["dist_bojana"] / 9.0)
 
    return dict(
        sst=sst,
        sst_lag3=sst - 0.35, sst_lag7=sst - 0.75,
        chl_surf=chl3d[0] + 0.6 * np.exp(-stat["dist_bojana"] / 10.0),
        chl3d=chl3d, chl_levels=levels,
        theta=theta, theta_levels=levels, salinity=sal,
        u=rng.normal(0, 0.12, LAT.shape), v=rng.normal(0, 0.12, LAT.shape),
        mld=np.full_like(sst, 22.0),
        wave_max_m=0.7 + 0.5 * (day.toordinal() % 3),
        wind_kn=9.0, gust_kn=14.0, flotsam=0.55,
    )
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinteticki", action="store_true")
    a = ap.parse_args()
    idx = run(synthetic=a.sinteticki)
    print(json.dumps(idx, ensure_ascii=False, indent=2)[:900])
 
