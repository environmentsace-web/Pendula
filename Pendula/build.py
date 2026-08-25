"""
Orkestrator - jedan dnevni ciklus.
 
    python -m panula.build            # stvarni podaci (trazi Copernicus nalog)
    python -m panula.build --sinteticki   # bez mreze, za provjeru logike
 
Izlaz je staticki GeoJSON u public/ koji frontend cita bez ikakvog kljuca.
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
    out = Path(OUTPUT_DIR).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log.info("Izlaz ide u: %s", out)
 
    lats, lons = static.analysis_grid()
    stat = static.synthetic() if synthetic else static.build()
 
    # Izobate su stalna osnova: prave se jednom i poslije samo stoje.
    # Prepisuju se same cim se promijene pravila po kojima su nacrtane.
    iz_path = out / "izobate.geojson"
    potpis = static.izobate_potpis()
    stari = None
    if iz_path.exists():
        try:
            stari = json.loads(iz_path.read_text()).get("potpis")
        except Exception:
            stari = None
 
    if stari == potpis:
        log.info("Izobate vec odgovaraju pravilima (%s) - ne diram ih.", potpis)
    else:
        if stari:
            log.info("Pravila izobata promijenjena (%s -> %s) - crtam iznova.",
                     stari, potpis)
        iz = static.izobate(stat.get("depth_raw", stat["depth"]), lats, lons,
                            maska=static.koridor(lats, lons),
                            **static.IZOBATE_PARAMS)
        iz["potpis"] = potpis
        iz_path.write_text(json.dumps(iz, ensure_ascii=False), encoding="utf-8")
        log.info("Izobate: %d linija, %.0f kB, potpis %s",
                 len(iz["features"]), iz_path.stat().st_size / 1024, potpis)
 
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
            pr = sp.profil(day.month)
            layers = dict(preds)
            layers["depth_band"] = predictors.depth_band_for(
                sp, stat["depth"], day.month)
 
            score = zones.score_species(sp, day.month, layers)
            feats = zones.extract_zones(score, lats, lons, drivers=layers,
                                        weights=pr["weights"])
 
            # Kad nema nijedne zone, korisniku treba razlog a ne prazan ekran.
            prazno = None if feats else _zasto_prazno(sp, day, layers)
 
            depth_field = fields.troll_depth_advice(
                pr["troll_depth_m"], vert["prey_center"], pr["follows_dcm"],
                MAX_TROLL_DEPTH)
            for f in feats:
                c = f["properties"]["centroid"]
                f["properties"]["dubina_panule_m"] = _at(depth_field, c, lats, lons)
                # Koliko se povrsinska temperatura mijenja u toj zoni -
                # prostorno na 3 km i vremenski za tri dana.
                f["properties"]["sst_raspon_C"] = _at(
                    preds.get("sst_raspon_C"), c, lats, lons)
                f["properties"]["sst_promjena_3d_C"] = _at(
                    preds.get("sst_promjena_3d_C"), c, lats, lons)
                f["properties"]["upwelling"] = _at(preds.get("upwelling"), c,
                                                   lats, lons)
                f["properties"]["hladnije_C"] = _at(
                    preds.get("hladnije_od_okoline_C"), c, lats, lons)
                f["properties"]["izgledi"] = _izgledi(
                    f["properties"]["score_mean"])
 
            fc = {
                "type": "FeatureCollection",
                "properties": {
                    "vrsta": key,
                    "naziv": sp.naziv,
                    "latinski": sp.latinski,
                    "datum": day.isoformat(),
                    "profil": pr["naziv_profila"],
                    "panula": {
                        "dubina_m": [pr["troll_depth_m"][0],
                                     min(pr["troll_depth_m"][1], MAX_TROLL_DEPTH)],
                        "brzina_kn": list(pr["troll_speed_kn"]),
                        "sati": pr.get("sati"),
                        "granica_dubine_m": MAX_TROLL_DEPTH,
                    },
                    "vertikala": {k: v for k, v in vert.items()
                                  if k != "prey_center"} |
                                 {"objasnjenje": predictors.vertical_note(vert)},
                    "bezbjednost": safety,
                    "napomena": pr["napomena"],
                    "razlog_praznog": prazno,
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
        "izobate": "izobate.geojson",
        "fajlovi": written,
    }
    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Zapisano %d fajlova u %s", len(written) + 1, out)
    for f in sorted(out.iterdir())[:5]:
        log.info("   %s  (%d B)", f.name, f.stat().st_size)
    return index
 
 
def _res_km(da) -> float:
    """Stvarni korak mreze u km, umjesto rucno upisane oznake."""
    names = set(da.dims) | set(da.coords)
    y = "latitude" if "latitude" in names else "lat"
    if y in da.coords and da[y].size > 1:
        return round(float(abs(da[y].values[1] - da[y].values[0])) * 111.32, 1)
    return None
 
 
MJESECI = ["", "januar", "februar", "mart", "april", "maj", "jun", "jul",
           "avgust", "septembar", "oktobar", "novembar", "decembar"]
KRATKI = ["", "jan", "feb", "mar", "apr", "maj", "jun", "jul",
          "avg", "sep", "okt", "nov", "dec"]
 
 
def _mjeseci(mjeseci: dict) -> str:
    """Aktivni mjeseci kao citljivi opsezi: 'apr-jun i sep-nov'."""
    aktivni = sorted(m for m, v in mjeseci.items() if v > 0)
    if not aktivni:
        return "nikad"
    grupe, tek = [], [aktivni[0]]
    for m in aktivni[1:]:
        if m == tek[-1] + 1:
            tek.append(m)
        else:
            grupe.append(tek); tek = [m]
    grupe.append(tek)
 
    dijelovi = [KRATKI[g[0]] if len(g) == 1 else f"{KRATKI[g[0]]}-{KRATKI[g[-1]]}"
                for g in grupe]
    return " i ".join(dijelovi) if len(dijelovi) <= 2 else \
           ", ".join(dijelovi[:-1]) + " i " + dijelovi[-1]
 
 
def _zasto_prazno(sp, day, layers) -> str:
    """Koja kapija je ugasila vrstu - sezona ili temperatura mora."""
    sez = zones.season_gate(sp, day.month)
    if sez <= 0:
        svi = dict(sp.months)
        if sp.ljeto:
            svi.update({m: v for m, v in sp.ljeto["months"].items() if v > 0})
        return f"Van sezone - {sp.naziv.lower()} se lovi {_mjeseci(svi)}."
 
    kapija = zones.sst_gate(sp, layers["sst"], day.month)
    srednja = float(np.nanmean(layers["sst"]))
    if float(np.nanmax(kapija)) <= 0.02:
        lo, hi = sp.profil(day.month)["sst_range"]
        if srednja > hi:
            return (f"More je {srednja:.1f}  C, iznad gornje granice od "
                    f"{hi:.0f}  C koju model uzima za ovu vrstu.")
        return (f"More je {srednja:.1f}  C, ispod donje granice od "
                f"{lo:.0f}  C koju model uzima za ovu vrstu.")
 
    return "Nema podrucja koje prelazi prag - uslovi su ujednaceni."
 
 
def _izgledi(skor: float) -> str:
    """
    Rijec uz broj. Skor je relativan u odnosu na domen, pa broj bez opisa
    zavarava: 20/100 je najbolje sto danas ima, ali nije dobro.
    """
    if skor >= 60: return "vrlo dobri"
    if skor >= 40: return "dobri"
    if skor >= 25: return "osrednji"
    if skor >= 12: return "slabi"
    return "vrlo slabi"
 
 
def _meteo_max(met, field: str, default: float) -> float:
    """Najveca vrijednost iz meteo prognoze; podrazumijevana ako je nema."""
    try:
        vals = [v for v in met["hourly"][field] if v is not None]
        return float(max(vals)) if vals else default
    except (TypeError, KeyError):
        return default
 
 
def _opciono(ds, ime, sel, rg, surf):
    """Polje ako postoji; None ako dataset nije stigao ili nema tu varijablu."""
    if ds is None or ime not in getattr(ds, "data_vars", {}):
        log.info("Sloj '%s' nije dostupan - model radi bez njega.", ime)
        return None
    try:
        return rg(surf(ds[ime].sel(**sel)))
    except Exception as e:
        log.warning("Sloj '%s' ne mogu procitati (%s)", ime, e)
        return None
 
 
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
 
 
def _at(field, centroid, lats, lons) -> float:
    if field is None:
        return None
    j = int(np.argmin(np.abs(lons - centroid[0])))
    i = int(np.argmin(np.abs(lats - centroid[1])))
    v = field[i, j]
    return None if not np.isfinite(v) else round(float(v), 1)
 
 
# Podaci se preuzimaju JEDNOM za sva tri dana, ne po danu.
_BUNDLE: dict = {}
 
 
def _fetch_day(day, today, stat, lats, lons, sources) -> dict:
    """Stvarni podaci. Popunjava `sources` radi izvjestaja o svjezini."""
    from . import fetch
 
    if not _BUNDLE:
        log.info("Preuzimam podatke jednom za sva %d dana...", FORECAST_DAYS)
        sst_hist = fetch.sst_with_history(today, SST_TENDENCY_LAGS)
        _log_res("SST", sst_hist)
        chl, chl_date = fetch.latest_chlorophyll(today)
        _BUNDLE.update(
            sst_hist=sst_hist, chl=chl, chl_date=chl_date,
            phy=fetch.physics_forecast(today),
            bgc=fetch.bgc_forecast(today),
            wav=fetch.waves_forecast(today),
            met=fetch.meteo(float(np.mean(lats)), float(np.mean(lons))),
            q=fetch.bojana_discharge(),
        )
        log.info("Preuzimanje zavrseno.")
 
    sst_hist = _BUNDLE["sst_hist"]
    chl, chl_date = _BUNDLE["chl"], _BUNDLE["chl_date"]
    phy, bgc, wav = _BUNDLE["phy"], _BUNDLE["bgc"], _BUNDLE["wav"]
    met, q = _BUNDLE["met"], _BUNDLE["q"]
 
    sources.setdefault("sst", {
        "datum": str(sst_hist.time.values[-1])[:10],
        "rezolucija_km": _res_km(sst_hist)})
    sources.setdefault("hlorofil", {"datum": chl_date.isoformat(),
                                    "senzor": "OLCI 300 m"})
    sources.setdefault("struje", {"rezolucija_km": 4.2})
    sources.setdefault("meteo", {"dostupan": met is not None})
 
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
        bottom_t=_opciono(phy["temp3d"], "bottomT", sel, rg, surf),
        kd490=_opciono(bgc.get("optika"), "kd490", sel, rg, surf),
        salinity=rg(surf(phy["sal3d"]["so"].sel(**sel))),
        u=rg(surf(phy["currents"]["uo"].sel(**sel))),
        v=rg(surf(phy["currents"]["vo"].sel(**sel))),
        mld=rg(phy["mld"]["mlotst"].sel(**sel)),
        wave_max_m=float(wav["VHM0"].sel(**sel).max()),
        wind_kn=_meteo_max(met, "wind_speed_10m", 8.0),
        gust_kn=_meteo_max(met, "wind_gusts_10m", 12.0),
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
        kd490=np.full_like(sst, 0.07) + 0.05 * np.exp(-stat["dist_bojana"] / 8.0),
        bottom_t=np.clip(sst - 0.03 * np.nan_to_num(depth, nan=0.0), 12.0, None),
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
 
