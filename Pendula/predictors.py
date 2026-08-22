"""
Sklapanje prediktora — od sirovih polja do rjecnika normalizovanih slojeva
cija imena tacno odgovaraju kljucevima tezina u species.py.
 
Prostorni prediktori su 2D polja 0..1.
Skalarni (doba dana, mjesecina, mirno more) se racunaju po danu i sire
na cijeli domen; najbolji sati se izvjestavaju posebno, jer korisnik bira
kad ce izaci i besmisleno je gasiti zonu zato sto je podne.
"""
from __future__ import annotations
 
import datetime as dt
import math
 
import numpy as np
 
from . import fields
from .config import MAX_TROLL_DEPTH
from .species import Species
 
# Svi kljucevi tezina koji moraju postojati kao prediktor
REQUIRED = {
    "floating_objects", "sst_front", "depth_band", "bojana_plume",
    "chl_gradient", "current_shear", "forage_index", "calm_sea", "diel",
    "current_edge", "dist_shelf_edge", "canyon_depth",
    "thermocline_depth", "dist_structure", "slope", "turbidity", "surf_zone",
    "uz_obalu",
}
 
 
def _norm(a):
    return fields._norm(a)
 
 
def _near(dist_km: np.ndarray, scale: float) -> np.ndarray:
    """Blizina kao eksponencijalno opadanje: 1 na nuli, ~0.37 na `scale` km."""
    return np.exp(-np.clip(dist_km, 0, None) / scale)
 
 
def _band(x: np.ndarray, lo: float, hi: float, soft: float = 0.35) -> np.ndarray:
    """Glatka pripadnost opsegu [lo, hi] sa mekim ivicama."""
    width = max((hi - lo) * soft, 1e-6)
    return np.clip(np.fmin((x - lo) / width + 1.0, (hi - x) / width + 1.0), 0, 1)
 
 
# ------------------------------------------------------------------ SKALARI
def moon_illumination(date: dt.date) -> float:
    """Udio osvijetljenog diska 0..1 (aproksimacija, dovoljna za skoriranje)."""
    known_new = dt.date(2000, 1, 6)
    days = (date - known_new).days
    phase = (days % 29.53058867) / 29.53058867
    return (1 - math.cos(2 * math.pi * phase)) / 2
 
 
def diel_windows(sunrise: dt.datetime, sunset: dt.datetime) -> list:
    """Najbolji sati: zora i sumrak, po sat i po sa svake strane."""
    return [
        (sunrise - dt.timedelta(minutes=45), sunrise + dt.timedelta(minutes=75)),
        (sunset - dt.timedelta(minutes=75), sunset + dt.timedelta(minutes=45)),
    ]
 
 
def calm_sea_factor(wave_max_m: float) -> float:
    """Mirno more pomaze uocavanju jate i radu varalice."""
    return float(np.clip(1.0 - (wave_max_m - 0.3) / 1.4, 0.15, 1.0))
 
 
# --------------------------------------------------------------- GLAVNI POSAO
def build_predictors(*, sst, sst_lag3, sst_lag7, chl_surf,
                     theta, theta_levels, chl3d, chl_levels,
                     salinity, u, v, mld, static, lats, lons,
                     date, wave_max_m, flotsam, res_deg=0.01) -> dict:
    """
    Vraca (predictors, vertical) gdje je `vertical` rjecnik sa dijagnostikom
    vertikalne strukture koja ide u izlaz za korisnika.
    """
    depth = static["depth"]
 
    # --- termalna struktura
    grad = fields.sst_gradient(sst, lats, res_deg)
    tend3 = fields.sst_tendency(sst, sst_lag3, 3)
    tend7 = fields.sst_tendency(sst, sst_lag7, 7)
 
    # --- vertikala
    # Fizicki i biogeohemijski model imaju razlicite vertikalne mreze,
    # pa svaki koristi svoje nivoe.
    z_thermo, thermo_strength = fields.thermocline(theta, theta_levels, mld=mld)
    z_dcm, dcm_ratio = fields.dcm_depth(chl3d, chl_levels)
    dcm_deb = fields.dcm_thickness(chl3d, chl_levels)
    prey_c, prey_t = fields.prey_layer(mld, z_thermo, z_dcm, dcm_deb)
    vconc = fields.vertical_concentration(thermo_strength, prey_t)
    reach = fields.reachability(prey_c, prey_t, MAX_TROLL_DEPTH)
 
    # Dohvatljivost mnozi koncentraciju: zbijeni plijen na 80 m ne vrijedi nista
    vgate = vconc * np.clip(reach, 0.05, 1.0)
 
    chl_grad = fields.sst_gradient(np.log10(np.clip(chl_surf, 0.01, None)),
                                   lats, res_deg)
    upw = fields.upwelling(sst, chl_surf, lats, res_deg)
 
    # Izdizanje dubinske vode dize produktivnost, pa ulazi kroz mamac.
    # Tezine po vrstama se ne diraju.
    forage = np.clip(fields.forage_index(chl_surf, chl_grad, vgate)
                     + 0.25 * upw, 0, 1)
    plume = fields.bojana_plume(salinity)
    shear = fields.current_shear(u, v, lats, res_deg)
    speed = np.hypot(u, v)
 
    ones = np.ones_like(sst)
 
    # Promjena temperature povrsine — prostorna i vremenska, obje mjerene
    # prema apsolutnom pragu od 0.5 C, ne prema ostatku domena.
    raspon = fields.sst_local_range(sst, lats, res_deg, window_km=3.0)
    prelaz = fields.prelazi_prag(raspon)
    tend_prag = fields.prelazi_prag(tend3 * 3.0)   # ukupna promjena za 3 dana
 
    predictors = {
        # kapije
        "sst": sst,
        "vertical_concentration": vgate,
 
        # termalni — front je sada jaci od 0.5 C na 3 km, a ne "jaci od ostalih"
        "sst_front": np.fmax(prelaz, 0.6 * tend_prag),
        "sst_raspon_C": raspon,
        "sst_warming": fields.prelazi_prag(np.clip(tend3, 0, None) * 3.0),
        "sst_cooling": fields.prelazi_prag(np.clip(-tend3, 0, None) * 3.0),
        "sst_promjena_3d_C": tend3 * 3.0,
        "upwelling": upw,
        "hladnije_od_okoline_C": fields.hladno_jezgro(sst, lats, res_deg),
 
        # produktivnost
        "chl_gradient": _norm(chl_grad),
        "forage_index": forage,
        "turbidity": _norm(np.clip(chl_surf, 0, 5)) * _near(
            static["dist_bojana"], 12.0),
 
        # dinamika
        "current_shear": _norm(shear),
        "current_edge": _norm(shear) * _norm(speed),
 
        # geometrija
        "slope": _norm(static["slope"]),
        "dist_structure": _near(static["dist_structure"], 4.0),
        "dist_shelf_edge": _near(static["dist_shelf_edge"], 6.0),
        "canyon_depth": _band(depth, 400.0, 1400.0),
        "surf_zone": _band(depth, 2.0, 15.0) * _near(static["dist_bojana"], 15.0),
        "bojana_plume": plume,
        # uzak pojas 0.3-1.5 km od obale — tu prolaze male sabljarke
        "uz_obalu": _band(static["dist_coast"], 0.3, 1.5),
 
        # vertikalni
        "thermocline_depth": _band(np.nan_to_num(z_thermo, nan=999.0),
                                   10.0, MAX_TROLL_DEPTH),
 
        # dogadjajni i skalarni
        "floating_objects": float(flotsam) * _near(static["dist_bojana"], 25.0),
        "calm_sea": ones * calm_sea_factor(wave_max_m),
        # Korisnik sam bira kad izlazi, pa doba dana ne gasi zonu; najbolji
        # sati se javljaju uz vrstu. Mjesecina vise nijednoj vrsti ne treba
        # otkad sabljarka nije nocna.
        "diel": ones,
    }
 
    vertical = dict(
        termoklina_m=_med(z_thermo), termoklina_C_po_m=_med(thermo_strength),
        dcm_m=_med(z_dcm), dcm_odnos=_med(dcm_ratio),
        dcm_debljina_m=_med(dcm_deb), mld_m=_med(mld),
        sloj_plijena_m=[_med(prey_c - prey_t / 2), _med(prey_c + prey_t / 2)],
        dohvatljivost=_med(reach), prey_center=prey_c,
    )
    return predictors, vertical
 
 
def depth_band_for(sp: Species, depth: np.ndarray,
                   month: int | None = None) -> np.ndarray:
    """
    `depth_band` je razlicit za svaku vrstu, a kod nekih i za dio sezone:
    strelka i lica su ljeti na otvorenom, u proljece i jesen uz obalu.
    """
    pr = sp.profil(month) if month is not None else dict(depth_pref=sp.depth_pref)
    return _band(depth, *pr["depth_pref"])
 
 
def vertical_note(v: dict) -> str:
    """Objasnjenje vertikalne situacije, sa lancem ishrane koji stoji iza nje."""
    zt, zd, r = v["termoklina_m"], v["dcm_m"], v["dohvatljivost"]
 
    lanac = ""
    if zd is not None:
        deb = v.get("dcm_debljina_m")
        raspon = f" (sloj debljine {deb:.0f} m)" if deb else ""
        lanac = (f" Najveća gustina hlorofila je na {zd:.0f} m{raspon} — tu se "
                 f"skuplja sitna riba koja se hrani planktonom, pa je i najveća "
                 f"vjerovatnoća da su grabljivice tu negdje.")
 
    if zt is None:
        return ("Vodeni stub izmiješan — plijen razvučen po dubini, "
                "koncentracije slabije." + lanac)
    if r is not None and r < 0.35:
        return (f"Termoklina na {zt:.0f} m. Sloj plijena leži pretežno ispod "
                f"{MAX_TROLL_DEPTH:.0f} m, dakle izvan dohvata panule." + lanac)
    return (f"Termoklina na {zt:.0f} m drži plijen stisnut u uzak sloj." + lanac)
 
 
def _med(a):
    if a is None:
        return None
    m = float(np.nanmedian(a)) if np.isfinite(np.nanmedian(a)) else None
    return None if m is None else round(m, 1)
 
 
def validate_coverage() -> list:
    """Provjerava da svaki kljuc tezine iz species.py ima svoj prediktor."""
    from .species import SPECIES
    used = set()
    for sp in SPECIES.values():
        used |= set(sp.weights.keys())
        if sp.ljeto:
            used |= set(sp.ljeto["weights"].keys())
    missing = sorted(used - REQUIRED)
    unused = sorted(REQUIRED - used)
    out = []
    if missing:
        out.append(f"NEDOSTAJU prediktori: {missing}")
    if unused:
        out.append(f"neiskorisceni prediktori: {unused}")
    return out
 
