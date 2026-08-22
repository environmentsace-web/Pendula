"""
Matrica vrsta: sezonske i temperaturne kapije, tezine prediktora,
vertikalne preferencije (dubina panule).

Skor(vrsta, celija, dan) = Sez(mjesec) * Temp(SST) * Vert * SUM(w_i * f_i)

Sez, Temp, Vert su kapije u opsegu 0..1.
Tezine w_i sumiraju na 1.0 unutar svake vrste.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Species:
    key: str
    naziv: str
    latinski: str
    months: Dict[int, float]           # mjesec -> mnozilac 0..1
    sst_range: Tuple[float, float]     # optimum
    sst_tolerance: float               # koliko C izvan optimuma do nule
    depth_pref: Tuple[float, float]    # povoljan raspon dubine dna (m)
    weights: Dict[str, float]
    troll_depth_m: Tuple[float, float]  # preporuceni raspon dubine panule
    follows_dcm: bool                   # da li prati dubinski maksimum hlorofila
    troll_speed_kn: Tuple[float, float]
    napomena: str = ""


def _m(**kwargs) -> Dict[int, float]:
    """Mjeseci navedeni kao m9=1.0; nenavedeni su 0."""
    out = {i: 0.0 for i in range(1, 13)}
    for k, v in kwargs.items():
        out[int(k[1:])] = v
    return out


SPECIES: Dict[str, Species] = {
    "lampuga": Species(
        key="lampuga", naziv="Lampuga", latinski="Coryphaena hippurus",
        months=_m(m8=0.4, m9=0.9, m10=1.0, m11=0.7, m12=0.2),
        sst_range=(22.0, 27.0), sst_tolerance=3.0, depth_pref=(50.0, 200.0),
        weights={
            "floating_objects": 0.30,   # naplavine iz Bojane, mreze, plutajuci otpad
            "sst_front": 0.20,
            "depth_band": 0.15,         # 50-200 m
            "bojana_plume": 0.15,
            "chl_gradient": 0.10,
            "current_shear": 0.10,
        },
        troll_depth_m=(0.0, 15.0), follows_dcm=False,
        troll_speed_kn=(5.0, 7.0),
        napomena="2-5 dana nakon jakih kisa Bojana izbacuje drvene naplavine — "
                 "to je predvidljiv generator pozicija.",
    ),
    "trupac": Species(
        key="trupac", naziv="Trupac", latinski="Auxis rochei",
        months=_m(m7=0.6, m8=1.0, m9=1.0, m10=0.8, m11=0.4),
        sst_range=(20.0, 26.0), sst_tolerance=3.0, depth_pref=(50.0, 400.0),
        weights={
            "forage_index": 0.30,
            "sst_front": 0.20,
            "depth_band": 0.15,         # >50 m
            "calm_sea": 0.15,
            "diel": 0.10,
            "current_shear": 0.10,
        },
        troll_depth_m=(0.0, 30.0), follows_dcm=True,
        troll_speed_kn=(4.0, 6.0),
    ),
    "plamida": Species(
        key="plamida", naziv="Plamida", latinski="Sarda sarda",
        months=_m(m9=0.6, m10=1.0, m11=1.0, m12=0.6),
        sst_range=(16.0, 22.0), sst_tolerance=3.0, depth_pref=(20.0, 100.0),
        weights={
            "forage_index": 0.30,
            "current_edge": 0.25,       # ivica struje, rtovi
            "sst_front": 0.20,
            "depth_band": 0.15,         # 20-100 m
            "diel": 0.10,
        },
        troll_depth_m=(0.0, 25.0), follows_dcm=True,
        troll_speed_kn=(4.0, 6.5),
    ),
    "tuna": Species(
        key="tuna", naziv="Tuna", latinski="Thunnus thynnus",
        months=_m(m5=0.4, m6=0.5, m7=0.6, m8=0.7, m9=1.0, m10=1.0, m11=0.8, m12=0.4),
        sst_range=(18.0, 26.0), sst_tolerance=4.0, depth_pref=(200.0, 1200.0),
        weights={
            "dist_shelf_edge": 0.30,
            "depth_band": 0.20,         # >200 m
            "sst_front": 0.20,
            "chl_gradient": 0.15,
            "forage_index": 0.15,
        },
        troll_depth_m=(0.0, 50.0), follows_dcm=True,
        troll_speed_kn=(6.0, 8.0),
    ),
    "sabljarka": Species(
        key="sabljarka", naziv="Sabljarka", latinski="Xiphias gladius",
        months=_m(m6=0.5, m7=0.8, m8=1.0, m9=1.0, m10=0.8, m11=0.5),
        sst_range=(19.0, 27.0), sst_tolerance=4.0, depth_pref=(400.0, 1200.0),
        weights={
            "canyon_depth": 0.30,       # >400 m
            "night": 0.20,
            "moon": 0.20,
            "thermocline_depth": 0.15,
            "chl_gradient": 0.15,
        },
        troll_depth_m=(15.0, 50.0), follows_dcm=True,
        troll_speed_kn=(1.5, 3.0),
        napomena="Nocni lov, spora panula. Ogranicenje 50 m znaci da se lovi samo kad se sabljarka nocu podigne u povrsinski sloj.",
    ),
    "gof": Species(
        key="gof", naziv="Gof", latinski="Seriola dumerili",
        months=_m(m6=0.7, m7=0.9, m8=1.0, m9=1.0, m10=0.9, m11=0.5),
        sst_range=(19.0, 26.0), sst_tolerance=3.0, depth_pref=(20.0, 80.0),
        weights={
            "dist_structure": 0.40,
            "slope": 0.20,
            "depth_band": 0.15,         # 20-80 m
            "current_edge": 0.15,
            "forage_index": 0.10,
        },
        troll_depth_m=(15.0, 50.0), follows_dcm=False,
        troll_speed_kn=(2.5, 4.5),
        napomena="Prakticno samo SZ dio poteza (Petrovac-Bar); mutna voda obara skor.",
    ),
    "strelka": Species(
        key="strelka", naziv="Strelka", latinski="Pomatomus saltatrix",
        months=_m(m4=0.6, m5=0.7, m6=0.5, m8=0.6, m9=0.8, m10=1.0, m11=0.8),
        sst_range=(15.0, 24.0), sst_tolerance=3.0, depth_pref=(2.0, 20.0),
        weights={
            "bojana_plume": 0.35,       # bocatni gradijent
            "turbidity": 0.20,
            "diel": 0.20,
            "forage_index": 0.15,
            "depth_band": 0.10,         # <20 m
        },
        troll_depth_m=(0.0, 10.0), follows_dcm=False,
        troll_speed_kn=(3.0, 5.0),
        napomena="Domen modela je more ispred usca i samo usce — ne i rijeka.",
    ),
    "lica": Species(
        key="lica", naziv="Lica", latinski="Lichia amia",
        months=_m(m4=0.7, m5=0.9, m6=0.6, m9=0.9, m10=1.0, m11=0.6),
        sst_range=(16.0, 25.0), sst_tolerance=3.0, depth_pref=(2.0, 15.0),
        weights={
            "surf_zone": 0.35,          # plitko pjeskovito + usce
            "turbidity": 0.25,
            "diel": 0.20,
            "forage_index": 0.20,
        },
        troll_depth_m=(0.0, 8.0), follows_dcm=False,
        troll_speed_kn=(3.0, 5.0),
        napomena="Prakticno samo JI dio (Velika plaza - Ada).",
    ),
}


def validate() -> List[str]:
    """Provjera da tezine sumiraju na 1 i da su mjeseci u opsegu."""
    problems = []
    for k, s in SPECIES.items():
        total = sum(s.weights.values())
        if abs(total - 1.0) > 1e-6:
            problems.append(f"{k}: suma tezina = {total:.3f}, ocekivano 1.000")
        if not any(v > 0 for v in s.months.values()):
            problems.append(f"{k}: nijedan mjesec nije aktivan")
    return problems


if __name__ == "__main__":
    issues = validate()
    print("OK" if not issues else "\n".join(issues))
    for k, s in SPECIES.items():
        aktivni = [m for m, v in s.months.items() if v > 0]
        print(f"{s.naziv:10s} {s.latinski:24s} mjeseci {aktivni} "
              f"panula {s.troll_depth_m[0]:.0f}-{s.troll_depth_m[1]:.0f} m")
