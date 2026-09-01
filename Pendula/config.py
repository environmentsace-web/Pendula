"""
Konfiguracija - domen, izvori podataka, pragovi.
Potez Petrovac -> Ada Bojana (more ispred usca Bojane i samo usce, bez rijeke).
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------- DOMEN
# Lustica (rt Ostro) ~42.40N 18.53E ; usce Bojane ~41.85N 19.35E
# Zapadna/juzna granica gura domen preko ivice selfa ka Juznojadranskoj kotlini.
BBOX = dict(lon_min=18.45, lon_max=19.50, lat_min=41.60, lat_max=42.45)

# Analiticka mreza = nativna mreza satelitskog SST-a (0.01 deg ~ 1.1 km lat)
GRID_RES = 0.01

FORECAST_DAYS = 3          # danas + 2

# Panula se ne vuce dublje od ovoga bez obzira na dubinu mora.
# Zona ciji sloj plijena lezi ispod ove granice nije upotrebljiva.
MAX_TROLL_DEPTH = 50.0     # m od povrsine

# Dno dublje od ovoga je predaleko od obale za jednodnevni izlazak.
# Sve preko izlazi iz domena, pa i vrste koje bi ga inace koristile
# (tuna, sabljarka) rade na ivici selfa unutar ove granice.
MAX_DUBINA_M = 250.0
SST_TENDENCY_LAGS = (3, 7)  # dana unazad za dSST/dt

# ------------------------------------------------------- COPERNICUS MARINE
# Pristup preko copernicusmarine toolbox-a; kredencijali iz env varijabli
# COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD
# Identifikatori se NE upisuju rucno - mijenjaju se uz nove verzije produkata.
# Umjesto toga zadajemo produkt i pravila prepoznavanja, pa panula/catalog.py
# nadje tacan dataset u zivom katalogu i ispise sve kandidate.
DATASETS = {
    # Satelitski SST, L4 gap-free, 0.01 stepen, dnevno (CNR-GOS)
    "sst_sat": dict(
        product="SST_MED_SST_L4_NRT_OBSERVATIONS_010_004",
        must=["sst", "l4"], prefer=["0.01", "uhr", "_c_"],
        avoid=["anomaly", "ssta", "climatology"],
        variables=["analysed_sst", "analysis_error"],
    ),
    # Povrsinske struje, 1/24 stepena, 10 dana prognoze
    "currents": dict(
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        must=["cur", "p1d"], prefer=["4.2km"], avoid=["detided", "p1m"],
        variables=["uo", "vo"],
    ),
    # Dubina mijesanog sloja
    "mld": dict(
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        must=["mld", "p1d"], prefer=["4.2km"], avoid=["p1m"],
        variables=["mlotst"],
    ),
    # 3D temperatura -> termoklina
    "temp3d": dict(
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        must=["tem", "p1d"], prefer=["4.2km"], avoid=["p1m", "detided"],
        variables=["thetao", "bottomT"], max_depth=250.0,
    ),
    # 3D salinitet -> haloklina, pluma Bojane
    "sal3d": dict(
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        must=["sal", "p1d"], prefer=["4.2km"], avoid=["p1m"],
        variables=["so"], max_depth=100.0,
    ),
    # Satelitski hlorofil, L3 dnevno (ima rupa od oblaka)
    "chl_sat": dict(
        product="OCEANCOLOUR_MED_BGC_L3_NRT_009_141",
        must=["plankton", "l3"], prefer=["olci", "300m"], avoid=["reflectance"],
        variables=["CHL"],
    ),
    # Rezerva kad je sve pod oblacima: L4 gap-free
    "chl_sat_gapfree": dict(
        product="OCEANCOLOUR_MED_BGC_L4_NRT_009_142",
        must=["plankton", "l4"], prefer=["gapfree", "1km"],
        avoid=["reflectance", "p1m"],
        variables=["CHL"],
    ),
    # 3D hlorofil -> DCM
    "bgc3d": dict(
        product="MEDSEA_ANALYSISFORECAST_BGC_006_014",
        must=["pft", "p1d"], prefer=["4.2km"], avoid=["p1m"],
        variables=["chl"], max_depth=200.0,
    ),
    # Nitrati -> nutriklina
    "nutrients": dict(
        product="MEDSEA_ANALYSISFORECAST_BGC_006_014",
        must=["nut", "p1d"], prefer=["4.2km"], avoid=["p1m"],
        variables=["no3"], max_depth=200.0,
    ),
    # Prozirnost vode - koeficijent slabljenja svjetlosti.
    # Gof lovi okom i trazi bistru vodu.
    "optika": dict(
        product="MEDSEA_ANALYSISFORECAST_BGC_006_014",
        must=["optics", "p1d"], prefer=["4.2km"], avoid=["p1m"],
        variables=["kd490"],
    ),
    # Talasi
    "waves": dict(
        product="MEDSEA_ANALYSISFORECAST_WAV_006_017",
        must=["wav"], prefer=["pt1h", "4.2km"], avoid=["p1m", "p1d"],
        variables=["VHM0", "VTPK", "VMDR"],
    ),
}

# NAPOMENA: identifikatori datasetova se povremeno mijenjaju uz nove verzije
# produkata. Prije prvog pokretanja provjeriti sa:
#   copernicusmarine describe --contains MEDSEA_ANALYSISFORECAST_PHY_006_013

# Meteo bez kljuca (vjetar, pritisak, oblacnost, padavine)
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
# Proticaj Bojane (GloFAS) - za plumu i naplavine
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"
BOJANA_GAUGE = dict(lat=41.87, lon=19.36)

# ------------------------------------------------------------- UPOZORENJA
@dataclass
class SafetyThresholds:
    wave_amber: float = 1.0      # m, znacajna visina talasa
    wave_red: float = 1.5        # m - korisnicki zadat prag
    wind_amber: float = 11.0     # cvorova
    wind_red: float = 15.0       # cvorova
    gust_red: float = 22.0       # cvorova


SAFETY = SafetyThresholds()

# ------------------------------------------------------------- ZONIRANJE
@dataclass
class ZoneParams:
    percentile: float = 85.0      # prag za izdvajanje zone
    min_area_km2: float = 2.0     # manje od ovoga se odbacuje
    max_area_km2: float = 60.0    # veca zona nije savjet nego karta mora
    smooth_sigma: float = 1.5     # gausovo glacanje skora prije konturisanja
    simplify_deg: float = 0.004   # pojednostavljenje poligona
    max_zones_per_species: int = 6


ZONES = ZoneParams()

# ------------------------------------------------------- STATICKI SLOJEVI
# Racunaju se jednom iz EMODnet/GEBCO batimetrije i kesiraju kao NetCDF
STATIC_CACHE = "data/static_layers.nc"
STATIC_VARS = [
    "depth",              # m
    "slope",              # stepeni
    "dist_shelf_edge",    # km do izobate 200 m
    "dist_structure",     # km do najblize seke/olupine/rta
    "dist_bojana",        # km do usca Bojane
    "dist_coast",         # km do obale
]

# Poznate strukture (seke, olupine, rtovi) - dopunjavati iz iskustva
STRUCTURES = [
    dict(name="Platamuni", lat=42.243, lon=18.719),
    dict(name="Katic", lat=42.098, lon=18.930),
    dict(name="Rt Volujica", lat=42.083, lon=19.083),
    dict(name="Stari Ulcinj / Mendra", lat=41.916, lon=19.196),
    dict(name="Usce Bojane", lat=41.852, lon=19.353),
]

OUTPUT_DIR = "public"      # staticki JSON/GeoJSON koji cita Artifact
