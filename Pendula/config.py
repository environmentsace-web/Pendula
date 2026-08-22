"""
Konfiguracija — domen, izvori podataka, pragovi.
Potez Petrovac -> Ada Bojana (more ispred usca Bojane i samo usce, bez rijeke).
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------- DOMEN
# Petrovac ~42.205N 18.945E ; usce Bojane ~41.85N 19.35E
# Zapadna/juzna granica gura domen preko ivice selfa ka Juznojadranskoj kotlini.
BBOX = dict(lon_min=18.60, lon_max=19.50, lat_min=41.60, lat_max=42.30)

# Analiticka mreza = nativna mreza satelitskog SST-a (0.01 deg ~ 1.1 km lat)
GRID_RES = 0.01

FORECAST_DAYS = 3          # danas + 2

# Panula se ne vuce dublje od ovoga bez obzira na dubinu mora.
# Zona ciji sloj plijena lezi ispod ove granice nije upotrebljiva.
MAX_TROLL_DEPTH = 50.0     # m od povrsine
SST_TENDENCY_LAGS = (3, 7)  # dana unazad za dSST/dt

# ------------------------------------------------------- COPERNICUS MARINE
# Pristup preko copernicusmarine toolbox-a; kredencijali iz env varijabli
# COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD
DATASETS = {
    # Satelitski SST, L4 gap-free, 0.01 deg, dnevno (CNR-GOS)
    "sst_sat": dict(
        dataset_id="cmems_obs-sst_med_phy_nrt_l4_P1D-m",
        product="SST_MED_SST_L4_NRT_OBSERVATIONS_010_004",
        variables=["analysed_sst"],
    ),
    # Povrsinske i 3D struje, 1/24 deg, 10 dana prognoze
    "currents": dict(
        dataset_id="cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        variables=["uo", "vo"],
    ),
    # Dubina mijesanog sloja
    "mld": dict(
        dataset_id="cmems_mod_med_phy-mld_anfc_4.2km_P1D-m",
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        variables=["mlotst"],
    ),
    # 3D temperatura -> termoklina
    "temp3d": dict(
        dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        variables=["thetao"],
        max_depth=250.0,
    ),
    # 3D salinitet -> haloklina, pluma Bojane
    "sal3d": dict(
        dataset_id="cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
        product="MEDSEA_ANALYSISFORECAST_PHY_006_013",
        variables=["so"],
        max_depth=100.0,
    ),
    # Satelitski hlorofil, OLCI 300 m, L3 dnevno (ima rupa od oblaka)
    "chl_sat": dict(
        dataset_id="cmems_obs-oc_med_bgc-plankton_nrt_l3-olci-300m_P1D",
        product="OCEANCOLOUR_MED_BGC_L3_NRT_009_141",
        variables=["CHL"],
    ),
    # Rezerva kad je OLCI pod oblacima: L4 gap-free 1 km
    "chl_sat_gapfree": dict(
        dataset_id="cmems_obs-oc_med_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
        product="OCEANCOLOUR_MED_BGC_L4_NRT_009_142",
        variables=["CHL"],
    ),
    # 3D hlorofil + nitrati -> DCM i nutriklina, 10 dana prognoze
    "bgc3d": dict(
        dataset_id="cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m",
        product="MEDSEA_ANALYSISFORECAST_BGC_006_014",
        variables=["chl"],
        max_depth=200.0,
    ),
    "nutrients": dict(
        dataset_id="cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m",
        product="MEDSEA_ANALYSISFORECAST_BGC_006_014",
        variables=["no3"],
        max_depth=200.0,
    ),
    # Talasi, 1/24 deg, satno
    "waves": dict(
        dataset_id="cmems_mod_med_wav_anfc_4.2km_PT1H-i",
        product="MEDSEA_ANALYSISFORECAST_WAV_006_017",
        variables=["VHM0", "VTPK", "VMDR"],
    ),
}

# NAPOMENA: identifikatori datasetova se povremeno mijenjaju uz nove verzije
# produkata. Prije prvog pokretanja provjeriti sa:
#   copernicusmarine describe --contains MEDSEA_ANALYSISFORECAST_PHY_006_013

# Meteo bez kljuca (vjetar, pritisak, oblacnost, padavine)
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
# Proticaj Bojane (GloFAS) — za plumu i naplavine
OPEN_METEO_FLOOD = "https://flood-api.open-meteo.com/v1/flood"
BOJANA_GAUGE = dict(lat=41.87, lon=19.36)

# ------------------------------------------------------------- UPOZORENJA
@dataclass
class SafetyThresholds:
    wave_amber: float = 1.0      # m, znacajna visina talasa
    wave_red: float = 1.5        # m — korisnicki zadat prag
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

# Poznate strukture (seke, olupine, rtovi) — dopunjavati iz iskustva
STRUCTURES = [
    dict(name="Katic", lat=42.098, lon=18.930),
    dict(name="Platamuni", lat=42.243, lon=18.719),
    dict(name="Rt Volujica", lat=42.083, lon=19.083),
    dict(name="Stari Ulcinj / Mendra", lat=41.916, lon=19.196),
    dict(name="Usce Bojane", lat=41.852, lon=19.353),
]

OUTPUT_DIR = "public"      # staticki JSON/GeoJSON koji cita Artifact
