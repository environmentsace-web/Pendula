"""
Izvedeni slojevi iz sirovih polja.
 
Dvije razlicite stvari koje se cesto brkaju:
  - PROSTORNI gradijent |grad SST|  -> fronte (gdje se voda mijenja po prostoru)
  - VREMENSKA tendencija dSST/dt    -> zone zagrijavanja / hladjenja
Obje ulaze u model, sa razlicitim znacenjem.
"""
import numpy as np
from scipy import ndimage
 
EARTH_R = 6371.0  # km
 
 
# ------------------------------------------------------------------ GEOMETRIJA
def cell_size_km(lat: np.ndarray, res_deg: float):
    """Velicina celije u km za dati niz geografskih sirina."""
    dy = res_deg * np.pi / 180.0 * EARTH_R
    dx = dy * np.cos(np.deg2rad(lat))
    return dx, dy
 
 
# ---------------------------------------------------------------- SST FRONTE
def sst_gradient(sst: np.ndarray, lat: np.ndarray, res_deg: float) -> np.ndarray:
    """
    Magnituda prostornog gradijenta SST-a u C/km (Sobel).
    sst: 2D polje [lat, lon] sa NaN nad kopnom i pod oblacima.
    """
    filled = np.where(np.isnan(sst), np.nanmean(sst), sst)
    gy = ndimage.sobel(filled, axis=0) / 8.0
    gx = ndimage.sobel(filled, axis=1) / 8.0
 
    dx_km, dy_km = cell_size_km(lat, res_deg)
    gy = gy / dy_km
    gx = gx / dx_km[:, None]
 
    grad = np.hypot(gx, gy)
    grad[np.isnan(sst)] = np.nan
    return grad
 
 
SST_PRAG = 0.5   # stepeni C — korisnicki zadat prag "znacajne promjene"
 
 
def sst_local_range(sst: np.ndarray, lat: np.ndarray, res_deg: float,
                    window_km: float = 3.0) -> np.ndarray:
    """
    Najveca razlika temperature povrsine unutar okoline zadatog poluprecnika.
 
    Ovo je ono sto se na moru zaista vidi: koliko se voda promijeni kad
    predjes par kilometara. Prostorni gradijent u C/km je isto to izrazeno
    po jedinici duzine, ali je ovako uporedivo sa pragom od 0.5 C.
    """
    dx_km, dy_km = cell_size_km(lat, res_deg)
    k = max(1, int(round(window_km / float(np.mean(dx_km)))))
    size = 2 * k + 1
 
    filled = np.where(np.isnan(sst), np.nanmean(sst), sst)
    hi = ndimage.maximum_filter(filled, size=size)
    lo = ndimage.minimum_filter(filled, size=size)
    raspon = hi - lo
    raspon[np.isnan(sst)] = np.nan
    return raspon
 
 
def prelazi_prag(promjena: np.ndarray, prag: float = SST_PRAG) -> np.ndarray:
    """
    Skala vezana za APSOLUTNI prag umjesto za percentile domena.
 
    0 pri nultoj promjeni, 1 na zadatom pragu, blago raste i iznad njega.
    Bez ovoga bi "najjaci front u domenu" uvijek dobijao punu ocjenu, pa i
    kad je cijelo more ujednaceno.
    """
    x = np.abs(promjena) / prag
    return np.clip(np.where(x <= 1.0, x, 1.0 + 0.15 * np.log1p(x - 1.0)), 0, 1.3) / 1.3
 
 
def front_persistence(grad_stack: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Udio dana u kojima je gradijent iznad praga (C/km).
    grad_stack: [dan, lat, lon]. Front prisutan vise dana je stvarna struktura,
    jednodnevni je najcesce artefakt oblaka ili interpolacije.
    """
    hits = (grad_stack > threshold).astype(float)
    hits[np.isnan(grad_stack)] = np.nan
    return np.nanmean(hits, axis=0)
 
 
def sst_tendency(sst_now: np.ndarray, sst_past: np.ndarray, days: int) -> np.ndarray:
    """
    dSST/dt u C/dan. Pozitivno = zona zagrijavanja, negativno = hladjenja.
    """
    return (sst_now - sst_past) / float(days)
 
 
# ------------------------------------------------------------- VERTIKALNI MODUL
def thermocline(theta: np.ndarray, depth: np.ndarray, od_dubine: float = 10.0):
    """
    Iz profila potencijalne temperature izvlaci dubinu i jacinu termokline.
 
    theta: [depth, lat, lon]; depth: [depth] u metrima (rastuce).
    Vraca (dubina_m, jacina_C_po_m) — dubina maksimalnog vertikalnog gradijenta.
 
    Prvih `od_dubine` metara se preskace. Ljeti se povrsina danju zagrije pa
    najjaci gradijent u profilu zna ispasti na dva-tri metra, sto je dnevna
    pojava koja nestane preko noci — a ne sezonska termoklina koja drzi plijen.
    """
    dtheta = np.diff(theta, axis=0)
    ddepth = np.diff(depth)[:, None, None]
    grad = np.abs(dtheta / ddepth)                      # C/m po sloju
 
    mid_depth = 0.5 * (depth[:-1] + depth[1:])
    grad = np.where((mid_depth < od_dubine)[:, None, None], np.nan, grad)
 
    idx = np.nanargmax(np.nan_to_num(grad, nan=-1), axis=0)
 
    zt = mid_depth[idx]
    strength = np.take_along_axis(grad, idx[None, :, :], axis=0)[0]
 
    # Gdje nema stratifikacije (izmijesan stub) termoklina nema smisla
    weak = strength < 0.02                              # C/m
    zt = np.where(weak, np.nan, zt)
    strength = np.where(weak, 0.0, strength)
    return zt, strength
 
 
def dcm_depth(chl: np.ndarray, depth: np.ndarray, min_ratio: float = 1.3):
    """
    Dubina dubinskog maksimuma hlorofila (DCM).
 
    Vraca (dubina_m, jacina) gdje je jacina odnos maksimuma prema povrsinskoj
    vrijednosti. Ako maksimum nije bar min_ratio puta veci od povrsinskog,
    DCM ne postoji kao izdvojena struktura -> NaN (izmijesan, jesenji rezim).
    """
    idx = np.nanargmax(np.nan_to_num(chl, nan=-1), axis=0)
    zmax = depth[idx]
    cmax = np.take_along_axis(chl, idx[None, :, :], axis=0)[0]
    csurf = chl[0]
 
    ratio = np.where(csurf > 0, cmax / csurf, np.nan)
    no_dcm = (ratio < min_ratio) | np.isnan(ratio)
    return np.where(no_dcm, np.nan, zmax), np.where(no_dcm, 1.0, ratio)
 
 
def dcm_thickness(chl: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Debljina dubinskog maksimuma hlorofila, mjerena kao sirina na pola visine.
 
    Ranije je ovo bila pretpostavka (dvostruko rastojanje termoklina-DCM), sto
    nije imalo utemeljenja u podacima. Sada se cita iz samog profila: trazi se
    raspon dubina na kojima hlorofil prelazi polovinu izmedju povrsinske
    vrijednosti i maksimuma.
    """
    cmax = np.nanmax(chl, axis=0)
    csurf = chl[0]
    half = csurf + 0.5 * (cmax - csurf)
 
    iznad = chl >= half[None, :, :]
    dubine = depth[:, None, None] * np.ones_like(chl)
    dubine = np.where(iznad, dubine, np.nan)
 
    with np.errstate(invalid="ignore"):
        gornja = np.nanmin(dubine, axis=0)
        donja = np.nanmax(dubine, axis=0)
    debljina = donja - gornja
    return np.where(np.isfinite(debljina) & (debljina > 0), debljina, np.nan)
 
 
def prey_layer(mld: np.ndarray, z_thermo: np.ndarray, z_dcm: np.ndarray,
               dcm_debljina: np.ndarray | None = None):
    """
    Sloj u kojem se ocekuje koncentracija planktonozdera (sardela, gavun, girica).
 
    Planktonozderi prate DCM kad on postoji, inace ostaju u mijesanom sloju.
    Debljina se uzima iz izmjerene sirine DCM-a; samo ako mjerenje nije moguce
    pada se na procjenu iz rastojanja do termokline.
    Vraca (centar_m, debljina_m).
    """
    center = np.where(np.isnan(z_dcm), mld * 0.6, z_dcm)
 
    upper = np.fmin(np.nan_to_num(z_thermo, nan=1e4), center)
    procjena = np.clip((center - upper) * 2.0, 5.0, 60.0)
 
    if dcm_debljina is None:
        thickness = procjena
    else:
        thickness = np.where(np.isfinite(dcm_debljina), dcm_debljina, procjena)
    return center, np.clip(thickness, 4.0, 80.0)
 
 
def reachability(prey_center: np.ndarray, prey_thickness: np.ndarray,
                 max_depth: float = 50.0) -> np.ndarray:
    """
    Udio sloja plijena koji lezi unutar dohvata panule (0..max_depth), 0..1.
 
    Ovo je ono sto razdvaja upotrebljivu zonu od zanimljive. Ljeti, kad je
    stratifikacija jaka i DCM sjedne na 60-70 m, plijen je zbijen ali izvan
    dohvata — zona se gasi. U jesen, kad mijesanje digne sloj ka povrsini,
    ista pozicija postaje najbolja na potezu.
    """
    top = np.clip(prey_center - prey_thickness / 2.0, 0, None)
    bottom = prey_center + prey_thickness / 2.0
    overlap = np.clip(np.fmin(bottom, max_depth) - top, 0, None)
    total = np.clip(bottom - top, 1e-6, None)
    return np.clip(overlap / total, 0, 1)
 
 
def vertical_concentration(thermo_strength: np.ndarray,
                           prey_thickness: np.ndarray) -> np.ndarray:
    """
    Indeks vertikalne koncentracije plijena, 0..1.
 
    Ostra termoklina + tanak sloj plijena = plijen je stisnut, predatori
    predvidljivi. Izmijesan stub = plijen razvucen, uspjesnost pada.
    """
    s = np.clip(thermo_strength / 0.15, 0, 1)           # 0.15 C/m = vrlo ostra
    t = np.clip((60.0 - prey_thickness) / 55.0, 0, 1)
    return np.sqrt(np.clip(s, 0.05, 1) * np.clip(t, 0.05, 1))
 
 
def troll_depth_advice(sp_range, prey_center: np.ndarray,
                       follows_dcm: bool, max_depth: float = 50.0) -> np.ndarray:
    """
    Preporucena dubina panule: presjek vrsti svojstvenog raspona, sloja plijena
    i tvrde granice od max_depth metara. Vrste koje ne prate DCM (lampuga,
    strelka, lica, gof) ostaju u svom rasponu.
    """
    lo = min(sp_range[0], max_depth)
    hi = min(sp_range[1], max_depth)
    if not follows_dcm:
        return np.full_like(prey_center, 0.5 * (lo + hi), dtype=float)
    return np.clip(prey_center, lo, hi)
 
 
# --------------------------------------------------------------- OSTALI SLOJEVI
def forage_index(chl_surf: np.ndarray, chl_grad: np.ndarray,
                 vert_conc: np.ndarray) -> np.ndarray:
    """
    Proxy za prisustvo mamca. Nije apsolutna vrijednost hlorofila nego
    kombinacija produktivnosti, njenog gradijenta i vertikalne koncentracije.
    """
    p = np.clip(np.log10(np.clip(chl_surf, 0.01, None)) + 1.5, 0, 2) / 2.0
    g = _norm(chl_grad)
    return np.clip(0.45 * p + 0.25 * g + 0.30 * vert_conc, 0, 1)
 
 
def current_shear(u: np.ndarray, v: np.ndarray, lat: np.ndarray,
                  res_deg: float) -> np.ndarray:
    """Magnituda smicanja povrsinskih struja — ivice struja skupljaju plijen."""
    dx_km, dy_km = cell_size_km(lat, res_deg)
    dudy = np.gradient(u, axis=0) / dy_km
    dvdx = np.gradient(v, axis=1) / dx_km[:, None]
    dudx = np.gradient(u, axis=1) / dx_km[:, None]
    dvdy = np.gradient(v, axis=0) / dy_km
    strain = np.hypot(dudx - dvdy, dvdx + dudy)
    return strain
 
 
def bojana_plume(salinity: np.ndarray, sal_ocean: float = 38.3) -> np.ndarray:
    """
    Jacina bocatnog uticaja, 0..1. Racuna se iz povrsinskog saliniteta:
    sto je nizi od otvorenog mora, jaci je uticaj Bojane.
    """
    deficit = np.clip(sal_ocean - salinity, 0, 8.0)
    return deficit / 8.0
 
 
def _norm(a: np.ndarray) -> np.ndarray:
    """Robusna normalizacija na 0..1 po 5. i 95. percentilu."""
    lo, hi = np.nanpercentile(a, [5, 95])
    if not np.isfinite(hi - lo) or hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)
 
