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
 
 
def hladno_jezgro(sto: np.ndarray, lat: np.ndarray, res_deg: float,
                  radius_km: float = 8.0) -> np.ndarray:
    """
    Koliko je celija hladnija od svoje okoline, u stepenima.
 
    Pozitivno znaci hladnije od okolnog mora. Front je granica dvije vodene
    mase i ide u oba smjera; upwelling je hladno JEZGRO usred toplije vode,
    pa se prepoznaje samo po negativnom odstupanju.
    """
    dx_km, _ = cell_size_km(lat, res_deg)
    k = max(1, int(round(radius_km / float(np.mean(dx_km)))))
 
    popuna = np.where(np.isnan(sto), np.nanmean(sto), sto)
    okolina = ndimage.uniform_filter(popuna, size=2 * k + 1)
    odstupanje = okolina - popuna
    odstupanje[np.isnan(sto)] = np.nan
    return odstupanje
 
 
def upwelling(sst: np.ndarray, chl: np.ndarray, lat: np.ndarray,
              res_deg: float, prag: float = SST_PRAG) -> np.ndarray:
    """
    Indeks izdizanja dubinske vode, 0..1.
 
    Trazi se poklapanje dva traga: povrsina hladnija od okoline za bar `prag`
    stepeni, i hlorofil visi nego okolo. Sama hladna mrlja moze biti i ostatak
    nocnog hladjenja; tek uz povecanu produktivnost govori o dotoku nutrijenata
    iz dubine. Mnoze se, pa oba traga moraju postojati.
    """
    hladnoca = np.clip(hladno_jezgro(sst, lat, res_deg) / prag, 0, 2) / 2
    log_chl = np.log10(np.clip(chl, 0.01, None))
    obogacenje = np.clip(-hladno_jezgro(log_chl, lat, res_deg) / 0.15, 0, 1)
    return np.sqrt(np.clip(hladnoca, 0, 1) * obogacenje)
 
 
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
def thermocline(theta: np.ndarray, depth: np.ndarray,
                mld: np.ndarray | None = None, od_dubine: float = 10.0,
                do_dubine: float = 90.0, prag_grad: float = 0.05):
    """
    Dubina i jacina sezonske termokline iz profila potencijalne temperature.
 
    theta: [depth, lat, lon]; depth: [depth] u metrima (rastuce).
 
    Ljeti vodeni stub cesto ima dvije termokline: ostru i tanku na bazi dnevno
    zagrijanog povrsinskog sloja, i sezonsku dublje. Tanka je obicno JACA, pa
    trazenje najjaceg gradijenta pogresno vraca nju — a plijen drzi sezonska.
 
    Zato se ne trazi najjaci nego NAJDUBLJI izrazen gradijent unutar pojasa
    koji je za panulu uopste relevantan. "Izrazen" znaci lokalni maksimum
    jaci od `prag_grad` (C/m). Dubina mijesanog sloja, kad je ima, pomjera
    donju granicu trazenja ispod povrsinskog sloja.
    """
    dtheta = np.diff(theta, axis=0)
    ddepth = np.diff(depth)[:, None, None]
    grad = np.abs(dtheta / ddepth)                      # C/m po sloju
    mid = 0.5 * (depth[:-1] + depth[1:])
 
    donja = (np.fmax(np.nan_to_num(mld, nan=od_dubine) * 0.9, od_dubine)
             if mld is not None else np.full(theta.shape[1:], od_dubine))
    pojas = ((mid[:, None, None] >= donja[None, :, :]) &
             (mid[:, None, None] <= do_dubine))
 
    g = np.where(pojas, np.nan_to_num(grad, nan=0.0), -1.0)
 
    # lokalni maksimumi po dubini
    lijevo = np.vstack([g[:1], g[:-1]])
    desno = np.vstack([g[1:], g[-1:]])
    vrh = (g >= lijevo) & (g >= desno) & (g >= prag_grad)
 
    # najdublji takav vrh
    ima = vrh.any(axis=0)
    idx = (len(mid) - 1) - np.argmax(vrh[::-1], axis=0)
 
    zt = np.where(ima, mid[idx], np.nan)
    jac = np.where(ima, np.take_along_axis(g, idx[None], axis=0)[0], 0.0)
    return zt, np.clip(jac, 0, None)
 
 
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
 
