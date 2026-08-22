# Panula — backend

Predikcija zona najveće aktivnosti pelagičnih riba, potez Petrovac → Ada Bojana
(more ispred ušća Bojane i samo ušće; rijeka nije u domenu).

Vrste: lampuga, trupac, plamida, tuna, sabljarka, gof, strelka, lica.

## Kako radi

Dnevni posao (cron, ~05:00) preuzme polja, izračuna izvedene slojeve, skorira
svaku ćeliju za svaku vrstu i izbaci **poligone zona** — ne tačke — za danas i
naredna dva dana. Rezultat je statički GeoJSON koji frontend čita bez
autentikacije. Nijedan ključ ne izlazi u pretraživač.

```
cron → fetch → fields → zones → public/*.json → Artifact
```

## Pokretanje bez servera (preporučeno)

Nema mašine koju treba održavati. GitHub sam pokreće posao svaki dan u 05:00
i objavljuje rezultat. Besplatno za javni repozitorij.

1. Otvori nalog na `github.com` i napravi novi repozitorij.
2. Ubaci ove fajlove u njega (dugme *Add file → Upload files*).
3. Otvori besplatan nalog na `marine.copernicus.eu`.
4. U repozitorijumu: *Settings → Secrets and variables → Actions → New
   repository secret*. Dodaj dva: `CMEMS_USERNAME` i `CMEMS_PASSWORD`.
5. *Settings → Pages → Source: Deploy from a branch → gh-pages*.
6. Kartica *Actions → Dnevni ciklus → Run workflow* — pokreni prvi put ručno.

Poslije toga radi sam. Rezultat je na
`https://<tvoj-nalog>.github.io/<repozitorij>/index.json`.

Batimetrija se preuzima automatski sa EMODnet-a pri prvom pokretanju i čuva
se između pokretanja — nema fajlova za ručno skidanje.

## Pokretanje na svom računaru (opciono)

```bash
pip install -r requirements.txt
export COPERNICUSMARINE_SERVICE_USERNAME=...
export COPERNICUSMARINE_SERVICE_PASSWORD=...
python -m panula.build              # jedan ciklus, stvarni podaci
python -m panula.build --sinteticki # cijeli lanac bez mreže
```

## Ugovor o izlazu

Ovo je granica između backenda i interfejsa. Kad se ovo zaključa, frontend se
može praviti nezavisno.

`public/index.json` — šta postoji i koliko je svježe:

```json
{
  "generisano": "2026-08-22T05:12:00Z",
  "domen": {"lon": [18.60, 19.50], "lat": [41.60, 42.30]},
  "dani": ["2026-08-22", "2026-08-23", "2026-08-24"],
  "vrste": ["lampuga", "trupac", "plamida", "tuna",
            "sabljarka", "gof", "strelka", "lica"],
  "izvori": {
    "sst": {"datum": "2026-08-21", "rezolucija_km": 1.1, "zastarjelo": false},
    "hlorofil": {"datum": "2026-08-19", "senzor": "OLCI 300 m",
                 "pokrivanje": 0.62, "zastarjelo": false},
    "struje": {"datum": "2026-08-22", "rezolucija_km": 4.2},
    "talasi": {"datum": "2026-08-22"}
  }
}
```

`public/{vrsta}_{datum}.geojson` — zone za jednu vrstu i jedan dan:

```json
{
  "type": "FeatureCollection",
  "properties": {
    "vrsta": "plamida",
    "latinski": "Sarda sarda",
    "datum": "2026-08-22",
    "panula": {"dubina_m": [0, 25], "brzina_kn": [4.0, 6.5]},
    "vertikala": {
      "termoklina_m": 38, "dcm_m": 52, "mld_m": 24,
      "sloj_plijena_m": [30, 62],
      "preporucena_dubina_m": 18,
      "objasnjenje": "Oštra termoklina na 38 m drži plijen stisnut..."
    },
    "bezbjednost": {
      "nivo": "zuto",
      "razlog": "talasi do 1.2 m",
      "izlazak_preporucen": true
    }
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Polygon", "coordinates": [[[19.08, 42.01], "..."]]},
      "properties": {
        "score_mean": 62.4, "score_max": 78.1, "area_km2": 14.3,
        "centroid": [19.11, 42.03],
        "razlog": "termalni front + prisustvo mamca"
      }
    }
  ]
}
```

Ključna polja za interfejs:

| Polje | Zašto postoji |
|---|---|
| `razlog` | korisnik mora znati **zašto** mu se nudi zona, inače je crna kutija |
| `zastarjelo` | star podatak se nikad ne prikazuje kao svjež |
| `bezbjednost.nivo` | crveno = zone se računaju ali se prikazuju zaključane |
| `vertikala` | za panulu je dubina jednako važna kao pozicija |

## Moduli

| Fajl | Sadržaj |
|---|---|
| `config.py` | domen, identifikatori Copernicus datasetova, pragovi |
| `species.py` | matrica vrsta — kapije, težine, vertikalne preferencije |
| `fetch.py` | Copernicus + Open-Meteo, logika „najskoriji dostupan hlorofil" |
| `fields.py` | fronte, tendencija, termoklina, DCM, sloj plijena |
| `zones.py` | skoriranje, izdvajanje poligona, upozorenja |
| `static.py` | nagib, rastojanja do izobate/strukture/Bojane |
| `bathymetry.py` | automatsko preuzimanje batimetrije sa EMODnet WCS |
| `predictors.py` | sklapanje 19 prediktora, provjera pokrivenosti težina |
| `build.py` | dnevni ciklus, pisanje izlaznog ugovora |

## Tvrda granica dubine

Panula se ne vuče dublje od **50 m** bez obzira na dubinu mora. To nije samo
ograničenje prikaza nego ulazi u skor: `fields.reachability()` računa koliki
udio sloja plijena leži unutar dohvata, i množi vertikalnu koncentraciju.

Posljedica je da ista pozicija mijenja vrijednost kroz sezonu. Ljeti, kad jaka
stratifikacija spusti DCM na 60–70 m, plijen je zbijen ali izvan dohvata i zona
se gasi. U jesen, kad miješanje digne sloj ka površini, ista pozicija postaje
najbolja na potezu. Polje `vertikala.dohvatljivost` (0–1) pokazuje koliko je
toga u dohvatu, a `dubina_panule_m` po zoni daje konkretnu preporuku.

Sabljarka je zbog ove granice svedena na 15–50 m — lovi se samo kad se noću
podigne u površinski sloj.

## Namjena

Lična upotreba, nije komercijalni proizvod. Copernicus i EMODnet podaci su
otvoreni, ali njihovi uslovi korišćenja traže navođenje izvora — to stoji u
izlazu. EMODnet batimetrija nosi izričitu napomenu da se **ne koristi za
navigaciju**; ova aplikacija predlaže gdje loviti, ne kuda ploviti.

## Poznato ograničenje

Kad je skor polje gotovo ravno (nema kontrasta u prediktorima), rasijecanje
prevelikih zona ne uspijeva i izlazi jedna difuzna zona preko cijelog domena.
Vidljivo na sintetičkom testu kod sabljarke. Sa stvarnim poljima batimetrije
kontrast postoji, ali izlaz treba obilježiti kao „difuzno" umjesto ponuditi
ga kao zonu.
