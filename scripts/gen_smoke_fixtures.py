"""One-off generator for the synthetic smoke fixtures (committed as static files)."""
import json
from pathlib import Path

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pipeline"


def rect(w, s, e, n):
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


# --- climate divisions: two adjacent rectangles, CLIMDIV = state*100 + div ---
divisions = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"CLIMDIV": 401, "NAME": "SYNTH NORTH COAST"},
         "geometry": rect(-124.0, 36.0, -120.0, 42.0)},
        {"type": "Feature", "properties": {"CLIMDIV": 402, "NAME": "SYNTH SOUTHEAST"},
         "geometry": rect(-120.0, 32.0, -114.0, 42.0)},
    ],
}
(FIX / "divisions_synthetic.geojson").write_text(json.dumps(divisions, indent=1) + "\n")


def props(name, year, alarm, cause, acres, **kw):
    p = {"YEAR_": year, "STATE": "CA", "AGENCY": "CDF", "UNIT_ID": "XXU",
         "FIRE_NAME": name, "INC_NUM": "00000001", "IRWINID": None,
         "ALARM_DATE": alarm, "CONT_DATE": alarm, "CAUSE": cause,
         "C_METHOD": 1, "OBJECTIVE": 1, "GIS_ACRES": acres,
         "COMPLEX_NAME": None, "Shape_Length": 1.0}
    p.update(kw)
    return p


# ~1 km squares: 5 vertices / ~4 km perimeter ≈ 1.25 vertices/km, above the
# coarse threshold; GAMMA's huge 4-vertex triangle stays far below it.
sq = lambda lon, lat, d=0.005: rect(lon - d, lat - d, lon + d, lat + d)
fires = {
    "type": "FeatureCollection",
    "features": [
        # ALPHA: normal, division 1; the kept member of the duplicate pair
        {"type": "Feature", "properties": props("ALPHA", 2001, "2001-07-15", 1, 1000.0),
         "geometry": sq(-122.0, 38.0)},
        # ALPHA duplicate (same year/name/alarm, smaller acres) -> dropped in s03
        {"type": "Feature", "properties": props("ALPHA", 2001, "2001-07-15", 1, 500.0),
         "geometry": sq(-122.01, 38.01)},
        # " beta ": lowercase + padding, division 2 -> tests name normalization
        {"type": "Feature", "properties": props(" beta ", 2001, "2001-08-02", 7, 300.0),
         "geometry": sq(-116.0, 33.0)},
        # GAMMA: huge triangle, 4 coords -> coarse_geometry
        {"type": "Feature", "properties": props("GAMMA", 2002, "2002-06-10", 2, 60000.0),
         "geometry": {"type": "Polygon", "coordinates": [[[-123.0, 40.0], [-122.4, 40.0], [-122.7, 40.5], [-123.0, 40.0]]]}},
        # DELTA: bow-tie (self-intersecting) -> repaired by make_valid
        {"type": "Feature", "properties": props("DELTA", 2001, "2001-07-20", 5, 120.0),
         "geometry": {"type": "Polygon", "coordinates": [[[-123.0, 40.8], [-122.9, 40.9], [-123.0, 40.9], [-122.9, 40.8], [-123.0, 40.8]]]}},
        # EPSILON: null geometry -> dropped in s02
        {"type": "Feature", "properties": props("EPSILON", 2001, "2001-09-01", 9, 10.0),
         "geometry": None},
        # ZETA: no alarm_date -> kept, climate covariates null
        {"type": "Feature", "properties": props("ZETA", 2003, None, 14, 50.0),
         "geometry": sq(-121.0, 37.0)},
        # ETA: just offshore west of division 1 -> nearest-division fallback
        {"type": "Feature", "properties": props("ETA", 2002, "2002-06-20", 14, 80.0),
         "geometry": sq(-124.08, 38.0)},
    ],
}
(FIX / "firep_synthetic.geojson").write_text(json.dumps(fires, indent=1) + "\n")

# --- climdiv fixed-width files: state 04, divisions 1-2, years 2001-2003 ---
# value patterns keyed for assertions: pdsi = div + month/100; tavg = 60 + month;
# pcpn = month/10. December is the sentinel (missing) for each element.
ELEMENTS = {"pdsidv": ("05", -99.99), "tmpcdv": ("02", -99.90), "pcpndv": ("01", -9.99)}
for name, (code, sentinel) in ELEMENTS.items():
    lines = []
    for div in (1, 2):
        for year in (2001, 2002, 2003):
            vals = []
            for m in range(1, 13):
                if m == 12:
                    v = sentinel
                elif name == "pdsidv":
                    v = div + m / 100
                elif name == "tmpcdv":
                    v = 60.0 + m
                else:
                    v = m / 10
                vals.append(f"{v:7.2f}")
            lines.append(f"{4:02d}{div:02d}{code}{year:04d}" + "".join(vals))
    (FIX / "climdiv" / f"climdiv-{name}-v1.0.0-99999999").write_text("\n".join(lines) + "\n")

# --- GSOM CSV: two stations in division 1, one in division 2 ---
rows = ["STATION,DATE,LATITUDE,LONGITUDE,AWND"]
station_months = [
    ("USW0001", 38.5, -122.5, {"2001-06": 2.0, "2001-07": 3.0, "2002-06": 6.0}),
    ("USW0002", 39.0, -121.0, {"2001-06": 4.0, "2001-07": 5.0, "2002-06": 8.0}),
    ("USW0003", 33.5, -116.5, {"2001-07": 9.0, "2001-08": 7.5}),
]
for sid, lat, lon, months in station_months:
    for date, awnd in months.items():
        rows.append(f"{sid},{date},{lat},{lon},{awnd}")
(FIX / "gsom_synthetic.csv").write_text("\n".join(rows) + "\n")
print("fixtures written")
