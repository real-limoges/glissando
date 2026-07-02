# s09 QC report — fires_enriched.parquet

- FRAP release: `fire25_1`
- artifact: `fires_enriched.parquet`
- artifact sha256: `222a8e9a3e50148b43448759a5bcf0d944c09927c43bd2cadf9c6acf8cd6133f`
- rows: 23205
- CRS: EPSG:3310
- coarse-geometry threshold (vertices/km): 3.0

## Row counts by stage

| stage | rows |
|---|---|
| s01 raw | 23334 |
| s02 cleaned | 23334 |
| s03 deduplicated | 23205 |
| s09 final | 23205 |

## Field summary

| column | nulls | null % | min | max |
|---|---|---|---|---|
| fire_id | 0 | 0.0% | 0001fe207103 | fffee3d8957d |
| year | 77 | 0.3% | 1878 | 2025 |
| state | 0 | 0.0% | AZ | OR |
| agency | 48 | 0.2% | BIA | USF |
| unit_id | 60 | 0.3% | ADR | YNP |
| fire_name | 6583 | 28.4% |  "Y" | ZWINGE |
| fire_name_norm | 0 | 0.0% |  | ZWINGE |
| inc_num | 953 | 4.1% | 00000000 | SKU03705 |
| irwin_id | 19575 | 84.4% |  | {f9653bc1-2e4a-42bb-81be-6df0f6a3a91b} |
| complex_name | 22594 | 97.4% | 2022 SRF Lightning Complex | YUBA RIVER COMPLEX |
| complex_id | 22622 | 97.5% |  | {FE46567A-A03E-4D5C-91E5-2788D89AA5A9} |
| alarm_date | 5384 | 23.2% | 1898-04-01 00:00:00 | 2025-12-24 00:00:00 |
| cont_date | 12526 | 54.0% | 1912-08-31 00:00:00 | 2025-12-24 00:00:00 |
| alarm_year | 5384 | 23.2% | 1898 | 2025 |
| alarm_month | 5384 | 23.2% | 1 | 12 |
| cause_code | 0 | 0.0% | 1 | 19 |
| cause_desc | 0 | 0.0% | Aircraft | Vehicle |
| collection_method | 0 | 0.0% | 1 | 8 |
| objective_code | 269 | 1.2% | 1 | 2 |
| gis_acres | 0 | 0.0% | 0.0008994204108603299 | 1032699.625 |
| area_km2 | 0 | 0.0% | 3.6398647128899313e-06 | 4179.1876304481375 |
| perimeter_km | 0 | 0.0% | 0.007265638128436615 | 1533.8200844015796 |
| n_vertices | 0 | 0.0% | 4 | 117172 |
| vertices_per_km | 0 | 0.0% | 0.7269535266017185 | 12293.384298256098 |
| coarse_geometry | 0 | 0.0% | False | True |
| centroid_lon | 0 | 0.0% | -124.39486796123775 | -114.13781827199284 |
| centroid_lat | 0 | 0.0% | 32.5382183872534 | 42.29516408532259 |
| division | 1 | 0.0% | 1 | 7 |
| division_assigned_nearest | 0 | 0.0% | False | True |
| pdsi | 5385 | 23.2% | -8.7 | 8.87 |
| tavg_degf | 5385 | 23.2% | 27.7 | 91.5 |
| precip_in | 5385 | 23.2% | 0.0 | 17.17 |
| awnd_ms | 11772 | 50.7% | 1.34 | 7.6 |
| awnd_n_stations | 11772 | 50.7% | 1 | 31 |
| src_row | 0 | 0.0% | 0 | 23333 |

## Climate coverage (among fires with a division and an alarm date)

- joinable fires: 17820 / 23205
- pdsi: 100.0%
- tavg_degf: 100.0%
- precip_in: 100.0%
- awnd_ms: 64.2%

## Flags

- coarse_geometry: 197 (0.8%)
- division_assigned_nearest: 42
- null division: 1
- null alarm_date: 5384
