# Data dictionary

The project uses the PHM Data Challenge 2016 CMP data set. Raw files are not
redistributed by this repository; place them under `data/raw/`.

Key identifiers and target:

| Field | Description |
|---|---|
| `WAFER_ID` | Wafer identifier |
| `STAGE` | CMP process stage (`A` or `B`) |
| `CHAMBER` | Processing chamber identifier |
| `TIMESTAMP` | Observation timestamp |
| `AVG_REMOVAL_RATE` | Average material removal rate; prediction target |

Major sensor families include chamber and air-bag pressures, head/wafer/stage
rotation, slurry flow, dressing-water status, and consumable usage counters.
