# CMP data dictionary

This document covers both the raw PHM Data Challenge 2016 CMP files and the
feature table produced by this repository. The raw column definitions below are
transcribed from [`docs/competition/2016_phm_data_challenge.pdf`](competition/2016_phm_data_challenge.pdf).
Raw competition data are not redistributed by this repository; place them under
`data/raw/`.

## Raw time-series columns

Each raw row is one observation identified by `WAFER_ID`, `STAGE`, `CHAMBER`,
and `TIMESTAMP`. The official document does not provide engineering units for
the pressure, rotation, flow, or usage signals; their original numeric scales
are therefore retained.

| Symbol | Column name | Official type / unit | Description |
|---|---|---|---|
| x1 | `MACHINE_ID` | Numeric ID | ID of machine |
| x2 | `MACHINE_DATA` | Numeric ID | ID of wafer-ring location in machine |
| x3 | `TIMESTAMP` | Seconds | Time-series observation timestamp |
| x4 | `WAFER_ID` | Number | Number representing the wafer ID |
| x5 | `STAGE` | `A` or `B` | Different type of processing stage |
| x6 | `CHAMBER` | Chamber | Chamber in the machine used for wafer processing |
| x7 | `USAGE_OF_BACKING_FILM` | `A` usage measure | Usage measure of polish-pad backing film |
| x8 | `USAGE_OF_DRESSER` | `A` usage measure | Usage measure of dresser |
| x9 | `USAGE_OF_POLISHING_TABLE` | `A` usage measure | Usage measure of polishing table |
| x10 | `USAGE_OF_DRESSER_TABLE` | `A` usage measure | Usage measure of dresser table |
| x11 | `PRESSURIZED_CHAMBER_PRESSURE` | Chamber pressure | Pressure in the chamber |
| x12 | `MAIN_OUTER_AIR_BAG_PRESSURE` | Pressure | Pressure related to wafer placement |
| x13 | `CENTER_AIR_BAG_PRESSURE` | Pressure | Pressure related to wafer placement |
| x14 | `RETAINER_RING_PRESSURE` | Pressure | Pressure related to wafer placement |
| x15 | `RIPPLE_AIR_BAG_PRESSURE` | Pressure | Pressure related to wafer placement |
| x16 | `USAGE_OF_MEMBRANE` | `A` usage measure | Usage measure of polishing membrane |
| x17 | `USAGE_OF_PRESSURIZED_SHEET` | `A` usage measure | Usage measure of wafer-carrier flexible sheet |
| x18 | `SLURRY_FLOW_LINE_A` | Flow rate | Flow rate of slurry type A |
| x19 | `SLURRY_FLOW_LINE_B` | Flow rate | Flow rate of slurry type B |
| x20 | `SLURRY_FLOW_LINE_C` | Flow rate | Flow rate of slurry type C |
| x21 | `WAFER_ROTATION` | Rotation rate | Rotation rate of wafer |
| x22 | `STAGE_ROTATION` | Rotation rate | Rotation rate of stage |
| x23 | `HEAD_ROTATION` | Rotation rate | Rotation rate of head |
| x24 | `DRESSING_WATER_STATUS` | Status | Status of dressing water |
| x25 | `EDGE_AIR_BAG_PRESSURE` | Pressure | Pressure of the bag on the edge of the wafer |

## Target and sample keys

| Column | Role | Description |
|---|---|---|
| `WAFER_ID` | Key | Wafer identifier |
| `STAGE` | Key / categorical input | Stage `A` or `B`; retained as a model input after encoding (`A=0`, `B=1`) |
| `AVG_REMOVAL_RATE` | Target | Average material removal rate (MRR) supplied by the label files |
| `START_TIMESTAMP` | Derived time field | Minimum raw `TIMESTAMP` for each `WAFER_ID × STAGE`; used for ordering and time diagnostics |

The model sample key is `WAFER_ID × STAGE`. `CHAMBER` is not a row key in the
final table; it is expanded into feature suffixes `Ch1` through `Ch6` so that
each sample retains chamber-specific information.

## Processed feature naming

The current processed training table has 166 columns: `WAFER_ID`, `STAGE`,
162 chamber-expanded features, `AVG_REMOVAL_RATE`, and `START_TIMESTAMP`.
Every chamber feature follows this pattern:

```text
<source_signal>_<aggregation_or_statistic>_Ch<chamber_number>
```

`Ch1`–`Ch6` refer to the original `CHAMBER` values. The following inventory
documents every generated feature family and its six chamber-specific columns.

| Feature family / pattern | Number | Source | Calculation |
|---|---:|---|---|
| `USAGE_OF_DRESSER_max_Ch1..Ch6` | 6 | `USAGE_OF_DRESSER` | Min-Max scale on training data, then maximum within `WAFER_ID × STAGE × CHAMBER` |
| `USAGE_OF_POLISHING_TABLE_max_Ch1..Ch6` | 6 | `USAGE_OF_POLISHING_TABLE` | Min-Max scale, then maximum |
| `USAGE_OF_DRESSER_TABLE_max_Ch1..Ch6` | 6 | `USAGE_OF_DRESSER_TABLE` | Min-Max scale, then maximum |
| `USAGE_OF_MEMBRANE_max_Ch1..Ch6` | 6 | `USAGE_OF_MEMBRANE` | Min-Max scale, then maximum |
| `pressure_duration_sum_Ch1..Ch6` | 6 | `PRESSURIZED_CHAMBER_PRESSURE`, `TIMESTAMP` | Sum of adjacent `Δt` while chamber pressure is greater than zero; proxy for total pressure-application time |
| `PRESSURIZED_CHAMBER_PRESSURE_nonzero_mean/std_Ch1..Ch6` | 12 | `PRESSURIZED_CHAMBER_PRESSURE` | Mean and sample standard deviation of values greater than `0.1` |
| `MAIN_OUTER_AIR_BAG_PRESSURE_nonzero_mean/std_Ch1..Ch6` | 12 | `MAIN_OUTER_AIR_BAG_PRESSURE` | Mean and sample standard deviation of values greater than `0.1` |
| `RETAINER_RING_PRESSURE_nonzero_mean/std_Ch1..Ch6` | 12 | `RETAINER_RING_PRESSURE` | Mean and sample standard deviation of values greater than `0.1` |
| `SLURRY_FLOW_LINE_A_sum_Ch1..Ch6` | 6 | `SLURRY_FLOW_LINE_A` | Sum of observed signal values |
| `SLURRY_FLOW_LINE_B_sum_Ch1..Ch6` | 6 | `SLURRY_FLOW_LINE_B` | Sum of observed signal values |
| `SLURRY_FLOW_LINE_C_sum_Ch1..Ch6` | 6 | `SLURRY_FLOW_LINE_C` | Sum of observed signal values |
| `WAFER_ROTATION_nonzero_mean/std_Ch1..Ch6` | 12 | `WAFER_ROTATION` | Mean and sample standard deviation of values greater than `0.1` |
| `STAGE_ROTATION_nonzero_median/std_Ch1..Ch6` | 12 | `STAGE_ROTATION` | Median and sample standard deviation of values greater than `0.1` |
| `HEAD_ROTATION_nonzero_mean/std_Ch1..Ch6` | 12 | `HEAD_ROTATION` | Mean and sample standard deviation of values greater than `0.1` |
| `DRESSING_WATER_STATUS_mean_Ch1..Ch6` | 6 | `DRESSING_WATER_STATUS` | Mean status value, interpreted as the observed proportion of active dressing water |
| `polishing_PRESSURIZED_CHAMBER_PRESSURE_nanmean/nanstd_Ch1..Ch6` | 12 | Pressure plus wafer/stage rotation | Mean and standard deviation after retaining samples where chamber pressure is greater than zero and both wafer and stage rotation exceed `0.1` |
| `polishing_MAIN_OUTER_AIR_BAG_PRESSURE_nanmean/nanstd_Ch1..Ch6` | 12 | Pressure plus wafer/stage rotation | Same polishing-condition mask as above |
| `polishing_RETAINER_RING_PRESSURE_nanmean/nanstd_Ch1..Ch6` | 12 | Pressure plus wafer/stage rotation | Same polishing-condition mask as above |

The feature counts above add up to 162 chamber-expanded columns. The current
baseline does not create separate `polishing_duration`, `soaking_duration`,
`spinning_duration`, or `idle_duration` columns; `pressure_duration` is the
single duration proxy used by the model.

## Raw fields not present in the processed table

These raw columns remain part of the source schema but are not emitted by the
current aggregation rules:

| Raw column | Current treatment | Reason |
|---|---|---|
| `MACHINE_ID` | Not aggregated | Machine identifier is not used as a process statistic |
| `MACHINE_DATA` | Not aggregated | Exactly one-to-one with `CHAMBER`, so retaining both would duplicate the same information; `CHAMBER` is retained as the process-path variable |
| `USAGE_OF_BACKING_FILM` | Not aggregated; also excluded by model preparation | Collinear with `USAGE_OF_PRESSURIZED_SHEET`; one of the redundant usage variables is omitted |
| `USAGE_OF_PRESSURIZED_SHEET` | Not aggregated | Collinear with `USAGE_OF_BACKING_FILM`; the current feature set keeps neither redundant usage signal |
| `CENTER_AIR_BAG_PRESSURE` | Dropped before aggregation | Listed in the project’s collinearity exclusion set |
| `RIPPLE_AIR_BAG_PRESSURE` | Dropped before aggregation | Listed in the project’s collinearity exclusion set |
| `EDGE_AIR_BAG_PRESSURE` | Dropped before aggregation | Listed in the project’s collinearity exclusion set |

Their absence from the processed table does not mean the raw signals are
invalid; it records the current feature-selection decision. Reintroducing any
of them should be a separate experiment with its own validation result.

## Missing values and aggregation conventions

- A chamber not visited by a wafer-stage produces missing values for that
  chamber’s feature columns. The modeling pipeline fills these missing values
  with `0` after the high/low chamber grouping is defined.
- A signal with no value above the nonzero threshold produces `NaN` for its
  nonzero statistics. This is different from an observed numeric zero.
- `pressure_duration` uses `PRESSURIZED_CHAMBER_PRESSURE > 0`; the nonzero
  summary statistics use the implementation threshold `> 0.1`.
- Min-Max scalers are fitted on training data and then reused for test data.
- The final table is aggregated at `WAFER_ID × STAGE × CHAMBER` first, then
  unstacked by chamber and joined to labels at `WAFER_ID × STAGE`.
