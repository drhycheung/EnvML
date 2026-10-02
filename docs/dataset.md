# Dataset card — UCI Beijing PM2.5 Data

## Source

| | |
|---|---|
| **Name** | Beijing PM2.5 Data |
| **Provider** | UCI Machine Learning Repository |
| **Link** | <https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data> |
| **Local copy** | `data/beijing_pm25.csv` (1.9 MB) |
| **Licence** | CC BY 4.0 |
| **Citation** | Song, C., Zhou, Y., Lu, J., & Qi, J. (2016). Detecting Outliers in Beijing's PM2.5: A Data Mining Approach Based on PM2.5 Five Years of Data in Beijing. *ISPRS International Journal of Geo-Information, 5*(4), 48. |

## Contents

Hourly meteorological and pollutant observations for Beijing, 2010–2014: **43,824** rows.

| Column | Meaning | Used? |
|---|---|---|
| `No` | Row index | no — pure index, carries no information |
| `year` `month` `day` `hour` | Timestamp | `month` and `hour` only, as cyclical sin/cos terms |
| `pm2.5` | PM2.5 concentration, µg/m³ | **target** |
| `DEWP` | Dew point, °C | yes |
| `TEMP` | Air temperature, °C | yes |
| `PRES` | Atmospheric pressure, hPa | yes |
| `cbwd` | Wind direction: `NW` / `NE` / `SE` / `cv` (calm and variable) | yes, as an ordinal code |
| `Iws` | Cumulative wind-speed **counter** reading | **no — see below** |
| `Is` | Cumulative precipitation, mm | no |
| `Ir` | Cumulative sunshine hours | no |

## How this project uses it

- Drops the **4.72%** of rows with missing `pm2.5`, leaving **41,757** rows for training.
- Builds a **10-dimensional** feature set: `TEMP`, `DEWP`, `TEMP−DEWP`, `PRES`,
  `PRES−1013.25`, sin/cos of `hour`, sin/cos of `month`, and `wind_dir` coded
  NW=0, NE=1, SE=2, cv=3.
- Deliberately **excludes** the co-measured pollutants (SO₂, NO₂, CO, O₃), along
  with `year`, `day` and `No`. Including same-hour SO₂ or NO₂ would let the model
  "predict" PM2.5 from its own strongest correlates rather than from meteorology,
  which is leakage dressed up as skill.
- **Excludes** the cumulative wind-speed counter, for the reason below.

## Data-quality problem: why `Iws` was dropped

`Iws` is a **cumulative counter**, not an instantaneous wind speed. Readings
accumulate from roughly 0.45 up to 585. Three checks on the raw column:

- Within a single year, **18.9%** of consecutive one-hour differences are
  negative (8,283 rows) — impossible for a real wind speed.
- About **1,301** of those are obvious counter resets (difference < −20).
- Taking the difference anyway yields physically impossible values such as
  −489 m/s.

So the true instantaneous wind speed cannot be recovered reliably, and this
model does not use the column.

The cost of that decision is worth stating plainly: **wind speed is the variable
you would most expect to matter, and it is the one variable this model cannot
use.** The page states this explicitly rather than omitting
the feature.

### The leak, measured rather than asserted

Feeding the raw cumulative reading in as though it were a wind speed raises
cross-validated R² from **0.533 to 0.555** — both figures are computed by
`scripts/train_model.py` on every run and printed to stdout, so this comparison
is reproducible rather than folklore.

The apparent improvement comes from the counter accumulating with elapsed time,
not from any physical dilution effect. A higher score obtained through leakage
is worse than a lower honest score, because it misleads everyone downstream.

## Licence and attribution

The data is published under CC BY 4.0, copyright UCI and the original authors.
This project's code is MIT licensed; the two are separately licensed.