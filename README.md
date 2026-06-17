# Six-Model Satellite-Derived Bathymetry Benchmark

## Models

- Stumpf
- Log-linear
- Random Forest
- LightGBM
- Geo_RF
- Geo_LightGBM

## Data files

Place the required CSV files in `data/` as described in `data/README.md`.

## Installation

```bash
pip install -r requirements.txt
```

## Run the six models

```bash
python scripts/RunSixModels.py
```

## Generate the figures

```bash
python scripts/plot_figures_geo6_3x2_bigfont.py
```

The experiment uses 3816 matched samples, a fixed random seed of 222, 2000 training samples, and 1816 non-overlapping validation samples.
