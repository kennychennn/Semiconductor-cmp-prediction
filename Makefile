PYTHON ?= python
DATA_DIR ?= data/raw/training
LABELS ?= data/raw/labels/CMP-training-removalrate.csv

.PHONY: help install data clean features eda train test

help:
	@echo "Available commands:"
	@echo "  make install   Install Python dependencies"
	@echo "  make data      Print raw-data summary"
	@echo "  make clean     Clean removal-rate labels"
	@echo "  make features  Build model-ready features"
	@echo "  make eda       Generate exploratory figures"
	@echo "  make train     Train the XGBoost model"
	@echo "  make test      Run syntax and import checks"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.make_dataset --data-dir "$(DATA_DIR)" --labels "$(LABELS)"

clean:
	$(PYTHON) -m src.preprocess.clean_data --data-dir "$(DATA_DIR)" --labels "$(LABELS)"

features:
	$(PYTHON) -m src.preprocess.build_features --data-dir "$(DATA_DIR)" --labels "$(LABELS)" --output data/processed/cmp_features.csv

eda:
	$(PYTHON) -m src.visualization.visualize --data-dir "$(DATA_DIR)" --labels "$(LABELS)" --output-dir reports/figures/eda

train:
	$(PYTHON) src/model/xgboost_model.py

test:
	$(PYTHON) -m compileall -q src
	$(PYTHON) -c "from src.data.make_dataset import load_sensor_data; from src.preprocess.build_features import build_features; from src.visualization.visualize import run_eda; print('Import checks passed')"
