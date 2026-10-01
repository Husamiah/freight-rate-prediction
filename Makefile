.PHONY: all data eda run score figures report
PYTHON ?= python3
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
DATA ?= .

all: data eda run score figures report

data:
	$(PYTHON) -m src.prepare_data $(DATA)
run:
	$(PYTHON) -m src.run_all | tee reports/run_log.txt
score:
	$(PYTHON) score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
figures:
	$(PYTHON) -m src.make_figures
report:
	$(PYTHON) -m src.make_report

eda:
	$(PYTHON) -m src.eda | tee reports/eda_log.txt
