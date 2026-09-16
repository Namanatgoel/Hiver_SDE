# Hiver SDE Intern — Makefile
# Usage: make <target>

CONDA_ENV  := hiver_sde
PYTHON     := conda run -n $(CONDA_ENV) python
PIP        := conda run -n $(CONDA_ENV) pip

.PHONY: help env torch data smoke test eval report run clean

help:
	@echo "Targets:"
	@echo "  env     — create conda env from environment.yml"
	@echo "  torch   — install cu130 nightly torch into the env"
	@echo "  data    — symlink dataset into data/raw/"
	@echo "  smoke   — Phase 0 gateway smoke test (3 calls → 0 warm calls)"
	@echo "  test    — run full pytest suite"
	@echo "  eval    — run evaluation on full golden set"
	@echo "  report  — regenerate REPORT.md from reports/eval_results.json"
	@echo "  run     — run the agent on a single message (for demos)"
	@echo "  clean   — remove __pycache__ and .pytest_cache"

env:
	conda env create -f environment.yml
	@echo ""
	@echo "Next step: conda activate $(CONDA_ENV) && make torch"

torch:
	bash scripts/install_torch.sh

data:
	$(PYTHON) scripts/00_download_data.py

smoke:
	$(PYTHON) scripts/smoke_gateway.py

test:
	conda run -n $(CONDA_ENV) pytest tests/ -v --tb=short

eval:
	$(PYTHON) scripts/06_evaluate.py

report:
	$(PYTHON) scripts/09_build_report.py

run:
	$(PYTHON) -c "from src.agent import demo; demo()"

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
