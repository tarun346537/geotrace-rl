# Local CPU workflows. Override PYTHON with the selected virtualenv interpreter.
PYTHON ?= python
OUTPUT_DIR ?= outputs

.DEFAULT_GOAL := check
.PHONY: lint typecheck test coverage sample demo check

lint:
	"$(PYTHON)" -m ruff check .
	"$(PYTHON)" -m ruff format --check .

typecheck:
	"$(PYTHON)" -m mypy geotrace monitoring scripts

# The test alias includes coverage so `make check test coverage` runs pytest once.
test: coverage

coverage:
	"$(PYTHON)" -m pytest tests -q --cov=geotrace --cov=monitoring --cov-report=term-missing --cov-report=xml

sample:
	"$(PYTHON)" scripts/generate_sample_data.py --output "$(OUTPUT_DIR)/sample-traces.jsonl"

demo:
	"$(PYTHON)" -m geotrace.demo --output "$(OUTPUT_DIR)/demo.json"

check: lint typecheck test sample demo
