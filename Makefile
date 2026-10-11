# Fair Code - contributor task runner.  See CONTRIBUTING.md.
# Reproduces locally what CI runs (.github/workflows: audits.yml, lint.yml,
# build-explainers.yml) so you can catch failures before you push.

.DEFAULT_GOAL := help
PY := python3

.PHONY: help setup test test-fast coverage build-explainers favicons fix-explainer-count lint audits check

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS = ":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Install the package plus the dev tools (pytest, pytest-cov, pre-commit, ruff)
	$(PY) -m pip install -e ".[excel,parquet,proxy,mcp,benchmark]" pytest pytest-cov pre-commit ruff

test:  ## Run the full test suite (mirrors CI)
	$(PY) -m pytest tests/ -q

test-fast:  ## Run the suite without the slow model-training benchmark tests (CI still runs everything)
	$(PY) -m pytest tests/ -q -m "not slow"

coverage:  ## Run the full test suite with a faircode/ coverage report (informational only)
	$(PY) -m pytest tests/ -q --cov=faircode --cov-report=term-missing

build-explainers:  ## Regenerate explainer pages, data.js, sitemap, and OG images (dark + light)
	$(PY) scripts/build_explainers.py
	$(PY) scripts/generate_og_images.py

favicons:  ## Regenerate favicon.ico/PNGs and apple-touch-icon.png from logo.svg
	$(PY) scripts/generate_favicons.py

fix-explainer-count:  ## Correct stale "N explainers" mentions in README/CONTRIBUTORS/METRICS/ROADMAP
	$(PY) scripts/check_explainer_count.py --fix

lint:  ## Enforce the em-dash-free rule + check for broken doc links + ruff (mirrors the lint workflow)
	$(PY) scripts/check_em_dash.py
	$(PY) scripts/check_broken_links.py
	ruff check faircode scripts tests

audits:  ## Run every audit's unfair.py/fair.py + the CLI smoke tests (mirrors audits.yml's required run-audits job, plus profiler/benchmark-harness's CLI steps)
	@set -e; \
	for d in "COMPAS" "AI Fair Recruitment" "German Credit Lending" "Insurance Denial" \
	         "Benefits Denial" "Healthcare Readmission" "Tenant Screening"; do \
		echo "== $$d - unfair.py =="; \
		$(PY) "$$d/unfair.py"; \
		echo "== $$d - fair.py =="; \
		$(PY) "$$d/fair.py"; \
	done
	@echo "== CLI smoke test: faircode profile =="
	faircode profile "Insurance Denial/insurance.csv"
	faircode profile "Benefits Denial/adult.csv" --json > /dev/null
	@echo "== CLI smoke test: faircode benchmark =="
	@tmp=$$(mktemp -d); \
	faircode benchmark "German Credit Lending/audit.yaml" \
		--n-resamples 100 --n-permutations 100 --no-plots \
		--out "$$tmp"; \
	test -s "$$tmp/results_fairness.csv"; \
	test -s "$$tmp/results_performance.csv"; \
	echo "Benchmark harness smoke test passed: $$(wc -l < "$$tmp/results_fairness.csv") fairness rows written"; \
	rm -rf "$$tmp"

check: lint test audits  ## Run everything CI runs (lint + full test suite + audits/CLI smoke tests)
	@echo "All checks passed."
