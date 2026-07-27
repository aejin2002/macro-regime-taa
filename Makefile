.PHONY: setup fetch build-signals evaluate run-all app test lint

UV := $(shell command -v uv 2> /dev/null)

setup:
ifdef UV
	uv venv
	uv pip install -e ".[dev]"
else
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"
endif

PY := .venv/bin/python

fetch:
	$(PY) -m macro_regime.cli fetch

build-signals:
	$(PY) -m macro_regime.cli build-signals

evaluate:
	$(PY) -m macro_regime.cli evaluate

run-all:
	$(PY) -m macro_regime.cli run-all

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src app tests
	.venv/bin/mypy src
