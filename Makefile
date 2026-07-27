.PHONY: setup fetch build-signals evaluate app test lint

UV := $(shell command -v uv 2> /dev/null)

setup:
ifdef UV
	uv venv
	uv pip install -e ".[dev]"
else
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"
endif

fetch:
	python -m macro_regime.cli fetch

build-signals:
	python -m macro_regime.cli build-signals

evaluate:
	python -m macro_regime.cli evaluate

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check src app tests
	mypy src
