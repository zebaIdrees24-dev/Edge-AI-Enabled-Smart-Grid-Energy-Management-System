.PHONY: install demo research test lint clean

install:
	python -m pip install -e ".[dev,notebook]"

demo:
	smartgrid-ems demo

research:
	smartgrid-ems research --include-lstm --lstm-epochs 8

test:
	pytest -q

lint:
	ruff check src tests

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in map(Path, ['build', 'dist', '.pytest_cache', '.ruff_cache'])]"

