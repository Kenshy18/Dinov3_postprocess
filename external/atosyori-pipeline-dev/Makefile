.PHONY: install doctor engine-check test smoke clean

install:
	python -m pip install -e . --no-deps

doctor:
	python -m atosyori_postprocess doctor

copy-models:
	python -m atosyori_postprocess copy-models --source-root .. --replace

engine-check:
	python -m atosyori_postprocess engine-check

test:
	python -m unittest discover -s tests
	python -m compileall -q src tests tools

smoke:
	python -m atosyori_postprocess smoke --work-dir output/smoke --force

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
