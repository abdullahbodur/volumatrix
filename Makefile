.PHONY: install test lint format clean build docs

install:
	pip install -e ".[dev,all]"
	pre-commit install

test:
	pytest tests/ --cov=volumatrix --cov-report=term-missing

lint:
	black --check .
	isort --check-only .
	flake8 .
	mypy volumatrix/

format:
	black .
	isort .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

docs:
	cd docs && make html 