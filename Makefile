.PHONY: help install install-dev lint lint-fix type-check security-code security-deps security test test-fast test-parallel docker-build docker-run docker-scan ci hooks-install hooks-run clean

IMAGE_NAME := medical-imaging-ai
SRC := src
PYTHON := python3

# ─── Help ────────────────────────────────────────────────────────────────────
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─── Setup ───────────────────────────────────────────────────────────────────
install: ## Install dependencies (CPU-only PyTorch, DICOM support)
	$(PYTHON) -m pip install --upgrade pip
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
	pip install ruff mypy bandit pip-audit pre-commit

install-dev: install ## Install with dev extras
	pip install pytest pytest-cov pytest-xdist

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check $(SRC) tests/
	ruff format --check $(SRC) tests/

lint-fix: ## Auto-fix lint issues
	ruff check --fix $(SRC) tests/
	ruff format $(SRC) tests/

type-check: ## Run mypy static type checking
	mypy $(SRC) --ignore-missing-imports --strict-optional

# ─── Security ─────────────────────────────────────────────────────────────────
security-code: ## Run bandit SAST on source
	bandit -r $(SRC) -ll -ii --format json -o bandit-report.json || true
	bandit -r $(SRC) -ll -ii

security-deps: ## Audit dependencies for CVEs
	pip-audit --desc --format json -o pip-audit-report.json || true
	pip-audit --desc

security: security-code security-deps ## Run all security checks

# ─── Testing ──────────────────────────────────────────────────────────────────
test: ## Run test suite with coverage (CNN/Grad-CAM unit tests)
	pytest tests/ -v --cov=$(SRC) --cov-report=term-missing --cov-report=xml --cov-fail-under=80

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v -x

test-parallel: ## Run tests in parallel
	pytest tests/ -v -n auto --cov=$(SRC)

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build: ## Build production Docker image
	docker build --target production -t $(IMAGE_NAME):latest .

docker-run: ## Run container locally
	docker run --rm -p 8080:8080 $(IMAGE_NAME):latest

docker-scan: ## Scan image for vulnerabilities (Trivy)
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy:latest image --severity HIGH,CRITICAL $(IMAGE_NAME):latest

# ─── CI Pipeline ──────────────────────────────────────────────────────────────
ci: lint type-check security test ## Run full CI pipeline locally

# ─── Pre-commit ───────────────────────────────────────────────────────────────
hooks-install: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

hooks-run: ## Run all hooks against all files
	pre-commit run --all-files

# ─── Utilities ────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -f bandit-report.json pip-audit-report.json coverage.xml
