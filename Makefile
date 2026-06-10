SHELL := /bin/sh
DC ?= docker compose

.PHONY: build up down logs shell-api shell-dashboard test train train-skip-data monitor

build:
	$(DC) build

up:
	$(DC) up -d

down:
	$(DC) down

logs:
	$(DC) logs -f --tail=200

shell-api:
	$(DC) exec api /bin/sh

shell-dashboard:
	$(DC) exec dashboard /bin/sh

test:
	# Run pytest inside the api container image
	$(DC) run --rm api pytest

train:
	python scripts/train_all.py

train-skip-data:
	python scripts/train_all.py --skip-data

.PHONY: monitor
monitor:
	@echo "Starting Enterprise Observability Stack (Prometheus & Grafana)..."
	docker network create hpe_network || true
	docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
	@echo "Grafana running at http://localhost:3000"
	@echo "Prometheus running at http://localhost:9090"
