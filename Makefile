SHELL := /bin/sh
DC ?= docker compose

.PHONY: build up down logs shell-api shell-dashboard test train train-skip-data

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
