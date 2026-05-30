SHELL := /bin/sh
DC ?= docker compose

.PHONY: build up down logs shell-api shell-dashboard test

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
