SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

export

python = 3.11

.PHONY: build-base
build-base:
	@python$(python) -m venv .venv

.PHONY: env
env: build-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt
