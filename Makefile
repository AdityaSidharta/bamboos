SHELL := /bin/bash

help:
	@echo "init - initialize the project"
	@echo "test - setup pyenv and pipenv"
	@echo "publish - setup pyenv and pipenv"
	@echo "bumpmajor - bump major version"
	@echo "bumpminor - bump minor version"
	@echo "bumppatch - bump patch version"


setup:
	bash bin/setup.sh
	pipenv shell

pylint:
	pylint src --reports=y

run: pylint
	bash bin/run.sh
