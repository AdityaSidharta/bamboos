#!/usr/bin/env bash

git init
git remote add origin git@github.com:AdityaSidharta/bamboos.git
pyenv install 3.7.2
pyenv local 3.7.2
pip install pipenv
pipenv install --python 3.7.2 --dev
flit init
