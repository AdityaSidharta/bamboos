#!/usr/bin/env bash

bumpversion minor --tag --commit
git push
flit publish