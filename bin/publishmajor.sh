#!/usr/bin/env bash

bumpversion major --tag --commit
git push
flit publish