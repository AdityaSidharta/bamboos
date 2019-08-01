#!/usr/bin/env bash

bumpversion patch --tag --commit
git push
flit publish