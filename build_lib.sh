#!/usr/bin/env bash

VERSION=$(sed -nE "s/^ *version='(.*)'.*$/\1/p" setup.py)
echo "last detected version is $VERSION"

echo 'Building distribution archive'
python3.11 -m build || exit

echo 'Reinstall fresh version of lib'
pip uninstall -y codexp
pip install ./dist/codexp-0.0.0-py3-none-any.whl
