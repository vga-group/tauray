#!/bin/sh

cd "$(dirname "$0")"
pandoc -s MANUAL.md -o tauray_user_manual.pdf \
       --filter ./pandoc_table_image.py
