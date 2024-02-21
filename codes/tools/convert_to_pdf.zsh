#!/bin/bash
for file in *.py; do
    enscript -E -q -Z -p - -f Courier10 ${file%.*}.py | ps2pdf - ${file%.*}.pdf
    #rm ${file%.*}.pdf
done
