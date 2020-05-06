#!/bin/sh

echo "============ DEPTH ESTIMATION ============"
python est_depth.py --config argument.yml
echo "============ LDI ESTIMATION ============"
python est_ldi.py --config argument.yml
