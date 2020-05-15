#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ RUNNING DATA TESTS ============"
python test_handling.py --config test.yml
