#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ MAKING TEST DATA ============"
cd ..
python construct_vid.py --config test-im/test.yml

cd test-im
echo "============ RUNNING TESTS ============"
python test_gen_im.py --control control-ims --test video-ims
