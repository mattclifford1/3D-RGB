#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ RUNNING TESTS ============"
python test_gen_im.py --control control-ims --test video-ims
# python test_gen_im.py --control control-im-crop --test video-ims
