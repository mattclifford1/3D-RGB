#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ MAKING ORIG DATA ============"
python orig_construct_vid.py --config test-im/test_vid.yml
echo "============ MAKING FAKE DATA ============"
python gen_fake_frames.py --config test_vid.yml --dir frames
python gen_fake_frames.py --config test_vid.yml --dir depth
python gen_fake_frames.py --config test_vid.yml --dir ldi
