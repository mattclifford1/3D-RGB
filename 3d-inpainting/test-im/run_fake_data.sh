#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ MAKING TEST IMS ============"
python gen_fake_frames.py --config test_vid.yml --dir frames
python gen_fake_frames.py --config test_vid.yml --dir depth
python gen_fake_frames.py --config test_vid.yml --dir ldi
