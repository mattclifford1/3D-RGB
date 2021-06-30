import os
import argparse
import shutil
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test_vid.yml',help='Configure of post processing')
parser.add_argument('--dir', type=str, default='frames')
args = parser.parse_args()
config = yaml.safe_load(open(args.config, 'r'))

# fake_nums = range(int(config['num_frames']))
fake_nums = [0, 100, 200]
postfix = '-fake'
new_dir = args.dir+postfix
os.makedirs(new_dir, exist_ok=True)
files = os.listdir(args.dir)
for file in files:
    format = file.split('.')[-1]
    orig_file = os.path.join(args.dir, file)
    for i in fake_nums:
        fake_file = os.path.join(new_dir, str(i)+'.'+format)
        shutil.copyfile(orig_file, fake_file)
