import cv2
import argparse
import os


def compare_images(control_dir, test_dir):
    fail_count = 0
    test_files = os.listdir(test_dir)
    for test_file in test_files:
        control_im = cv2.imread(os.path.join(control_dir, test_file))
        test_im = cv2.imread(os.path.join(test_dir, test_file))
        if (control_im == test_im).all():
            print('++++++++ '+test_file+' PASS ++++++++')
        else:
            print('-------- '+test_file+' FAIL --------')
            fail_count += 1
    print(str(fail_count)+' fail cases')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=str, default='control-ims')
    parser.add_argument('--test', type=str, default='test-ims')
    args = parser.parse_args()

    compare_images(args.control, args.test)
