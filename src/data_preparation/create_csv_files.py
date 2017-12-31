"""

Tiny imagenet directory -> csv_label files

"""

import os
import glob
from src.common.paths import DATA_DIR, DATA_PATH

if __name__ == "__main__":

    print('Creating csv files')

    labels = [lab for lab in os.listdir(os.path.join(DATA_DIR, 'train')) if lab[0] != '.']
    label_dict = {lab: i for i, lab in enumerate(labels)}

    # train
    # train_csv = os.path.join(DATA_PATH, 'train.csv')
    # paths_gen = glob.glob((os.path.join(DATA_DIR, 'train'))+'/*/images/*JPEG')
    # with open(train_csv, 'w') as f:
    #     for path in paths_gen:
    #         f.write('{},{}\n'.format(os.path.abspath(path), label_dict[path.split('/')[-3]]))

    # validation
    # val_csv = os.path.join(DATA_PATH, 'val.csv')
    # print val_csv
    # paths_gen = glob.glob((os.path.join(DATA_DIR, 'val'))+'/images/*JPEG')
    #
    # label_dict_val = {line.split('\t')[0]: line.split('\t')[1]
    #                   for line in open(os.path.join(DATA_DIR, 'val/val_annotations.txt'))}
    #
    # with open(val_csv, 'w') as f:
    #     for path in paths_gen:
    #         wordnet_code = label_dict_val[path.split('/')[-1]]
    #         f.write('{},{}\n'.format(os.path.abspath(path), label_dict[wordnet_code]))

    # test
    test_csv = os.path.join(DATA_PATH, 'test.csv')
    print(test_csv)
    paths_gen = glob.glob((os.path.join(DATA_DIR, 'test'))+'/images/*JPEG')
    with open(test_csv, 'w') as f:
        for path in paths_gen:
            f.write('{},\n'.format(os.path.abspath(path)))

    # train_example
    # train_csv = os.path.join(DATA_PATH, 'train_example.csv')
    # print(train_csv)
    # paths_gen = glob.glob((os.path.join(DATA_DIR, 'train'))+'/*/images/*JPEG')
    # with open(train_csv, 'w') as f:
    #     # for path in paths_gen:
    #     for i in range(0, 100000, 100):
    #         path = paths_gen[i]
    #         f.write('{},{}\n'.format(os.path.abspath(path), label_dict[path.split('/')[-3]]))
