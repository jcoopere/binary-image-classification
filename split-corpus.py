import os
import sys
import random
import shutil

def copyfile(src_dir, dest_dir, file):
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  shutil.copyfile(os.path.join(src_dir, file), os.path.join(dest_dir, file))

def create_split_datasets(dataset_dir, target_dir, validate_split_percent, test_split_percent):
  dirs = []
  for (dirpath, dirnames, filenames) in os.walk(dataset_dir):
    dirs.extend(dirnames)
    break

  train_split_percent = 100 - (validate_split_percent + test_split_percent)

  target_train_dir = os.path.join(target_dir, 'train')
  target_validate_dir = os.path.join(target_dir, 'validate')
  target_test_dir = os.path.join(target_dir, 'test')

  for dir in dirs:
    subdir = os.path.join(dataset_dir, dir)
    files = os.listdir(subdir)
    random.shuffle(files)
    i1 = int(len(files) * (train_split_percent / 100))
    i2 = int(len(files) * ((train_split_percent + validate_split_percent) / 100))
    train, validate, test = files[:i1], files[i1:i2], files[i2:]
    label = dir

    for file in train:
      copyfile(subdir, os.path.join(target_train_dir, label), file)
    for file in validate:
      copyfile(subdir, os.path.join(target_validate_dir, label), file)
    for file in test:
      copyfile(subdir, os.path.join(target_test_dir, label), file)

corpus_dir = '/home/cdsw/sorted-images'
target_dir = '/home/cdsw/split-sorted-images'
validate_split_percent = 20
test_split_percent = 20

print('Splitting full corpus ({}) into train ({}%), validate ({}%), and test ({}%) sets...'.format(corpus_dir, (100 - (validate_split_percent + test_split_percent)), validate_split_percent, test_split_percent))
create_split_datasets(corpus_dir, target_dir, validate_split_percent, test_split_percent)
print('Splitting complete. Train, validate, and test sets available in directory: {}'.format(target_dir))
