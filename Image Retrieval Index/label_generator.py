import os
import argparse

parser = argparse.ArgumentParser(description='Train IRGen')
parser.add_argument('--data_dir', default='data', type=str, help='datasets path')
opt = parser.parse_args()
dir = opt.data_dir

# train data
train_dir = os.path.join(dir, 'images/train')
classes = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

with open('classes.txt', 'w') as file:
    for i, cls in enumerate(classes):
        file.write(f"{i+1} {cls}\n")

# clear file
with open(os.path.join(dir, 'image_class_labels.txt'), 'w') as file:
    file.write('')
with open(os.path.join(dir, 'images.txt'), 'w') as file:
    file.write('')
with open(os.path.join(dir, 'train_test_split.txt'), 'w') as file:
    file.write('')

cnt = 0
for k, cls in enumerate(classes):
    cls_dir = os.path.join(train_dir, cls)
    files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
    with open(os.path.join(dir, 'images.txt'), 'a') as file:
        for i, f in enumerate(files):
            file.write(f"{cnt+i+1} {os.path.join('train', cls, f)}\n")
    with open(os.path.join(dir, 'image_class_labels.txt'), 'a') as file:
        for i, f in enumerate(files):
            file.write(f"{cnt+i+1} {k+1}\n")
    with open(os.path.join(dir, 'train_test_split.txt'), 'a') as file:
        for i, f in enumerate(files):
            file.write(f"{cnt+i+1} 1\n")
    cnt += len(files)

# test data
test_dir = os.path.join(dir, 'images/valid')
test_classes = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
# should not have new classes # skip record classes
for cls in test_classes:
    try:
        k = classes.index(cls)
    except ValueError:
        print("validation classes should be in the train classes")
        break
    cls_dir = os.path.join(test_dir, cls)
    files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
    with open(os.path.join(dir, 'images.txt'), 'a') as file:
        for i, f in enumerate(files):
            file.write(f"{cnt+i+1} {os.path.join('valid', cls, f)}\n")
    with open(os.path.join(dir, 'image_class_labels.txt'), 'a') as file:
        for i, f in enumerate(files):
            file.write(f"{cnt+i+1} {k+1}\n")
    with open(os.path.join(dir, 'train_test_split.txt'), 'a') as file:
        for i, f in enumerate(files):
            file.write(f"{cnt+i+1} 0\n")
    cnt += len(files)