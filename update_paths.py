import glob
import os
import random
import numpy as np
import shutil
import pathlib


source_dir = None

c_labels = fr'{source_dir}\yolov5_dataset\labels'
c_images = fr'{source_dir}\yolov5_dataset\images'

# files = glob.glob(f'{source_dir}/*.txt', recursive=True)
files = [f for f in glob.glob(f'{source_dir}/**/*.txt', recursive=True) if 'classes' not in f]
videos = [list(x) for x in np.array_split(np.array(files), int(len(files) / 16))]

all_files = [random.sample(range(0, len(x)), 2) for x in videos]
train_list, val_list = [v[x[0]] for x, v in zip(all_files, videos)], [v[x[1]] for x, v in zip(all_files, videos)]

# for file in files:
#     if 'classes.txt' in file:
#         continue
#     if int(pathlib.Path(file).parts[-1].split('_')[0]) < 888:
#         image_file = file[:-3] + 'png'
#         print(f"{int(pathlib.Path(file).parts[-1].split('_')[0])}, {image_file}")
#         os.remove(file)
#         os.remove(image_file)
# for file in bad_files:
#     if os.path.exists(pathlib.Path(file).parent):
#         print(f"Removing {pathlib.Path(file).parent}")
#         shutil.rmtree(pathlib.Path(file).parent)

# train_list = random.sample(list(file_list), int(len(file_list)*0.3))
# val_files = [i for i in file_list if i not in train_list]
# val_list = random.sample(list(val_files), int(len(val_files)))

print("Deleting old files")
old_files = glob.glob(f'{c_images}/**/*.*', recursive=True) + glob.glob(f'{c_labels}/**/*.*', recursive=True)
for file in old_files:
    print(f"\t{old_files.index(file)} / {len(old_files)}: {file}")
    os.remove(file)

print("Copying train split")
for label_file in train_list:
    # label_file = files[file]
    print(f"\t{train_list.index(label_file)} / {len(train_list)}: {label_file}")
    image_file = label_file[:-3] + 'png'
    if os.path.exists(label_file):
        if os.path.exists(image_file):
            shutil.copy(label_file, os.path.join(c_labels, 'train',
                                                 f"{pathlib.Path(label_file).parts[-2]}_{pathlib.Path(label_file).name}"))
            shutil.copy(image_file, os.path.join(c_images, 'train',
                                                 f"{pathlib.Path(image_file).parts[-2]}_{pathlib.Path(image_file).name}"))
print("Copying test split")
for label_file in val_list:
    # label_file = files[file]
    print(f"\t{val_list.index(label_file)} / {len(val_list)}: {label_file}")
    image_file = label_file[:-3] + 'png'
    if os.path.exists(label_file):
        if os.path.exists(image_file):
            shutil.copy(label_file, os.path.join(c_labels, 'val',
                                                 f"{pathlib.Path(label_file).parts[-2]}_{pathlib.Path(label_file).name}"))
            shutil.copy(image_file, os.path.join(c_images, 'val',
                                                 f"{pathlib.Path(image_file).parts[-2]}_{pathlib.Path(image_file).name}"))

# python train.py --img 640 --batch 4 --epochs 300 --data "yolov5_dataset\dataset.yaml" --weights yolov5l.pt