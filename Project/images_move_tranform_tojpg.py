import os
import shutil
from PIL import Image
import pandas as pd


# DATA_PATH = '/kaggle/input/german-traffic-sign-detection-benchmark-gtsdb'
DATA_PATH = 'DATA/GTSDB/'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'TestIJCNN2013/TestIJCNN2013Download')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'TrainIJCNN2013/TrainIJCNN2013')
current_train_dir = TRAIN_DATA_PATH
new_train_dir = os.path.join(DATA_PATH, 'Train')
os.makedirs(new_train_dir, exist_ok=True)


file_names_in = []
for root, dirs, files in os.walk(current_train_dir):
    for file in files:
        if file.endswith(".ppm") and file not in file_names_in:
            # Open the .ppm image
            img = Image.open(os.path.join(root, file))
            # Convert the image to .jpg
            rgb_im = img.convert('RGB')
            # Save the .jpg image to the new directory
            rgb_im.save(os.path.join(new_train_dir, file.replace(".ppm", ".jpg")))
            file_names_in.append(file)

gt2 = pd.read_csv(os.path.join(DATA_PATH, 'gt.txt'), delimiter=";", header=None)
gt2.iloc[:,0] = gt2.iloc[:,0].str[:-4] + '.jpg'
gt2.to_csv(os.path.join(DATA_PATH, 'gt_jpg.txt'), sep=";", header=None, index=False)
