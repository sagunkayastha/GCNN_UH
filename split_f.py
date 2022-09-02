import os
import random
from tqdm import tqdm
import shutil

os.makedirs("training_data", exist_ok=True)
os.makedirs("training_data/training", exist_ok=True)
os.makedirs("training_data/validation", exist_ok=True)

random.seed(922022)

data_path = "CelebA-HQ-img"
all_files = os.listdir(data_path)
train_percent = 90


k = int(len(all_files) * train_percent // 100)
random.shuffle(all_files)
training_list = all_files[:k]
validation_list = all_files[k:]

# copy training
for file in tqdm(training_list):
    source = os.path.join(data_path, file)
    dest = os.path.join("training_data,training")
    shutil.copy(source, dest)
    
# copy validation
for file in tqdm(validation_list):
    source = os.path.join(data_path, file)
    dest = os.path.join("training_data,validation")
    
    shutil.copy(source, dest)