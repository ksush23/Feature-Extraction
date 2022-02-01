from PIL import Image
import numpy as np
import pathlib

data_dir_apple = pathlib.Path("./dogs_test/Blenheim_spaniel")
list_apple = np.array([item for item in data_dir_apple.glob('*')])

data_dir_banana = pathlib.Path("./dogs_test/Japanese_spaniel")
list_banana = np.array([item for item in data_dir_banana.glob('*')])

# data_dir_orange = pathlib.Path("./dataset_test/orange")
# list_orange = np.array([item for item in data_dir_orange.glob('*')])

for i in range(len(list_apple)):
    img = Image.open(list_apple[i])
    img = img.resize((277, 277))
    img = img.convert("RGB")
    img.save("Blenheim_spaniel" + str(i) + ".jpg")

for i in range(len(list_banana)):
    img = Image.open(list_banana[i])
    img = img.resize((277, 277))
    img = img.convert("RGB")
    img.save("Japanese_spaniel" + str(i) + ".jpg")

# for i in range(len(list_orange)):
#     img = Image.open(list_orange[i])
#     img = img.resize((277, 277))
#     img = img.convert("RGB")
#     img.save("orange" + str(i) + ".jpg")
