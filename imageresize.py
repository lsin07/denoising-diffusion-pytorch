from PIL import Image
import os
from tqdm import tqdm

origin_dir = './dataset/castle/'
dst_dir = './dataset/castle_128/'
filename = os.listdir(origin_dir)
for fn in tqdm(filename):
    img = Image.open(origin_dir + fn)
    img_resize = img.resize((128, 128))
    img_resize.save(dst_dir + fn)