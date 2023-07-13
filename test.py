# remove all images which number is more than 32000

import os

for file in os.listdir("data\\img_align_celeba"):
    if int(file.split(".")[0]) > 32000:
        os.remove(os.path.join("data\\img_align_celeba", file))
