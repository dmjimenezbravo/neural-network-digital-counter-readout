import os
from PIL import Image
import os, os.path
import os
from pathlib import Path

list_folders = []

easy_samples = open('easy_samples.txt', 'r')
easy_lines = easy_samples.readlines()
easy_files = dict()


difficult_files = open('difficult_samples_for_all.txt', 'r')
difficult_lines = difficult_files.readlines()
difficult_files = dict()


count = 0
for line in easy_lines:
    count += 1
    elem = line.strip().split()
    numeros = elem[1].split(",")
    easy_files[elem[0]] = numeros

count = 0

for line in difficult_lines:
    count += 1
    elem = line.strip().split()
    numeros = elem[1].split(",")
    difficult_files[elem[0]] = numeros

folders = dict()
folders["difficult"] = difficult_files
folders["easy"] = easy_files


for dir_name, file_dict in folders.items():

    for n in range(0, 20):
        if not os.path.exists(str(f'./{dir_name}/{n}')):
            os.makedirs(str(f'./{dir_name}/{n}'))

    for key,value in file_dict.items():

        im = Image.open(r""+key)
        width, height = im.size

        # Setting the points for cropped image
        width_n = width / 5
        imagelist = [im.crop((0, 0, width_n, height)), im.crop((width_n, 0, 2 * width_n, height)),
                     im.crop((2 * width_n, 0, 3 * width_n, height)), im.crop((3 * width_n, 0, 4 * width_n, height)),
                     im.crop((4 * width_n, 0, width, height))]


        for idx, img in enumerate(imagelist):
            test_image = img.resize((20, 32), Image.NEAREST)
            path, dirs, files = next(os.walk('./'+dir_name+'/'+str(value[idx])))
            file_count = len(files)
            save_name = './'+dir_name+'/'+value[idx] + '/' + str(file_count)+".jpg"
            test_image.save(save_name, "JPEG")

