from PIL import Image
from sys import argv, exit
from os import listdir, system, chdir
from os.path import exists, join
from time import ctime, sleep

def main():
    if len(argv) != 4:
        print("Usage: python change_rs.py folder_with_images/ width height")
        exit(1)
    
    folder = argv[1]
    resolution = (int(argv[2]), int(argv[3]))

    if not exists(folder):
        print("Folder does not exist")

    for img in sorted(listdir(folder)):

        img_path = join(folder, img)
        image = Image.open(img_path)
        fname = f"data_set/sanay2/{ctime()}.jpg"

        new_image = image.resize(resolution)

        new_image.save(fname, 'PNG')
        print(f"Saving image to {fname}")
        sleep(1)






if __name__ == '__main__':
    main()