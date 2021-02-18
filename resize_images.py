import os
from PIL import Image, ImageFile
from time import ctime, sleep

PERSON = 'sanay'
DIRECTORY_PATH = "data_set/"


def crop(image_path, coords, saved_location):
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)

    cropped_image.save(saved_location, 'PNG')
    if image_obj.getbbox():
        pass

        # if os.path.getsize(saved_location) < 46000 or os.path.getsize(saved_location) > 68000:
        #     os.remove(saved_location)
        #     print(f"removed {saved_location}")
    else:
        print(f"{saved_location} has no bbox")

def main():

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img_dir = os.path.join(DIRECTORY_PATH, PERSON)
    
    for img in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img) 
        fname = f'data_set/sanay1/{ctime()}.jpg'
        crop(img_path, (614, 261, 1300, 910), fname)
        print(f"Saved {img_path} to {fname}")
        sleep(1)
        


if __name__ == '__main__':
    main()