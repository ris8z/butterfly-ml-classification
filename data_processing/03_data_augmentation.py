import os
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm
import albumentations as A

# Constants
IMG_H = 64
IMG_W = 64
AUG_MULTIPLIER = 3  # how many new image for one original 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "./../data_set")

TRAIN_CSV = os.path.join(BASE_DIR, 'train_set.csv')
AUGMENTED_CSV = os.path.join(BASE_DIR, 'train_set_augmented.csv')
IMAGES_INPUT_FOLDER = os.path.join(BASE_DIR, 'images')
IMAGES_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'augmented_images')


def data_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  
        A.Rotate(limit=15, p=0.5),
        A.GaussNoise(p=0.5),  
        A.RandomBrightnessContrast(p=0.5),  
    ])

def load_and_resize_image(img_name):
    img_path = os.path.join(IMAGES_INPUT_FOLDER, img_name)
    img = imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_resized = resize(img, (IMG_H, IMG_W), anti_aliasing=True)
    return (img_resized * 255).astype(np.uint8) 

def save_image(img, filename):
    output_path = os.path.join(IMAGES_OUTPUT_FOLDER, filename)
    imsave(output_path, img)

def augment_and_save(csv_path):
    data = pd.read_csv(csv_path)  
    augmented_data = []  
    augmenter = data_augmentation_pipeline()  

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing images"):
        img_name = row['filename']
        label = row['label']

        try:
            # load img
            img_resized = load_and_resize_image(img_name)

            # save img
            original_filename = f"original_{img_name}"
            save_image(img_resized, original_filename)
            augmented_data.append([original_filename, label])

            # generate new imgs 
            for i in range(AUG_MULTIPLIER):
                augmented = augmenter(image=img_resized)['image']
                augmented_filename = f"aug_{i}_{img_name}"
                save_image(augmented, augmented_filename)
                augmented_data.append([augmented_filename, label])

        except Exception as e:
            print(f"Errore durante l'elaborazione di {img_name}: {e}")

    # save new csv file
    augmented_df = pd.DataFrame(augmented_data, columns=["filename", "label"])
    augmented_df.to_csv(AUGMENTED_CSV, index=False)
    print(f"Augmented data saved to {AUGMENTED_CSV}")


def main():
    # check if outpufolder exist or create it
    os.makedirs(IMAGES_OUTPUT_FOLDER, exist_ok=True)
    # create and save new images
    augment_and_save(TRAIN_CSV)

if __name__ == "__main__":
    main()
