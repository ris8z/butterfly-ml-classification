import os
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import albumentations as A


# constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SET_FOLDER = os.path.join(BASE_DIR, "./../data_set")

INPUT_IMAGES_FOLDER = os.path.join(DATA_SET_FOLDER, "images")
INPUT_CSV = os.path.join(DATA_SET_FOLDER, "train_set.csv")

OUTPUT_IMAGES_FOLDER = os.path.join(DATA_SET_FOLDER, "balance_augmented_images")
OUTPUT_CSV = os.path.join(DATA_SET_FOLDER, "train_set_augmented_balance.csv")

TOT_IMAGE_FOR_CLASS = 100
H, W = 150, 150


# functions
def get_tot_number_images_for_class(csv_path, col_name):
    df = pd.read_csv(csv_path)
    result = df[col_name].value_counts()
    return result.items(), df


def get_random_row(data_frame):
    return data_frame.sample(1).iloc[
        0
    ]  # sample give u a set of 1 element ln, iloc pick the first one


def load_and_resize_image(file_name, folder_path, height, width):
    path = os.path.join(folder_path, file_name)
    img = imread(path)

    if img is None:
        raise FileNotFoundError(f"Image not found : {path}")

    img_resized = resize(img, (height, width), anti_aliasing=True)
    return (img_resized * 255).astype(np.uint8)  # from 0-1 float -> 0-255 int


def data_augmentation_pipeline():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(std_range=(0.1, 0.2), noise_scale_factor=0.5, p=0.1),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )


def save_image(img, folder, filename):
    output_path = os.path.join(folder, filename)
    imsave(output_path, img)


def copy_orignal_images(df_label, seen_files_general, n, label, result_csv):
    for _, row in df_label.iterrows():
        filename = row["filename"]

        if filename in seen_files_general:
            continue
        seen_files_general.add(filename)

        copy_img = load_and_resize_image(filename, INPUT_IMAGES_FOLDER, H, W)
        copy_filename = f"original_({n},0)_{filename}"
        save_image(copy_img, OUTPUT_IMAGES_FOLDER, copy_filename)
        result_csv.append([copy_filename, label])


def generate_new_images(needed_image_n, df_label, n, label, result_csv):
    seen_files_random = set()
    for i in range(needed_image_n):

        try_n = 0
        filename = ""
        while try_n < 10:
            filename = get_random_row(df_label)["filename"]
            if filename in seen_files_random:
                try_n += 1
                continue
            else:
                seen_files_random.add(filename)
                break

        if try_n > 10:
            break

        origianl_img = load_and_resize_image(filename, INPUT_IMAGES_FOLDER, H, W)

        if origianl_img.ndim == 2:  # maybe not that usefull 
            origianl_img = np.stack((origianl_img,) * 3, axis=-1)

        new_img = data_augmentation_pipeline()(image=origianl_img)["image"]
        new_filename = f"augumented_({n},{i})_{filename}"
        save_image(new_img, OUTPUT_IMAGES_FOLDER, new_filename)
        result_csv.append([new_filename, label])


def images_generation(class_to_images_n, df, col_name):
    result_csv = []  # (filename, label) used to create the new csv file
    seen_files_general = set()

    for label, n in class_to_images_n:
        df_label = df[df[col_name] == label]  # filename, label
        needed_image_n = TOT_IMAGE_FOR_CLASS - n

        copy_orignal_images(df_label, seen_files_general, n, label, result_csv)

        if needed_image_n <= 0:
            continue

        generate_new_images(needed_image_n, df_label, n, label, result_csv)

    return result_csv


def save_csv(csv_data, filepath):
    df = pd.DataFrame(csv_data, columns=["filename", "label"])
    df.to_csv(filepath, index=False)


def setup():
    os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)

    # clear previous image's files
    for file_name in os.listdir(OUTPUT_IMAGES_FOLDER):
        file_path = os.path.join(OUTPUT_IMAGES_FOLDER, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # clear previous csv file
    with open(OUTPUT_CSV, "w") as _:
        pass


# body
def main():
    setup()
    class_to_images_n, df = get_tot_number_images_for_class(INPUT_CSV, "label")
    new_csv = images_generation(class_to_images_n, df, "label")
    save_csv(new_csv, OUTPUT_CSV)


if __name__ == "__main__":
    main()
