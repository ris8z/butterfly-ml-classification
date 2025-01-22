from collections import Counter
from joblib import load
import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import time

# constants for testing
IMG_H = 64
IMG_W = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SET_FOLDER = os.path.join(BASE_DIR, "./../data_set")

#######################################################################
# chose which type of dataset to use (check docs to create the data)  #
#######################################################################

# base split case (code 00)
#TRAIN_IMAGES_FOLDER = os.path.join(DATA_SET_FOLDER, "images")
#TRAIN_CSV = os.path.join(DATA_SET_FOLDER, "train_set.csv")
#MODEL_PATH = os.path.join(BASE_DIR, "./../trained/svm_0.joblib")

# tipled the data set with data augmentation case (code 01)
#TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set_augmented.csv")
#TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "augmented_images")
#MODEL_PATH = os.path.join(BASE_DIR, "./../trained/svm_1.joblib")

# balanced augmented dataset (you need to run 05_balancing_data.py to create these images)
TRAIN_IMAGES_FOLDER = os.path.join(DATA_SET_FOLDER, "balance_augmented_images")
TRAIN_CSV = os.path.join(DATA_SET_FOLDER, "train_set_augmented_balance.csv")
MODEL_PATH = os.path.join(BASE_DIR, "./../trained/svm_2.joblib")

#######################################################################

TEST_IMAGES_FOLDER = os.path.join(DATA_SET_FOLDER, "images")
TEST_CSV = os.path.join(DATA_SET_FOLDER, "test_set.csv")

RANDOM_GENERATION_SEED = 33

# functions
def load_img_from_csv(csv_path, images_folder):
    data = pd.read_csv(csv_path)
    X, Y = [], []

    for _, row in data.iterrows():
        current_img_name = row["filename"]
        current_img_label = row["label"]
        try:
            img = imread(os.path.join(images_folder, str(current_img_name)))

            # all the images in the images folder are not preprocessed so we need to resize them
            if images_folder == os.path.join(DATA_SET_FOLDER, "images"):
                img = resize(
                    img, (IMG_H, IMG_W), anti_aliasing=True
                )  # bc the testing data are not resized yet

            X.append(img.flatten())
            Y.append(current_img_label)
        except Exception as e:
            print(f"Something went wrong during the load img from csv process: {e}")

    return np.array(X), np.array(Y)


def display_accuracy_for_each_class(class_lst, expected_label_lst, predicted_label_lst):
    if class_lst is None or expected_label_lst is None:
        raise Exception("Something wrong in display all the accuracy")

    d = {}  # key: class -> value: accuracy

    for number, label in enumerate(class_lst):
        # list of index where expected label lst is equal to the current one
        idxs = np.where(expected_label_lst == number)[0]
        total_samples = len(idxs)
        if total_samples > 0:
            # how many indexes of idxs share the same value between predicted and expected
            correct_predictions = (
                expected_label_lst[idxs] == predicted_label_lst[idxs]
            ).sum()
            d[label] = correct_predictions / total_samples
        else:
            d[label] = 0.0

    print(f"{'CLASS NAME':<30} {'ACCURACY':<10} {'TOTAL SAMPLES':<15}")
    for label, acc in d.items():
        number = list(class_lst).index(label)
        total_samples = len(np.where(expected_label_lst == number)[0])
        print(f"{label:<30} {acc:<10.4f} {total_samples:<15}")

# body
def main():
    start_time = time.time()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("Loading the saved model...")

        model, label_encoder, scaler = load(MODEL_PATH)
    else:
        print("Loading... Training set data")
        X_train, Y_train = load_img_from_csv(TRAIN_CSV, TRAIN_IMAGES_FOLDER)

        print("Encoding the labels ...")
        label_encoder = LabelEncoder()
        Y_train_encoded = label_encoder.fit_transform(Y_train)

        print("Normalize Features ...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        print("Training the model ...")
        model = SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_GENERATION_SEED
        )
        model.fit(X_train, Y_train_encoded)

        # Save the model along with the LabelEncoder and StandardScaler
        from joblib import dump

        dump((model, label_encoder, scaler), MODEL_PATH)
        print(f"Model saved as '{MODEL_PATH}'")

    print("Loading... Testing set data")
    X_test, Y_test = load_img_from_csv(TEST_CSV, TEST_IMAGES_FOLDER)

    print("Encoding the labels ...")
    # Use the saved LabelEncoder
    Y_test_encoded = label_encoder.transform(Y_test)

    print("Normalize Features ...")
    # Use the saved StandardScaler
    X_test = scaler.transform(X_test)

    print("Testing the model ...")
    Y_pred = model.predict(X_test)

    print("\n\n\nModel accuracy:", accuracy_score(Y_test_encoded, Y_pred))
    print(
        "\n\n\nClassification Report: (precision, recall, F1-score)\n",
        classification_report(
            Y_test_encoded, Y_pred, target_names=label_encoder.classes_
        ),
    )

    print("\n\n\nAccuracy for each class")
    display_accuracy_for_each_class(label_encoder.classes_, Y_test_encoded, Y_pred)

    end_time = time.time()
    print(f"\n\n\nExecution time: {end_time - start_time}")

    print("Predicted class distribution:", Counter(Y_pred))


if __name__ == "__main__":
    main()
