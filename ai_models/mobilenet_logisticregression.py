import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
import time

# constants 
IMG_H = 150
IMG_W = 150

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), "./../data_set")

#######################################################################
# chose which type of dataset to use (check docs to create the data)  #
#######################################################################

# base split case (code 00)
#TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set.csv")
#TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_00.joblib")

# tipled the data set with data augmentation case (code 01)
#TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set_augmented.csv")
#TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "augmented_images")
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_01.joblib")

# data augmented and balanced (code 02)
TRAIN_CSV = os.path.join(DATASET_FOLDER, "train_set_augmented_balance.csv")
TRAIN_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "balance_augmented_images")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "./../trained/mn_lr_02.joblib")

#######################################################################


TEST_CSV = os.path.join(DATASET_FOLDER, "test_set.csv")
TEST_IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")


# functions 
def load_dataframe(csv_path):
    return pd.read_csv(csv_path)


def create_data_generators_train(train_df):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_IMAGES_FOLDER,
        x_col="filename",
        y_col="label",
        target_size=(IMG_H, IMG_W),
        batch_size=32,
        class_mode="categorical",
    )
    return train_generator


def create_data_generators_test(test_df):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=TEST_IMAGES_FOLDER,
        x_col="filename",
        y_col="label",
        target_size=(IMG_H, IMG_W),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )
    return test_generator


def extract_features(generator, model):
    features, labels = [], []
    for X_batch, y_batch in generator:
        features_batch = model.predict(X_batch)
        features.append(features_batch)
        if y_batch is not None:
            labels.append(y_batch)
        if len(features) * generator.batch_size >= generator.samples:
            break
    return np.vstack(features), np.vstack(labels)


def display_metrics(label_encoder, y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    print(f"{'CLASS NAME':<30}|{'PRECISION':<10}|{'RECALL':<10}|{'F1-SCORE':<10}|{'SUPPORT':<10}|{'ACCURACY':<10}")

    for key in report:
        # jump unwanted metrics 
        if key in ['accuracy', 'macro avg', 'weighted avg']:
            continue

        class_name = key
        data = report[key]

        p, r, f1, s = data['precision'], data['recall'], data['f1-score'], int(data['support'])

        # evalute accuracy for each class 
        class_idx = np.where(label_encoder.classes_ == class_name)[0][0]
        idxs = np.where(y_true == class_idx)[0]
        total_samples = len(idxs)
        class_accuracy = 0.0
        if total_samples > 0:
            correct_predictions = (y_true[idxs] == y_pred[idxs]).sum()
            class_accuracy = correct_predictions / total_samples

        print(f"{class_name:<30}|{p:<10.4f}|{r:<10.4f}|{f1:<10.4f}|{s:<10}|{class_accuracy:<10.4f}")


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    train_time = None

    if os.path.exists(MODEL_PATH):
        print("Loading the saved model...")
        model, label_encoder, scaler = load(MODEL_PATH)
    else:
        start_time = time.time()

        print("Loading the dataframes...")
        train_df = load_dataframe(TRAIN_CSV)

        train_generator = create_data_generators_train(train_df)

        print("Loading MobileNet for feature extraction...")
        base_model = MobileNet(weights="imagenet", include_top=False, pooling="avg")

        print("Extracting training features...")
        X_train, y_train = extract_features(train_generator, base_model)

        print("Encoding the labels...")
        label_encoder = LabelEncoder()
        label_encoder.fit(list(train_generator.class_indices.keys()))
        y_train_encoded = np.argmax(y_train, axis=1)

        print("Normalizing features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        print("Training the model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train_encoded)

        print(f"Saving the model to {MODEL_PATH}...")
        dump((model, label_encoder, scaler), MODEL_PATH)

        end_time = time.time()
        train_time = end_time - start_time

    # Test the model
    print("Loading the test dataframe...")
    test_df = load_dataframe(TEST_CSV)
    test_generator = create_data_generators_test(test_df)

    print("Loading MobileNet for feature extraction...")
    base_model = MobileNet(weights="imagenet", include_top=False, pooling="avg")

    print("Extracting test features...")
    X_test, y_test = extract_features(test_generator, base_model)

    print("Encoding the labels...")
    y_test_encoded = np.argmax(y_test, axis=1)

    print("Normalizing test features...")
    X_test = scaler.transform(X_test)

    print("Testing the model...")
    y_pred = model.predict(X_test)

    print("\n\n\nModel Total accuracy:", accuracy_score(y_test_encoded, y_pred))

    print("\n\n\nMetrics based on each class:")
    display_metrics(label_encoder, y_test_encoded, y_pred)

    if train_time is not None:
        print(f"\n\n\nTrain time: {train_time}")


if __name__ == "__main__":
    main()
