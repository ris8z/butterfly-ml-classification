import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SET_FOLDER = os.path.join(BASE_DIR, "./../data_set")
OUTPUT_CSV = os.path.join(DATA_SET_FOLDER, "train_set_augmented_balance.csv")


def get_tot_number_images_for_class(csv_path, col_name):
    df = pd.read_csv(csv_path) 
    result = df[col_name].value_counts()
    return result.items(), df

def main():
    res, _ = get_tot_number_images_for_class(OUTPUT_CSV, 'label')

    tot = 0
    for label, n in res:
        tot += n
        print(f"{n} <- {label}")

    print(f"\n\n\n{tot} <- tot number of images")

if __name__ == "__main__":
    main()
 
