### code inspiration from here jjwww.kaggle.com/code/edumisvieramartin/butterfly-multiclass-image-classification-cnn?fbclid=PAY2xjawH-lYpleHRuA2FlbQIxMAABphOFOlt8ujLEsNu917VVCqNorFgA30HmBmS5qMwN1RXmm9UaD3v4dh5UfQ_aem_KPLESFlRnjzMOJu2Bj--xw
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SET_FOLDER = os.path.join(BASE_DIR, "./../data_set")
DATA_SET_FILE = "01_origianl_data_set.csv"

def plot_data_distribution_from_csv(path_csv, title_name):
    df = pd.read_csv(path_csv)
    class_counts = df['label'].value_counts().sort_index()

    plt.figure(figsize=(14, 8))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title(f"Distribution of Butterfly Classes ({title_name}), (tot img = {len(df)})")
    plt.xlabel('Butterfly Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.show()  # Now display the plot
    plt.close()  # Close the plot to free memory


def main():
    print("Which data set do you want to plot?")
    print("-1 = original, 00, 01, 02")
    code = input().strip()

    if code == "-1":
        plot_data_distribution_from_csv(os.path.join(DATA_SET_FOLDER, "01_origianl_data_set.csv"), 'origianl dataset -1')
    elif code == "00":
        plot_data_distribution_from_csv(os.path.join(DATA_SET_FOLDER, "train_set.csv"), 'after split 00')
    elif code == "01":
        plot_data_distribution_from_csv(os.path.join(DATA_SET_FOLDER, "train_set_augmented.csv"), 'augmented 01')
    elif code == "02":
        plot_data_distribution_from_csv(os.path.join(DATA_SET_FOLDER, "train_set_augmented_balance.csv"), 'balanced and augmented 02')
    else:
        print("No valid code")


if __name__ == "__main__":
    main()
