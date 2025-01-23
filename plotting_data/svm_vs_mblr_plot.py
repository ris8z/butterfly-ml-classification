import pandas as pd
import matplotlib.pyplot as plt

mn_lr_accuracies = [0.8966733870967742, 0.4939516129032258, 0.8860887096774194]
svm_accuracies = [0.34526209677419356, 0.4002016129032258, 0.01310483870967742]

datasets = ["Simple Split (00)", "Augmented (01)", "Augmented Balanced (02)"]


plt.figure(figsize=(10, 6))
x = range(len(datasets))
plt.bar([p - 0.2 for p in x], mn_lr_accuracies, width=0.4, label="MobileNet + Logistic Regression")
plt.bar([p + 0.2 for p in x], svm_accuracies, width=0.4, label="SVM")
plt.xticks(x, datasets)
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.title("Overall Accuracy Comparison Across Datasets")
plt.legend()
plt.tight_layout()
plt.show()
