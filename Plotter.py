import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns




LABELS = ["Normal", "Fraud"]
threshold_fixed = 5


def visualize_data_set(data_set):
    count_classes = pd.value_counts(data_set['Class'], sort=True)
    count_classes.plot(kind='bar', rot=0)
    plt.xticks(range(2), LABELS)
    plt.title("Frequency by observation number")
    plt.xlabel("Class")
    plt.ylabel("Number of Observations")
    plt.show()


def visualize_transaction_amount_data(normal_df,fraud_df):
    bins = np.linspace(200, 2500, 100)
    plt.hist(normal_df.Amount, bins, alpha=1, normed=True, label='Normal')
    plt.hist(fraud_df.Amount, bins, alpha=0.6, normed=True, label='Fraud')
    plt.legend(loc='upper right')
    plt.title("Amount by percentage of transactions (transactions \$200+)")
    plt.xlabel("Transaction amount (USD)")
    plt.ylabel("Percentage of transactions (%)");
    plt.show()


def visualize_data_by_hour(normal_df,fraud_df):
    bins = np.linspace(0, 48, 48)  # 48 hours
    plt.hist((normal_df.Time / (60 * 60)), bins, alpha=1, normed=True, label='Normal')
    plt.hist((fraud_df.Time / (60 * 60)), bins, alpha=0.6, normed=True, label='Fraud')
    plt.legend(loc='upper right')
    plt.title("Percentage of transactions by hour")
    plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
    plt.ylabel("Percentage of transactions (%)");
    plt.show()


def visualize_transactions_amount_vs_hour(normal_df,fraud_df):
    plt.scatter((normal_df.Time / (60 * 60)), normal_df.Amount, alpha=0.6, label='Normal')
    plt.scatter((fraud_df.Time / (60 * 60)), fraud_df.Amount, alpha=0.9, label='Fraud')
    plt.title("Amount of transaction by hour")
    plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
    plt.ylabel('Amount (USD)')
    plt.legend(loc='upper right')
    plt.show()


def visualize_model_loss(history):
    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.ylim(ymin=0.70,ymax=1)
    plt.show()


def roc_curve_check(error_df):
    false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
    roc_auc = auc(false_pos_rate, true_pos_rate, )

    plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], linewidth=5)

    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def recall_vs_precision(precision_rt, recall_rt, threshold_rt):
    plt.plot(recall_rt, precision_rt, linewidth=5, label='Precision-Recall curve')
    plt.title('Recall vs Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def recall_vs_prescision_for_different_threshold(precision_rt, recall_rt, threshold_rt):
    plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
    plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()


def reconstruction_error_vs_threshold_check(error_df):
    threshold_fixed = 5
    groups = error_df.groupby('True_class')
    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                label="Fraud" if name == 1 else "Normal")
    ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for different classes")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.show()


def visualize_confusion_matrix(error_df):
    col_list = ["cerulean", "scarlet"]  # https://xkcd.com/color/rgb/
    # Set color_codes to False there is a bug in Seaborn 0.9.0 -- https://github.com/mwaskom/seaborn/issues/1546
    sns.set(style='white', font_scale=1.75, palette=sns.xkcd_palette(col_list), color_codes=False)

    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()