from sklearn.metrics import confusion_matrix, precision_recall_curve



def split_data_set(data_Set):
    normal_df = data_Set[data_Set.Class == 0]  # save normal_df observations into a separate df
    fraud_df = data_Set[data_Set.Class == 1]  # do the same for frauds

    print("\n\nDescribe normal dataset")
    print(normal_df.Amount.describe())

    print("\n\n Desrcibe fraud dataset")
    print(fraud_df.Amount.describe())

    return normal_df, fraud_df


def calculate_precision(error_df):
    precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    return precision_rt, recall_rt, threshold_rt

