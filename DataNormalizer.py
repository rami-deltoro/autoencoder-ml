from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


RANDOM_SEED = 314 #used to help randomly select the data points
TEST_PCT = 0.2 # 20% of the data

def normalize_data(data_set):
    df_norm = data_set
    df_norm['Time'] = StandardScaler().fit_transform(df_norm['Time'].values.reshape(-1, 1))
    df_norm['Amount'] = StandardScaler().fit_transform(df_norm['Amount'].values.reshape(-1, 1))
    return df_norm

def divide_set(normalized_data_set):
    train_x, test_x = train_test_split(normalized_data_set, test_size=TEST_PCT, random_state=RANDOM_SEED)
    train_x = train_x[train_x.Class == 0]  # where normal transactions
    train_x = train_x.drop(['Class'], axis=1)  # drop the class column

    test_y = test_x['Class']  # save the class column for the test set
    test_x = test_x.drop(['Class'], axis=1)  # drop the class column

    train_x = train_x.values  # transform to ndarray
    test_x = test_x.values

    print("train_x shape : "+str(train_x.shape))
    print("test_x shape : "+str(test_x.shape))
    print("test_y shape : "+str(test_y.shape))

    return train_x, test_x, test_y

