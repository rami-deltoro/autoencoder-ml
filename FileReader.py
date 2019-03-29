import pandas as pd


number_of_lines_to_print=5


def get_data_set(filename):
    print("Filename : "+filename)
    data_set = pd.read_csv(filename)  # unzip and read in data downloaded to the local directory
    print("\nHead data set:")
    print(data_set.head(n=number_of_lines_to_print))
    print("Data_set shape = "+str(data_set.shape))
    print("Data set has nulls? = "+str(data_set.isnull().values.any()))
    return data_set


