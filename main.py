import FileReader as fileReader
import Plotter as plot
import DatasetManipulator as datasetManipulator
import DataNormalizer as dataNormalizer
import ModelService as modelService


data_set = fileReader.get_data_set("creditcard.csv")


#Exploratory Analysis
plot.visualize_data_set(data_set)
normal_df, fraud_df = datasetManipulator.split_data_set(data_set)
plot.visualize_transaction_amount_data(normal_df,fraud_df)
plot.visualize_data_by_hour(normal_df,fraud_df)
plot.visualize_transactions_amount_vs_hour(normal_df,fraud_df)

#normalize data
normalized_data = dataNormalizer.normalize_data(data_set)
train_x, test_x, test_y = dataNormalizer.divide_set(normalized_data)
#Create and train model
autoencoder = modelService.create_model(train_x)
history, autoencoder_trained = modelService.train_model(autoencoder,train_x,test_x)


error_df = modelService.reconstruct_error_check(autoencoder_trained,test_x,test_y)
precision_rt, recall_rt, threshold_rt = datasetManipulator.calculate_precision(error_df)

plot.visualize_model_loss(history)
plot.roc_curve_check(error_df)
plot.recall_vs_precision(precision_rt, recall_rt, threshold_rt)
plot.recall_vs_prescision_for_different_threshold(precision_rt, recall_rt, threshold_rt)
plot.reconstruction_error_vs_threshold_check(error_df)
plot.visualize_confusion_matrix(error_df)
