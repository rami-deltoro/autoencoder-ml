from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd


nb_epoch = 100
batch_size = 128


def create_model(train_x):

    input_dim = train_x.shape[1]  # num of columns, 30
    encoding_dim = 14
    hidden_dim = int(encoding_dim / 2)  # i.e. 7
    learning_rate = 1e-7

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(hidden_dim, activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    return autoencoder


def train_model(autoencoder,train_x,test_x):
    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')

    cp = ModelCheckpoint(filepath="autoencoder_fraud.h5",
                         save_best_only=True,
                         verbose=0)

    tb = TensorBoard(log_dir='./logs',
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)

    history = autoencoder.fit(train_x, train_x,
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(test_x, test_x),
                              verbose=1,
                              callbacks=[cp, tb]).history
    autoencoder = load_model('autoencoder_fraud.h5')
    return history, autoencoder


def reconstruct_error_check(autoencoder,test_x,test_y):
    test_x_predictions = autoencoder.predict(test_x)
    mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': test_y})
    print("\n\nError data set:")
    print(error_df.describe())
    return error_df


