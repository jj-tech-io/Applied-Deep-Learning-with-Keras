


import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Flatten, Input
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
#import mnist
import keras.datasets.mnist as mnist
from sklearn.model_selection import train_test_split




class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.opt = keras.optimizers.Adam(learning_rate=1e-4)
    
    self.enc_in = Input(shape=(3), name="img")

    self.enc_d1 = layers.Dense(70, activation="relu")(self.enc_in)
    self.enc_d2 = layers.Dense(70, activation="relu")(self.enc_d1)
    self.enc_out = layers.Dense(5, activation="relu")(self.enc_d2)
    self.encoder = keras.Model(self.enc_in, self.enc_out, name="encoder")
    #call parent opt

    self.encoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
    print(self.encoder.summary())
    # self.dec_in =  layers.InputLayer(input_shape=(5,))(self.enc_out)
    # self.dec_in = InputLayer(shape=(5), name="dec_in")(self.enc_out)
    self.dec_in = layers.Dense(5, activation="relu")(self.enc_out)
    self.dec_d1 = layers.Dense(70, activation="relu")(self.dec_in)
    self.dec_d2 = layers.Dense(70, activation="relu")(self.dec_d1)
    self.dec_out = layers.Dense(3, activation="relu")(self.dec_d2)

    self.decoder = keras.Model(self.dec_in, self.dec_out, name="decoder")
    # self.decoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
    print(self.decoder.summary())


    self.autoencoder = keras.Model(self.enc_in, self.dec_out, name="autoencoder")
    self.autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse")

  def call(self, inputs):
    #   #wrap your loss computation in a zero argument `lambda`
    self.enc_in = inputs
    outputs = self.dec_out
    pixels = self.enc_in
    enc = self.enc_out
    dec = self.dec_out
    # #convert to tensor
    # pixels = tf.convert_to_tensor(pixels)
    # enc = tf.convert_to_tensor(enc)
    # dec = tf.convert_to_tensor(dec)
    self.autoencoder.add_loss(lambda: keras.losses.mean_squared_error(pixels, dec))
    return inputs, outputs

if __name__ == "__main__":
    # Load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    np.random.seed(42)
    #load csv into 
    # data_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\EncoderDecoder\JJ_LUT.csv"
    headers = "Cm,Ch,Bm,Bh,T,sR,sG,sB"
    # df = pd.read_csv(config.LUTv1_PATH, sep=",", header=None, names=headers.split(","))
    # df = pd.read_csv(config.LUTv2_PATH, sep=",", header=None, names=headers.split(","))
    # df = pd.read_csv(config.LUTv3_PATH, sep=",", header=None, names=headers.split(","))

    df = pd.read_csv(r"C:\Users\joeli\OneDrive\Documents\GitHub\Applied-Deep-Learning-with-Keras\data\JJ_LUTv1.csv", sep=",", header=None, names=headers.split(","))

    df.head()
    #remove header
    df = df.iloc[1:]
    #inputs = Cm,Ch,Bm,epi_thick
    y = df[['Cm','Ch','Bm','Bh','T']]
    print(y.head())

    #outputs = sR,sG,sB
    x = df[['sR','sG','sB']]
    print(x.head())

    df.head()
    #remove headers and convert to numpy array
    x = df[['sR','sG','sB']].iloc[1:].to_numpy()
    y = df[['Cm','Ch','Bm','Bh','T']].iloc[1:].to_numpy()
    #train nn on x,y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)

    #numpy arrays
    x_train = np.asarray(x_train).reshape(-1,3).astype('float32')
    x_test = np.asarray(x_test).reshape(-1,3).astype('float32')
    print(f"bef norm x_train[0] {x_train[0]}")
    #normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(f"aft norm x_train[0] {x_train[0]}")   
    auto_encoder = MyModel()
    #build model
    auto_encoder.build(input_shape=(None, 3))
    auto_encoder.compile(loss="mse")
    auto_encoder.summary()
    print(x_train.shape)
    print(y_train.shape)
    auto_encoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_split=0.1)
#%%
