
import sys
from autoencoder import Autoencoder
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
import glob




LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 1000

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(f"\n\n THE TYPE OF X_TRAIN WAS {type(x_train)}\n\n")
    x_train = x_train.astype("float32")/255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32")/255
    # x_test = x_test.reshape(x_train.shape + (1,))
    # print(f"\n\n THE TYPE OF X_TRAIN IS {type(x_train)}\n\n")
    return x_train, y_train, x_test, y_test


def load_covers():
    filelist = glob.glob("/home/omer/Documents/Vinci/vinci_core/local_assets/Tracks_and_Covers/*/cover (64, 64).png")
    for i in filelist:
        print(i)
    # print("PRINTING FILE LIST")
    # print(filelist)

    # filelist = "/home/omer/Documents/Vinci/vinci_core/local_assets/Tracks_and_Covers/Zombie/cover (64, 64).png","/home/omer/Documents/Vinci/vinci_core/local_assets/Tracks_and_Covers/Zombie/cover (64, 64).png","/home/omer/Documents/Vinci/vinci_core/local_assets/Tracks_and_Covers/Zombie/cover (64, 64).png"

    # images_array = np.array([np.array(Image.open(fname)) for fname in filelist])
    images_array = np.array([np.array(Image.open(fname)) for fname in filelist[:50]])
    images_array = images_array.astype("float32")/255

    print(f"\n\n THE SHAPE OF IMAGES ARRAY IS {images_array.shape}\n\n")
    return images_array
    


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
         input_shape=(64,64,3),
        conv_filters=(64,128,128,128),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=20
    )

    autoencoder.summary()
    autoencoder.compile(learning_rate=learning_rate)
    autoencoder.train(x_train, batch_size=batch_size, num_epochs=epochs)
    return autoencoder



if __name__ == "__main__":

    x_train, y_train, x_test, ytest = load_mnist()
    
    images_array = load_covers()

    
    autoencoder = train(images_array, LEARNING_RATE, BATCH_SIZE, EPOCHS)

    autoencoder.save("model")
    autoencoder2 = Autoencoder.load("model")
    autoencoder2.summary()