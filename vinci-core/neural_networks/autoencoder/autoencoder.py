# An auto encoder for audio sytheses
# uses Keras, tensor flow


from warnings import filters
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, \
BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
import numpy as np
# deep convolutoinal autoencoder architecture with mirrored encoded and decoder components
class Autoencoder():

    def __init__(self,
                input_shape,
                conv_filters,
                conv_kernels,
                conv_strides,
                latent_space_dim):

        self.input_shape = input_shape # [28, 28, 1] 28 pixels by 28 pixels, by 1 channel
        self.conv_filters = conv_filters # [2,4,8] 2 layers, 4 layers 8 layers
        self.conv_kernels = conv_kernels # [3, 5, 5]
        self.conv_strides = conv_strides # [1,2,2]
        self.latent_space_dim = latent_space_dim # an int ex: 2

        self.encoder = None
        self.decoder = None
        self.Model = None
        self._shape_before_bottleneck = None
        
        self._num_conv_layers = len(conv_filters)
        
        self._build()
        

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        

    # HIGH LEVEL FUNC TO BUIILD AUTOENCODER
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        #self._build_autoencoder()


#### ENCODER

    # HIGH LEVEL FUNC TO BUIILD ENCODER
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    # creates all convolutionals blocks in encoder
    def _add_conv_layers(self, encoder_input):
        x = encoder_input # current conv layer
        
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)

        return x
        
    def _add_conv_layer(self,layer_index, x):
        # adds convolutional block of layers consisting of 
        # conv2d + ReLU + batch normalizatoin

        layer_number = layer_index + 1

        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )

        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)

        return x

    # flatten data and add bottleneck (dense layer)
    def _add_bottleneck(self,x):

        self._shape_before_bottleneck = K.int_shape(x)[1:] # [2,7,7,32] ignoring 2

        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x


#### DECODER 

    ### HIGH LEVEL FUNC TO BUIILD DECODER
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    
    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self,decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 1*2*4 = 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer

    def _add_conv_transpose_layers(self,x):
        # add convolutional transpose blocks
        # loop through all the conv layers in reverse order and stop at the first layer
        for layer_index in reversed(range(1,self._num_conv_layers)):
            # [1 , 2] -> [2, 1]
            x = self._add_conv_transpose_layer(layer_index, x)
        return x 

    def _add_conv_transpose_layer(self,layer_index, x):
        
        layer_num = self._num_conv_layers - layer_index

        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )


        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self,x):

        conv_transpose_layer = Conv2DTranspose(
            filters=1, # [24, 24, 1]
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )

        x = conv_transpose_layer(x)
        output_layer =  Activation("sigmoid", name = "sigmoid_layer")(x)
        return output_layer

if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2
    )

    autoencoder.summary()