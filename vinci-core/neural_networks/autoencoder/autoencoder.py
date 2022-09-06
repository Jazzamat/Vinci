# An auto encoder for audio sytheses
# uses Keras, tensor flow


from warnings import filters
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K
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
        

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        #self._build_autoencoder()


#### ENCODER

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(self.input_shape, name="encoder_input")

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


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_space_dim=2
    )

    autoencoder.summary()