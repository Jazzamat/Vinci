# An auto encoder for audio sytheses
# uses Keras, tensor flow


from tensorflow.keras import Model
from tensorflow.keras.layers import Input

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
        
        self._num_conv_layers = len(conv_filters)
        
        self._build()
        

    def build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()



    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(self.input_shape)


    