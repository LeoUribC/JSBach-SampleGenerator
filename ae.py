
import keras as ks
from keras import Model, Input



class Autoencoder:
    """
    autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        
        # these are lists
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides

        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        # private attributes
        self._num_conv_layers = len(conv_filters)
        
        self._build()

    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
    

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name='encoder')
    

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')
    

    def _add_conv_layers(self, encoder_input):
        """
        This creates all convolutionals blocks in encoder
        """
        x = encoder_input

        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layers(layer_index, x)
        
        return x


    def _add_conv_layers(self, layer_index, x):
        """
        Adds a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization layer
        """

        layer_number = layer_index + 1

        conv_layer = ks.layers.Conv2D(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"encoder_conv_layer_{layer_number}"
        )

        x = conv_layer(x)
        x = ks.layers.ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = ks.layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)

        return x
    

    

