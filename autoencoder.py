
import keras as ks
from keras import Model, Input
#from keras import backend as K
import numpy as np



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
        self._shape_before_bottleneck = None
        self._model_input = None
        
        self._build()
    

    def summary(self):
        """
        to print on console information about the architecture
        """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    

    def compile(self, learning_rate=0.0001):
        optimizer = ks.optimizers.Adam(learning_rate=learning_rate)
        mse_loss = ks.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)
    

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit( x_train,
                        x_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        shuffle=True )

    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
    

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder( self.encoder(model_input) )
        self.model = Model(model_input, model_output, name="autoencoder")
    

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
    

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input")
    

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = ks.layers.Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer
    

    def _add_reshape_layer(self, dense_layer):
        return ks.layers.Reshape(self._shape_before_bottleneck)(dense_layer)
    

    def _add_conv_transpose_layers(self, x):
        """
        Add conv transpose blocks
        """
        # loop through all the conv layers in reverse order and stop at the 1st layer
        for layer_index in reversed( range(1, self._num_conv_layers) ):
            x = self._add_conv_transpose_layer(layer_index, x)

        return x
    

    def _add_conv_transpose_layer(self, layer_index, x):

        layer_num = self._num_conv_layers - layer_index

        conv_transpose_layer = ks.layers.Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )

        x = conv_transpose_layer(x)
        x = ks.layers.ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = ks.layers.BatchNormalization(name=f"decoder_bn_{layer_num}")(x)

        return x
    

    def _add_decoder_output(self, x):

        conv_transpose_layer = ks.layers.Conv2DTranspose(
            filters = 1,
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )

        x = conv_transpose_layer(x)
        output_layer = ks.layers.Activation("sigmoid", name="sigmoid_layer")(x)

        return output_layer


    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')
    

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')
    

    def _add_conv_layers(self, encoder_input):
        """
        This creates all convolutionals blocks in encoder
        """
        x = encoder_input

        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        
        return x


    def _add_conv_layer(self, layer_index, x):
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
    

    def _add_bottleneck(self, x):
        """
        Flatten data and add bottleneck (Dense layer)
        """

        self._shape_before_bottleneck = x.shape[1:]
        x = ks.layers.Flatten()(x)
        x = ks.layers.Dense(self.latent_space_dim, name="encoder_output")(x)

        return x
    


if __name__ == "__main__":

    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3 ,3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )

    autoencoder.summary()

