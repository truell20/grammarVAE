import copy
from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy, mse
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Concatenate, Reshape
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
import tensorflow as tf
import pdb

MAX_LEN_FUNCTIONAL = 200

class MoleculeVAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = 120,
               max_length_functional=MAX_LEN_FUNCTIONAL,
               latent_rep_size = 292,
               weights_file = None):
        charset_length = len(charset)
        
        x = Input(shape=(max_length, charset_length))
        f = Input(shape=(max_length_functional, 1))

        _, z = self._buildEncoder(x, f, latent_rep_size, max_length, max_length_functional)
        self.encoder = Model([x, f], z)

        encoded_input = Input(shape=(latent_rep_size,))
        o, fo = self._buildDecoder(
            encoded_input,
            latent_rep_size,
            max_length,
            max_length_functional,
            charset_length
        )
        self.decoder = Model(
            encoded_input,
            [o, fo]
        )

        x1 = Input(shape=(max_length, charset_length))
        f1 = Input(shape=(max_length_functional, 1))
        vae_loss, z1 = self._buildEncoder(x1, f1, latent_rep_size, max_length, max_length_functional)
        o1, fo1 = self._buildDecoder(
            z1,
            latent_rep_size,
            max_length,
            max_length_functional,
            charset_length
        )
        self.autoencoder = Model(
            [x1, f1],
            [o1, fo1]
        )


        x2 = Input(shape=(max_length, charset_length))
        f2 = Input(shape=(max_length_functional, 1))
        (z_m, z_l_v) = self._encoderMeanVar(x2, f2, latent_rep_size, max_length, max_length_functional)
        self.encoderMV = Model(inputs=[x2, f2], outputs=[z_m, z_l_v])
        
        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)
            self.encoderMV.load_weights(weights_file, by_name = True)

        self.autoencoder.compile(optimizer = 'Adam',
                                 loss = {'decoded_mean': vae_loss, 'decoded_mean_2': vae_loss},
                                 metrics = ['accuracy'])

        
        
    def _towers(self, x, f, max_length, max_length_func):
         # Tower 1
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)

        # Tower 2
        hf = Flatten( name='tower_2_flatten_enc')(f)
        hf = Dense(256, activation = 'relu', name='tower_2_dense_1')(hf)

        # Merge
        h = Concatenate()([h, hf])
        return Dense(435, activation = 'relu', name='dense_1')(h)
        
     
    def _encoderMeanVar(self, x, f, latent_rep_size, max_length, max_length_func, epsilon_std = 0.01):
         h = self._towers(x, f, max_length, max_length_func)

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        return (z_mean, z_log_var) 


    def _buildEncoder(self, x, f, latent_rep_size, max_length, max_length_func, epsilon_std = 0.01):
        h = self._towers(x, f, max_length, max_length_func)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            
            if K.int_shape(x_decoded_mean)[1] == max_length:
                x = K.flatten(x)
                x_decoded_mean = K.flatten(x_decoded_mean)
                xent_loss = max_length * binary_crossentropy(x, x_decoded_mean)
            elif K.int_shape(x_decoded_mean)[1] == max_length_func:
                t = tf.reshape(x, (-1, max_length_func))
                p = tf.reshape(x_decoded_mean, (-1, max_length_func))
                xent_loss = mse(t, p)
            else:
                raise ValueError('UNRECOGNIZED SHAPE')
                
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, max_length_functional, charset_length):
        l = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        
        # Tower 1
        h = RepeatVector(max_length, name='repeat_vector')(l)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        h = TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)
        
        # Tower 2
        # hf = Dense(128, name='dense_tower_1', activation = 'relu')(l)	
        hf = Dense(200, name='dense_tower_2', activation = 'sigmoid')(l)
        hf = Reshape((200, 1), name='decoded_mean_2')(hf)
        
        return h, hf

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 292, max_length = 120):
        self.create(charset, weights_file = weights_file, latent_rep_size = latent_rep_size)
