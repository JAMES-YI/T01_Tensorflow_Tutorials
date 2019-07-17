'''
This file is from
[1] https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder

Modified by JYI, 05/23/2019
(1) load data
(2) tune parameters
(3) maybe train a GAN
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.datasets import mnist
# matplotlib inline

from scipy.stats import norm

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

# load data and preprocessing
K.clear_session()
np.random.seed(237)

(X_train,Y_train), (X_test,Y_test) = mnist.load_data()
X_train = X_train / 255
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test / 255
X_test = X_test.reshape(-1,28,28,1)
X_valid = X_test

y_train = Y_train
y_valid = Y_test

# construct VAE model, before this step, you should have the following
'''
Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
Decoder architecture: 

encoder = Model(input_img, z_mu)
decoder = Model(decoder_input, x)

'''
img_shape = (28, 28, 1)
batch_size = 128
latent_dim = 2

input_img = keras.Input(shape=img_shape) # how to use input_img
print(f'input img {input_img}\n')
x = layers.Conv2D(32, 3,
                  strides=(1,1), padding='same',
                  activation='relu')(input_img) # 28-3+1=26
print(f'1st conv2d x {x}\n')
x = layers.Conv2D(64, 3,
                  padding='same',
                  activation='relu',
                  strides=(2, 2))(x) # 13
print(f'2st conv2d x {x}\n')
x = layers.Conv2D(64, 3,
                  strides=(1,1), padding='same',
                  activation='relu')(x) # 11
print(f'3st conv2d x {x}\n')
x = layers.Conv2D(64, 3,
                  strides=(1,1), padding='same',
                  activation='relu')(x) # 9
print(f'4st conv2d x {x}\n')

shape_before_flattening = K.int_shape(x) # (None,14,14,64)
print(f'shape_before_flattening {shape_before_flattening}')
x = layers.Flatten()(x)
print(f'flattened x {x}\n')
x = layers.Dense(32, activation='relu')(x)
print(f'1st dense x {x}\n')

# Two outputs, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)

# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)
    print(f'K.shape(z_mu)[0] {K.shape(z_mu)[0]}\n')
    return z_mu + K.exp(z_log_sigma) * epsilon #

# sample vector from the latent distribution
'''
dd sampling operation as a new layer
'''
z = layers.Lambda(sampling)([z_mu, z_log_sigma]) # (?,2)
print(f'z after sampling {z}\n')

# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:])
print(f'decoder_input {decoder_input}\n')

# Expand to 784 total pixels
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)
print(f'x 1st dense {x}\n') # (?,12544)
x = layers.Reshape(shape_before_flattening[1:])(x)

# use Conv2DTranspose to reverse the conv layers from the encoder
x = layers.Conv2DTranspose(32, 3,
                           padding='same',
                           activation='relu',
                           strides=(2, 2))(x)
print(f'conv2DT x {x}\n')
x = layers.Conv2D(1, 3,
                  padding='same',
                  activation='sigmoid')(x)

decoder = Model(decoder_input, x) # output shape (None,28,28,1)
print(f'decoder {decoder}\n')

# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)

# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomVariationalLayer()([input_img, z_decoded])

# VAE model statement
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

vae.fit(x=X_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(X_valid, None))

# report results
'''
1 2D visualization of MNIST data set, training data and testing data
2 from latent space to digit space
'''
VisTest2DPath = 'MNIST_2D_VisTest.png'
VisTrain2DPath = 'MNIST_2D_VisTrain.png'

# Isolate original training set records in validation set
X_valid_noTest = X_train
y_valid_noTest = y_train

encoder = Model(input_img, z_mu)
x_valid_noTest_encoded = encoder.predict(X_valid_noTest, batch_size=batch_size)
VisTrain2D = plt.figure(figsize=(10, 10))
plt.scatter(x_valid_noTest_encoded[:, 0], x_valid_noTest_encoded[:, 1], c=y_valid_noTest, cmap='brg')
plt.colorbar()
VisTrain2D.savefig(VisTrain2DPath)

# set colormap so that 11's are gray
custom_cmap = matplotlib.cm.get_cmap('brg')
custom_cmap.set_over('gray')

x_valid_encoded = encoder.predict(X_valid, batch_size=batch_size)
VisTest2D = plt.figure(figsize=(10, 10))
gray_marker = mpatches.Circle(4,radius=0.1,color='gray', label='Test')
plt.legend(handles=[gray_marker], loc = 'best')
plt.scatter(x_valid_encoded[:, 0], x_valid_encoded[:, 1], c=y_valid, cmap=custom_cmap)
plt.clim(0, 9)
plt.colorbar()
VisTest2D.savefig(VisTest2DPath)

# Display a 2D manifold of the digits
n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Construct grid of latent variable values
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# decode for each square in the grid
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')