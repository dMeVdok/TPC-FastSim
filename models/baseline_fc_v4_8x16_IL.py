import tensorflow as tf
import math

@tf.function(experimental_relax_shapes=True)
def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    bin_fractions = features[:,-2:] % 1
    features = (
        features[:,:3] - tf.constant([[0., 0., 162.5]])
    ) / tf.constant([[20., 60., 127.5]])
    return tf.concat([features, bin_fractions], axis=-1)

_f = preprocess_features

def get_generator(activation, kernel_init, num_features, latent_dim):
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation=activation, input_shape=(num_features + latent_dim,)),
        tf.keras.layers.Dense(units=64, activation=activation),
        tf.keras.layers.Dense(units=64, activation=activation),
        tf.keras.layers.Dense(units=64, activation=activation),
        tf.keras.layers.Dense(units=6, activation=activation),
        MomentsToImage((8, 16)),
        tf.keras.layers.Activation(tf.keras.activations.relu)
    ], name='generator')
    return generator


class MomentsToImage(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(MomentsToImage, self).__init__()
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'moments_to_image_dim': self.output_dim,
        })
        return config

    def build(self, input_shape):
        self.grid_dim = self.output_dim[0]*self.output_dim[1]
        self.grid = tf.constant([
            [x,y] for x in range(self.output_dim[1]) for y in range(self.output_dim[0])
        ], dtype=tf.float32)

    def call(self, moments):
        mu, ms, it = (
            tf.repeat(
                tf.expand_dims(
                    moments[:,:2],
                    axis=1
                ),
                axis=1,
                repeats=self.grid_dim
            ),
            moments[:,2:-1],
            tf.repeat(
                tf.expand_dims(
                    moments[:,-1],
                    axis=1
                ),
                axis=1,
                repeats=self.grid_dim
            )
        )
        covmat = tf.clip_by_value(
            tf.repeat(
                    tf.expand_dims(
                        tf.gather(ms, [[0,1],[1,2]], axis=1),
                    axis=1), 
                axis=1, 
                repeats=self.grid_dim
            ),
            -1,
            1
        )
        covmat_inv = tf.linalg.inv(covmat + tf.eye(2)*1e-5)
        covmat_det = tf.linalg.det(covmat + tf.eye(2)*1e-5)
        norm = it / tf.math.sqrt(covmat_det*((2*math.pi)**2))
        dens = tf.squeeze(tf.math.exp((-1/2) * (tf.expand_dims((self.grid-mu),-2) @ covmat_inv @ tf.expand_dims((self.grid-mu),-1))))
        return tf.reshape(norm*dens, (-1, self.output_dim[0], self.output_dim[1]))


class ImageToMoments(tf.keras.layers.Layer):
    def __init__(self):
        super(ImageToMoments, self).__init__()

    def build(self, input_shape):
        self.ix = tf.repeat(
            tf.constant(range(input_shape[1]), dtype=tf.float32),
            axis=0,
            repeats=input_shape[2]
        )
        self.iy = tf.repeat(
            tf.constant(range(input_shape[2]), dtype=tf.float32),
            axis=0,
            repeats=input_shape[1]
        )

    def call(self, image):
        image_vec = tf.reshape(image, (-1, image.shape[1]*image.shape[2]))
        it = tf.reduce_sum(image_vec, -1)
        mean_x = tf.reduce_sum(self.ix*image_vec, 1)
        mean_y = tf.reduce_sum(self.iy*image_vec, -1)
        mean_ix = tf.repeat(tf.expand_dims(mean_x, axis=1), axis=1, repeats=self.ix.shape[-1])
        mean_iy = tf.repeat(tf.expand_dims(mean_y, axis=1), axis=1, repeats=self.iy.shape[-1])
        var_x = tf.reduce_sum(image_vec*((self.ix-mean_ix)**2), -1)
        cov_xy = tf.reduce_sum(image_vec*((self.ix-mean_ix)*(self.iy-mean_iy)), -1)
        var_y = tf.reduce_sum(image_vec*((self.iy-mean_iy)**2), -1)
        out = tf.stack([
            mean_x,
            mean_y,
            var_x,
            cov_xy,
            var_y,
            it
        ], axis=-1)
        return out
        


def get_discriminator(activation, kernel_init, dropout_rate, num_features, num_additional_layers, cramer=False,
                      features_to_tail=False):
    input_img = tf.keras.Input(shape=(8, 16))
    moments = ImageToMoments()(input_img)
    features_input = tf.keras.Input(shape=(num_features,))

    if features_to_tail:
        moments_features = tf.concat(
            [moments, features_input],
            axis=-1
        )

    discriminator_tail = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation=activation),
        tf.keras.layers.Dense(units=64, activation=activation),
        tf.keras.layers.Dense(units=64, activation=activation),
        tf.keras.layers.Dense(units=64, activation=activation),
    ], name='discriminator_tail')

    head_input = tf.keras.layers.Concatenate()([features_input, discriminator_tail(moments_features)])

    head_layers = [
        tf.keras.layers.Dense(units=128, activation=activation, input_shape=(num_features + 64,)),
        tf.keras.layers.Dropout(dropout_rate),
    ]
    for _ in range(num_additional_layers):
        head_layers += [
            tf.keras.layers.Dense(units=128, activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
        ]

    discriminator_head = tf.keras.Sequential(
        head_layers + [tf.keras.layers.Dense(units=1 if not cramer else 256,
                                             activation=None)],
        name='discriminator_head'
    )

    inputs = [features_input, input_img]
    outputs = discriminator_head(head_input)

    discriminator = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='discriminator'
    )

    return discriminator


def disc_loss(d_real, d_fake):
    return tf.reduce_mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return tf.reduce_mean(d_real - d_fake)


def disc_loss_cramer(d_real, d_fake, d_fake_2):
    return -tf.reduce_mean(
        tf.norm(d_real - d_fake, axis=-1) +
        tf.norm(d_fake_2, axis=-1) - 
        tf.norm(d_fake - d_fake_2, axis=-1) -
        tf.norm(d_real, axis=-1)
    )

def gen_loss_cramer(d_real, d_fake, d_fake_2):
    return -disc_loss_cramer(d_real, d_fake, d_fake_2)

class BaselineModel_8x16_IL:
    def __init__(self, activation=tf.keras.activations.relu, kernel_init='glorot_uniform',
                 dropout_rate=0.02, lr=1e-4, latent_dim=32, gp_lambda=10., num_disc_updates=8,
                 gpdata_lambda=0., num_additional_layers=0, cramer=False,
                 features_to_tail=True, stochastic_stepping=True):
        self.disc_opt = tf.keras.optimizers.RMSprop(lr)
        self.gen_opt = tf.keras.optimizers.RMSprop(lr)
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.gpdata_lambda = gpdata_lambda
        self.num_disc_updates = num_disc_updates
        self.num_features = 5
        self.cramer = cramer
        self.stochastic_stepping = stochastic_stepping

        self.generator = get_generator(
            activation=activation, kernel_init=kernel_init, latent_dim=latent_dim, num_features=self.num_features
        )
        self.discriminator = get_discriminator(
            activation=activation, kernel_init=kernel_init, dropout_rate=dropout_rate, num_features=self.num_features,
            num_additional_layers=num_additional_layers, cramer=cramer, features_to_tail=features_to_tail
        )

        self.step_counter = tf.Variable(0, dtype='int32', trainable=False)

#        # compile the models with an arbitrary loss func for serializablility
#        self.generator.compile(optimizer=self.gen_opt,
#                               loss='mean_squared_error')
#        self.discriminator.compile(optimizer=self.disc_opt,
#                               loss='mean_squared_error')


    @tf.function
    def make_fake(self, features):
        size = tf.shape(features)[0]
        latent_input = tf.random.normal(shape=(size, self.latent_dim), dtype='float32')
        return self.generator(
            tf.concat([_f(features), latent_input], axis=-1)
        )

    def gradient_penalty(self, features, real, fake):
        alpha = tf.random.uniform(shape=[len(real), 1, 1])
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as t:
            t.watch(interpolates)
            d_int = self.discriminator([_f(features), interpolates])
#            if self.cramer:
#                d_fake = self.discriminator([_f(features), interpolates])
#                d_int = tf.norm(d_int - d_fake, axis=-1)
        grads = tf.reshape(t.gradient(d_int, interpolates), [len(real), -1])
        return tf.reduce_mean(tf.maximum(tf.norm(grads, axis=-1) - 1, 0)**2)

    def gradient_penalty_on_data(self, features, real):
        with tf.GradientTape() as t:
            t.watch(real)
            d_real = self.discriminator([_f(features), real])
#            if self.cramer:
#                d_real = tf.norm(d_real, axis=-1)
        grads = tf.reshape(t.gradient(d_real, real), [len(real), -1])
        return tf.reduce_mean(tf.reduce_sum(grads**2, axis=-1))

    @tf.function
    def calculate_losses(self, feature_batch, target_batch):
        fake = self.make_fake(feature_batch)
        d_real = self.discriminator([_f(feature_batch), target_batch])
        d_fake = self.discriminator([_f(feature_batch), fake])
        if self.cramer:
            fake_2 = self.make_fake(feature_batch)
            d_fake_2 = self.discriminator([_f(feature_batch), fake_2])

        if not self.cramer:
            d_loss = disc_loss(d_real, d_fake)
        else:
            d_loss = disc_loss_cramer(d_real, d_fake, d_fake_2)

        if self.gp_lambda > 0:
            d_loss = (
                d_loss +
                self.gradient_penalty(
                    feature_batch, target_batch, fake
                ) * self.gp_lambda
            )
        if self.gpdata_lambda > 0:
            d_loss = (
                d_loss +
                self.gradient_penalty_on_data(
                    feature_batch, target_batch
                ) * self.gpdata_lambda
            )
        if not self.cramer:
            g_loss = gen_loss(d_real, d_fake)
        else:
            g_loss = gen_loss_cramer(d_real, d_fake, d_fake_2)

        return {'disc_loss': d_loss, 'gen_loss': g_loss}

    def disc_step(self, feature_batch, target_batch):
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        grads = t.gradient(losses['disc_loss'], self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return losses

    def gen_step(self, feature_batch, target_batch):
        feature_batch = tf.convert_to_tensor(feature_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        with tf.GradientTape() as t:
            losses = self.calculate_losses(feature_batch, target_batch)

        grads = t.gradient(losses['gen_loss'], self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return losses

    @tf.function
    def training_step(self, feature_batch, target_batch):
        if self.stochastic_stepping:
            if tf.random.uniform(
                shape=[], dtype='int32',
                maxval=self.num_disc_updates + 1
            ) == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
            else:
                result = self.disc_step(feature_batch, target_batch)
        else:
            if self.step_counter == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
                self.step_counter.assign(0)
            else:
                result = self.disc_step(feature_batch, target_batch)
                self.step_counter.assign_add(1)
        return result
