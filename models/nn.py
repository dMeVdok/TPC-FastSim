import six
import tensorflow as tf
import numpy as np
import math


custom_objects = {}


def get_activation(activation):
    try:
        activation = tf.keras.activations.get(activation)
    except ValueError:
        activation = eval(activation)
    return activation


class MomentsToImage(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(MomentsToImage, self).__init__()
        self.output_dim = output_dim
        self.max_dim = max(output_dim)

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
        super(MomentsToImage, self).build(input_shape)

    def call(self, moments):
        sx, sy = self.output_dim[1]-1, self.output_dim[0]-1
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
        ms = tf.stack([tf.tanh(ms[:,0]), tf.tanh(ms[:,1]), tf.tanh(ms[:,2])], axis=1)
        ms = tf.clip_by_value(ms, [[0.03, -0.999, 0.03]], [[1, 0.999, 1]])
        ms = tf.stack([ms[:,0]*3*sx, ms[:,1]*tf.sqrt(ms[:,0]*ms[:,2]*9*sx*sy), ms[:,2]*3*sy], axis=1)
        mu = tf.tanh(mu) * tf.constant([[sx/2, sy/2]], dtype=tf.float32) + tf.constant([[sx/2, sy/2]], dtype=tf.float32)
        it = 10*tf.sigmoid(it)
        covmat = tf.repeat(
            tf.expand_dims(
                tf.gather(ms, [[0,1],[1,2]], axis=1),
            axis=1),
            axis=1, 
            repeats=self.grid_dim
        )
        covmat_inv = tf.linalg.inv(covmat)
        covmat_det = tf.linalg.det(covmat)
        norm = it / tf.math.sqrt(covmat_det*((2*math.pi)**2))
        dens = tf.squeeze(tf.math.exp((-1/2) * (tf.expand_dims((self.grid-mu),-2) @ covmat_inv @ tf.expand_dims((self.grid-mu),-1))))
        return tf.transpose(tf.reshape(norm*dens, (-1, self.output_dim[1], self.output_dim[0])), (0,2,1))


class ImageToMoments(tf.keras.layers.Layer):
    def __init__(self):
        super(ImageToMoments, self).__init__()

    def build(self, input_shape):
        if len(input_shape) < 3:
            self.input_dim = input_shape
        else:
            self.input_dim = input_shape[1:]
        self.grid = [(i,j) for i in range(input_shape[1]) for j in range(input_shape[2])]
        super(ImageToMoments, self).build(input_shape)

    def call(self, image):
        sy, sx = self.input_dim[0]-1, self.input_dim[1]-1
        iy = tf.constant([[g[0] for g in self.grid]], dtype=tf.float32)
        ix = tf.constant([[g[1] for g in self.grid]], dtype=tf.float32)
        image_vec = tf.reshape(image, (-1, image.shape[1]*image.shape[2]))
        it = tf.reduce_sum(image_vec, -1)
        it_t = tf.repeat(tf.expand_dims(it, axis=1), axis=1, repeats=image_vec.shape[-1])
        image_vec = image_vec / it_t
        mean_x = tf.reduce_sum(ix*image_vec, 1)
        mean_y = tf.reduce_sum(iy*image_vec, 1)
        mean_ix = tf.repeat(tf.expand_dims(mean_x, axis=1), axis=1, repeats=ix.shape[-1])
        mean_iy = tf.repeat(tf.expand_dims(mean_y, axis=1), axis=1, repeats=iy.shape[-1])
        var_x = tf.reduce_sum(image_vec*((ix-mean_ix)**2), -1)
        cov_xy = tf.reduce_sum(image_vec*((ix-mean_ix)*(iy-mean_iy)), -1)
        var_y = tf.reduce_sum(image_vec*((iy-mean_iy)**2), -1)

        """
        # reverse transform
        out = tf.stack([
            tf.math.atanh((mean_x-sx/2)/(sx/2)),
            tf.math.atanh((mean_y-sy/2)/(sy/2)),
            tf.math.atanh(var_x/(3*sx)),
            tf.math.atanh(cov_xy/tf.sqrt(var_x*var_y)),
            tf.math.atanh(var_y/(3*sy)),
            tf.math.log((it/10)/(1-it/10))
        ], axis=-1)
        """

        out = tf.stack([
            mean_x,
            mean_y,
            var_x,
            cov_xy,
            var_y,
            it
        ], axis=-1)
        
        return out 


def moments_to_image_block(output_shape, name=None):
    return tf.keras.Sequential([MomentsToImage(output_shape)])

def image_to_moments_block(features_shape, image_shape, axis, name=None):
    in1 = tf.keras.Input(shape=features_shape)
    in2 = tf.keras.Input(shape=image_shape)
    concat1, concat2 = in1, ImageToMoments()(in2)
    out = tf.keras.layers.Concatenate(axis=axis)([concat1, concat2])
    args = dict(inputs=[in1, in2], outputs=out)
    if name:
        args['name'] = name
    return tf.keras.Model(**args)

def fully_connected_block(units, activations,
                          kernel_init='glorot_uniform', input_shape=None,
                          output_shape=None, dropouts=None, name=None):
    assert len(units) == len(activations)
    if dropouts:
        assert len(dropouts) == len(units)

    activations = [get_activation(a) for a in activations]

    layers = []
    for i, (size, act) in enumerate(zip(units, activations)):
        args = dict(units=size, activation=act, kernel_initializer=kernel_init)
        if i == 0 and input_shape:
            args['input_shape'] = input_shape

        layers.append(tf.keras.layers.Dense(**args))

        if dropouts and dropouts[i]:
            layers.append(tf.keras.layers.Dropout(dropouts[i]))

    if output_shape:
        layers.append(tf.keras.layers.Reshape(output_shape))

    args = {}
    if name:
        args['name'] = name

    return tf.keras.Sequential(layers, **args)


def fully_connected_residual_block(units, activations, input_shape,
                                   kernel_init='glorot_uniform', batchnorm=True,
                                   output_shape=None, dropouts=None, name=None):
    assert isinstance(units, int)
    if dropouts:
        assert len(dropouts) == len(activations)
    else:
        dropouts = [None] * len(activations)

    activations = [get_activation(a) for a in activations]

    def single_block(xx, units, activation, kernel_init, batchnorm, dropout):
        xx = tf.keras.layers.Dense(units=units, kernel_initializer=kernel_init)(xx)
        if batchnorm:
            xx = tf.keras.layers.BatchNormalization()(xx)
        xx = activation(xx)
        if dropout:
            xx = tf.keras.layers.Dropout(dropout)(xx)
        return xx

    input_tensor = tf.keras.Input(shape=input_shape)
    xx = input_tensor
    for i, (act, dropout) in enumerate(zip(activations, dropouts)):
        args = dict(units=units, activation=act, kernel_init=kernel_init,
                    batchnorm=batchnorm, dropout=dropout)
        if len(xx.shape) == 2 and xx.shape[1] == units:
            xx = xx + single_block(xx, **args)
        else:
            assert i == 0
            xx = single_block(xx, **args)

    if output_shape:
        xx = tf.keras.layers.Reshape(output_shape)(xx)

    args = dict(inputs=input_tensor, outputs=xx)
    if name:
        args['name'] = name
    return tf.keras.Model(**args)


def concat_block(input1_shape, input2_shape, reshape_input1=None,
                 reshape_input2=None, axis=-1, name=None):
    in1 = tf.keras.Input(shape=input1_shape)
    in2 = tf.keras.Input(shape=input2_shape)
    concat1, concat2 = in1, in2
    if reshape_input1:
        concat1 = tf.keras.layers.Reshape(reshape_input1)(concat1)
    if reshape_input2:
        concat2 = tf.keras.layers.Reshape(reshape_input2)(concat2)
    out = tf.keras.layers.Concatenate(axis=axis)([concat1, concat2])
    args = dict(inputs=[in1, in2], outputs=out)
    if name:
        args['name'] = name
    return tf.keras.Model(**args)


def conv_block(filters, kernel_sizes, paddings, activations, poolings,
               kernel_init='glorot_uniform', input_shape=None, output_shape=None,
               dropouts=None, name=None):
    assert len(filters) == len(kernel_sizes) == len(paddings) == len(activations) == len(poolings)
    if dropouts:
        assert len(dropouts) == len(filters)

    activations = [get_activation(a) for a in activations]

    layers = []
    for i, (nfilt, ksize, padding, act, pool) in enumerate(zip(filters, kernel_sizes, paddings,
                                                               activations, poolings)):
        args = dict(filters=nfilt, kernel_size=ksize,
                    padding=padding, activation=act, kernel_initializer=kernel_init)
        if i == 0 and input_shape:
            args['input_shape'] = input_shape

        layers.append(tf.keras.layers.Conv2D(**args))

        if dropouts and dropouts[i]:
            layers.append(tf.keras.layers.Dropout(dropouts[i]))

        if pool:
            layers.append(tf.keras.layers.MaxPool2D(pool))

    if output_shape:
        layers.append(tf.keras.layers.Reshape(output_shape))

    args = {}
    if name:
        args['name'] = name

    return tf.keras.Sequential(layers, **args)


def vector_img_connect_block(vector_shape, img_shape, block,
                             vector_bypass=False, concat_outputs=True, name=None):
    vector_shape = tuple(vector_shape)
    img_shape = tuple(img_shape)

    assert len(vector_shape) == 1
    assert 2 <= len(img_shape) <= 3

    input_vec = tf.keras.Input(shape=vector_shape)
    input_img = tf.keras.Input(shape=img_shape)

    block_input = input_img
    if len(img_shape) == 2:
        block_input = tf.keras.layers.Reshape(img_shape + (1,))(block_input)
    if not vector_bypass:        
        reshaped_vec = tf.tile(
            tf.keras.layers.Reshape((1, 1) + vector_shape)(input_vec),
            (1, *img_shape[:2], 1)
        )
        block_input = tf.keras.layers.Concatenate(axis=-1)([block_input, reshaped_vec])

    block_output = block(block_input)

    outputs = [input_vec, block_output]
    if concat_outputs:
        outputs = tf.keras.layers.Concatenate(axis=-1)(outputs)

    args = dict(
        inputs=[input_vec, input_img],
        outputs=outputs,
    )

    if name:
        args['name'] = name

    return tf.keras.Model(**args)


def build_block(block_type, arguments):
    if block_type == 'fully_connected':
        block = fully_connected_block(**arguments)
    elif block_type == 'conv':
        block = conv_block(**arguments)
    elif block_type == 'connect':
        inner_block = build_block(**arguments['block'])
        arguments['block'] = inner_block
        block = vector_img_connect_block(**arguments)
    elif block_type == 'concat':
        block = concat_block(**arguments)
    elif block_type == 'fully_connected_residual':
        block = fully_connected_residual_block(**arguments)
    elif block_type == 'moments_to_image':
        block = moments_to_image_block(**arguments)
    elif block_type == 'image_to_moments':
        block = image_to_moments_block(**arguments)
    else:
        raise(NotImplementedError(block_type))

    return block


def build_architecture(block_descriptions, name=None, custom_objects_code=None):
    if custom_objects_code:
        print("build_architecture(): got custom objects code, executing:")
        print(custom_objects_code)
        exec(custom_objects_code, globals(), custom_objects)

    blocks = [build_block(**descr)
              for descr in block_descriptions]

    inputs = [
        tf.keras.Input(shape=i.shape[1:])
        for i in blocks[0].inputs
    ]
    outputs = inputs
    for block in blocks:
        outputs = block(outputs)

    args = dict(
        inputs=inputs,
        outputs=outputs
    )
    if name:
        args['name'] = name
    return tf.keras.Model(**args)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    inp = tf.constant([
        [-0.286156327, 0.209562385, 0.05146236, 0, 0.05727632, 2.93366218],
        [0, 0, 0.09, 1, 0.08, 100]
    ], dtype=tf.float32)
    mti = MomentsToImage((8, 16))
    itm = ImageToMoments()
    print(itm(mti(inp)))
    plt.imshow(mti(inp)[1].numpy())
    plt.show()
