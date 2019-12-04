import tensorflow as tf

def apply_conv(x, n_out, name):
    n_in = x.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.compat.v1.get_variable(scope + "w",
                                 shape=[3, 3, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        bias_init_var = tf.constant(0.0, dtype=tf.float32, shape=[n_out])
        biases = tf.Variable(bias_init_var, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
    return z


def apply_conv_3D(x, n_out, name):
    n_in = x.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[3, 3, 3, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv3d(x, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
        bias_init_var = tf.constant(0.0, dtype=tf.float32, shape=[n_out])
        biases = tf.Variable(bias_init_var, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
    return z

def conv_op(input_op, name, kh, kw, n_out, dh, dw, ifactivate):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())

        conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
        bias_init_var = tf.constant(0.0, dtype=tf.float32, shape=[n_out])
        biases = tf.Variable(bias_init_var, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        if ifactivate is True:
            activation = tf.nn.relu(z, name=scope)
        else:
            activation = z
        return activation

def real2complex(input_op, inv=False):
    if inv == False:
        return tf.complex(input_op[:, :, :, 0], input_op[:, :, :, 1])
    else:
        input_real = tf.cast(tf.real(input_op), dtype=tf.float32)
        input_imag = tf.cast(tf.imag(input_op), dtype=tf.float32)
        return tf.stack([input_real, input_imag], axis=3)


                

def getADMM_2D(y_m, mask, n_iter, n_coil):
    kdata = tf.stack([tf.math.real(y_m), tf.math.imag(y_m)], axis=3)
    beta = tf.zeros_like(kdata)
    z = tf.zeros_like(kdata)
    x = tf.zeros_like(kdata)

    for iter in range(n_iter):
        with tf.compat.v1.variable_scope('recon_layer_{}_{}'.format(n_coil, iter)):
            # y_cplx = tf.complex(y_m[:, :, :, 0], y_m[:, :, :, 1])
            # evalop_cplx = tf.ifft2d(y_cplx)
            # evalop = tf.stack([tf.real(evalop_cplx), tf.imag(evalop_cplx)], axis=3)
            # update = tf.concat([evalop, z-beta], axis=-1)

            x_cplx = tf.complex(x[..., 0], x[..., 1])
            Ax = tf.signal.fft2d(x_cplx) * mask
            evalop_k = tf.stack([tf.math.real(Ax), tf.math.imag(Ax)], axis=3)
            update = tf.concat([evalop_k, kdata], axis=-1)
            update = tf.nn.relu(apply_conv(update, n_out=16, name='update1'), name='relu_1')
            update = apply_conv(update, n_out=2, name='update2')

            update_cplx = tf.complex(update[:, :, :, 0], update[:, :, :, 1])
            input1_cplx = tf.signal.ifft2d(update_cplx * mask)
            input1 = tf.stack([tf.math.real(input1_cplx), tf.math.imag(input1_cplx)], axis=3)

            v = z - beta
            update = tf.concat([v, x, input1], axis=-1)

            update = tf.nn.relu(apply_conv(update, n_out=16, name='update3'), name='relu_1')
            update = tf.nn.relu(apply_conv(update, n_out=16, name='update4'), name='relu_2')
            update = apply_conv(update, n_out=2, name='update5')

            x = x + update

        with tf.compat.v1.variable_scope('denoise_layer_{}'.format(iter)):
            update = tf.nn.relu(apply_conv(x + beta, n_out=8, name='update6'), name='relu_1')
            update = tf.nn.relu(apply_conv(update, n_out=8, name='update7'), name='relu_2')
            update = apply_conv(update, n_out=2, name='update8')
            z = x + beta + update

        with tf.compat.v1.variable_scope('update_layer_{}'.format(iter)):
            eta = tf.Variable(tf.constant(1, dtype=tf.float32), name='eta')
            beta = beta + tf.multiply(eta, x - z)
    output = tf.complex(x[..., 0], x[..., 1])
    return output

def DC_CNN_2D(input_image_Net, mask, kspace):
    # D5C5
    temp = input_image_Net
    for i in range(5):
        conv_1 = conv_op(temp, name='conv'+str(i+1)+'_1', kh=3, kw=3, n_out=16, dh=1, dw=1, ifactivate=True)
        conv_2 = conv_op(conv_1, name='conv'+str(i+1)+'_2', kh=3, kw=3, n_out=16, dh=1, dw=1, ifactivate=True)
        conv_3 = conv_op(conv_2, name='conv'+str(i+1)+'_3', kh=3, kw=3, n_out=16, dh=1, dw=1, ifactivate=True)
        conv_4 = conv_op(conv_3, name='conv'+str(i+1)+'_4', kh=3, kw=3, n_out=16, dh=1, dw=1, ifactivate=True)
        conv_5 = conv_op(conv_4, name='conv'+str(i+1)+'_5', kh=3, kw=3, n_out=2, dh=1, dw=1, ifactivate=False)
        block = temp + conv_5
        block_dc = dc_DCCNN(block, ku_complex=kspace, mask=mask)
        temp = block_dc
    return temp

def getMultiCoilImage(y_m_multicoil, mask, n_iter):
    x = []
    for c in range(20):
        y_m = y_m_multicoil[:, :, :, c]
        output_c = getADMM_2D(y_m, mask, n_iter, c)
        #output_c = dc(output_c, y_m, mask)
        x.append(output_c)
    output = tf.stack([x[i] for i in range(20)], axis=-1)
    return output

def getCoilCombineImage(y_m_multicoil, mask, n_iter):
    x = []
    nSlice, nFE, nPE, nCoil = y_m_multicoil.shape
    for c in range(nCoil):
        y_m = y_m_multicoil[:, :, :, c]
        output_c = getADMM_2D(y_m, mask, n_iter, c)
        output_c = dc(output_c, y_m, mask)
        x.append(output_c)
    output = tf.stack([x[i] for i in range(nCoil)], axis=-1)
    # output: complex tensor: batch, nx, ny, 20

    x = tf.concat([tf.math.real(output), tf.math.imag(output)], axis=-1)
    x = tf.nn.relu(apply_conv(x, n_out=32, name='recon_conv1'))
    x = tf.nn.relu(apply_conv(x, n_out=32, name='recon_conv2'))
    x = apply_conv(x, n_out=2, name='recon_conv3')
    x = tf.abs(tf.complex(x[..., 0], x[..., 1]))
    return x

def getCoilCombineImage_DCCNN(y_m_multicoil, mask, n_iter):
    x = []
    nSlice, nFE, nPE, nCoil = y_m_multicoil.shape
    for c in range(nCoil):
        y_m = y_m_multicoil[:, :, :, c]
        x_m = tf.signal.ifft2d(y_m)
        x_m = real2complex(x_m, inv=True)
        output_c = DC_CNN_2D(x_m, mask, y_m)
        output_c = real2complex(output_c)
        x.append(output_c)
    output = tf.stack([x[i] for i in range(nCoil)], axis=-1)
    # output: complex tensor: batch, nx, ny, 20

    x = tf.concat([tf.math.real(output), tf.math.imag(output)], axis=-1)
    x = tf.nn.relu(apply_conv(x, n_out=32, name='recon_conv1'))
    x = tf.nn.relu(apply_conv(x, n_out=32, name='recon_conv2'))
    x = apply_conv(x, n_out=2, name='recon_conv3')
    x = tf.abs(tf.complex(x[..., 0], x[..., 1]))
    return x

def dc(x0_complex, ku_complex, mask):
    k0_complex = tf.signal.fft2d(x0_complex, 'fft2')
    k0_complex_dc = tf.multiply((1-mask), k0_complex) + ku_complex
    x0_dc = tf.signal.ifft2d(k0_complex_dc)
    return x0_dc

def dc_DCCNN(input_op, ku_complex, mask):
    image = real2complex(input_op)
    k0_complex = tf.signal.fft2d(image, 'fft2')
    k0_complex_dc = tf.multiply((1-mask), k0_complex) + ku_complex
    input_dc = tf.signal.ifft2d(k0_complex_dc)
    input_dc = real2complex(input_dc, inv=True)
    return input_dc







