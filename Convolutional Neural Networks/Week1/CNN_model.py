import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)


# GRADED FUNCTION: zero_pad
# pads all the images of a batch of examples X with zeros.
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

    return X_pad


# GRADED FUNCTION: conv_single_step
def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)

    return Z


# GRADED FUNCTION: conv_forward
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above.
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :]  # Select ith training example's padded activation
        for h in range(n_H):  # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):  # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):  # loop over channels (= #filters) of the output volume

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (
                    # ≈3 line)
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


# GRADED FUNCTION: pool_forward
def pool_forward(A_prev, hparameters, mode='max'):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters['f']
    stride = hparameters['stride']
    n_H = int((n_H_prev - f) / stride + 1)
    n_W = int((n_W_prev - f) / stride + 1)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):
                    a_prev_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)

    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


# GRADED FUNCTION:
def conv_backward(dZ, cache):

    (A_prev, W, b, hparameters) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = np.zeros(A_prev_pad.shape)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :]
        da_prev_pad = dA_prev_pad[i, :]

        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):

                    a_slice = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]

                    da_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW [:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db [:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad: -pad, pad: -pad, :]

    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


# GRADED FUNCTION: create_mask_from_window
def create_mask_from_window(x):
    mask = (x == np.max(x))

    return mask


# GRADED FUNCTION: distribute_value(dz, shape)
def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = n_H * n_W
    a = np.ones(shape) * dz / average

    return a


# GRADED FUNCTION: pool_backward
def pool_backward(dA, cache, mode='max'):

    (A_prev, hparameters) = cache

    stride = hparameters['stride']
    f = hparameters['f']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i, :]

        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_satrt = w * stride
                horiz_end = horiz_satrt + f

                for c in range(n_C):

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start: vert_end, horiz_satrt: horiz_end, c]

                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i, vert_start: vert_end, horiz_satrt: horiz_end, c] += np.multiply(mask, dA[i,
                        vert_start: vert_end, horiz_satrt: horiz_end, c])

                    elif mode == 'average':
                        da = np.mean(dA[i, vert_start: vert_end, horiz_satrt: horiz_end, c])
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_satrt: horiz_end, c] += (distribute_value(da, shape) + da)

    assert (dA_prev.shape == A_prev.shape)

    return dA_prev
