import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ============================================================
# WEEK 3 ASSIGNMENT, Q1 - 2-Layer CNN on CIFAR-10
#
# Your job: implement the four functions marked TODO.
#   - conv_forward
#   - conv_backward
#   - maxpool_forward (already provided)
#   - maxpool_backward (already provided)
#
# Everything else (ReLU, FC, loss, training) is provided.
# NumPy only inside the four functions — no PyTorch autograd.
#
# Architecture:
#   Conv(3→8, 3x3, pad=1) → ReLU → MaxPool(2x2)
#   Conv(8→16, 3x3, pad=1) → ReLU → MaxPool(2x2)
#   Flatten → FC(1024→10) → Softmax
#
# Target: ~30% test accuracy after 20 epochs (CPU is fine).
# ============================================================


# ------------------------------------------------------------
# Data loading  (nothing to change here)
# ------------------------------------------------------------

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    # Small subset so it runs in reasonable time on CPU
    train_set = torch.utils.data.Subset(train_set, range(3000))
    test_set  = torch.utils.data.Subset(test_set,  range(500))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ------------------------------------------------------------
# TODO 1 — Conv forward pass
# ------------------------------------------------------------

def conv_forward(X, W, b, stride=1, pad=1):
    """
    Forward pass for a convolutional layer.

    Args:
        X  : numpy array, shape (N, C_in, H, W)
        W  : filters,     shape (C_out, C_in, kH, kW)
        b  : bias,        shape (C_out,)

    Returns:
        out   : numpy array, shape (N, C_out, H_out, W_out)
        cache : tuple of everything needed for the backward pass
    """
    N, C_in, H, W_in = X.shape
    C_out, _, kH, kW = W.shape

    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W_in + 2 * pad - kW) // stride + 1

    # TODO: pad X spatially using np.pad (pad only H and W dimensions)
    X_pad = np.pad(
    X,
    ((0, 0), (0, 0), (pad, pad), (pad, pad)),
    mode='constant'
    )

    out = np.zeros((N, C_out, H_out, W_out))

    # TODO: fill in `out` using a loop over H_out and W_out.
    # For each position (h, w):
    #   1. extract the patch: X_pad[:, :, h*stride : h*stride+kH, w*stride : w*stride+kW]
    #   2. for each filter f: out[:, f, h, w] = sum(patch * W[f]) + b[f]
    # Hint: np.einsum or a reshape + dot product works here.
    for h in range(H_out):
        for w in range(W_out):
            patch = X_pad[
                :,
                :,
                h * stride : h * stride + kH,
                w * stride : w * stride + kW
            ]

            for f in range(C_out):
                out[:, f, h, w] = (
                    np.sum(patch * W[f], axis=(1, 2, 3))
                    + b[f]
                )

    cache = (X_pad, W, b, stride, pad)
    return out, cache


# ------------------------------------------------------------
# TODO 2 — Conv backward pass
# ------------------------------------------------------------

def conv_backward(dout, cache):
    """
    Backward pass for the convolutional layer.

    Args:
        dout  : upstream gradient, shape (N, C_out, H_out, W_out)
        cache : from conv_forward

    Returns:
        dX : shape (N, C_in, H, W)
        dW : shape (C_out, C_in, kH, kW)
        db : shape (C_out,)
    """
    X_pad, W, b, stride, pad = cache
    N, C_in, H_pad, W_pad = X_pad.shape
    C_out, _, kH, kW = W.shape
    _, _, H_out, W_out = dout.shape

    dX_pad = np.zeros_like(X_pad)
    dW     = np.zeros_like(W)

    # TODO: compute db
    # db[f] = sum of dout[:, f, :, :] over all N, H_out, W_out
    db = np.sum(dout, axis=(0, 2, 3))

    # TODO: compute dW and dX_pad by looping over (h, w) positions.
    # For each (h, w):
    #   patch    = X_pad[:, :, h*stride:h*stride+kH, w*stride:w*stride+kW]
    #   dW      += einsum('nf,nchw->fchw', dout[:,:,h,w], patch)
    #   dX_pad[:, :, h*stride:h*stride+kH, w*stride:w*stride+kW]
    #            += einsum('nf,fchw->nchw', dout[:,:,h,w], W)
    for h in range(H_out):
        for w in range(W_out):
            patch = X_pad[
                :,
                :,
                h * stride : h * stride + kH,
                w * stride : w * stride + kW
            ]

            dW += np.einsum(
                'nf,nchw->fchw',
                dout[:, :, h, w],
                patch
            )

            dX_pad[
                :,
                :,
                h * stride : h * stride + kH,
                w * stride : w * stride + kW
            ]+= np.einsum(
                'nf,fchw->nchw',
                dout[:, :, h, w],
                W
            )

    dX = dX_pad[:, :, pad:-pad, pad:-pad] if pad > 0 else dX_pad
    return dX, dW, db


# ------------------------------------------------------------
# maxpool_forward
# ------------------------------------------------------------

def maxpool_forward(X, pool_size=2, stride=2):
    N, C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out))

    for h in range(H_out):
        for w in range(W_out):
            # Extract the pooling window for all N and C at once.
            window = X[:, :,
                       h * stride : h * stride + pool_size,
                       w * stride : w * stride + pool_size]
            # Take max over the spatial extent of the window (axes 2 and 3).
            out[:, :, h, w] = window.max(axis=(2, 3))

    cache = (X, pool_size, stride)
    return out, cache


# ------------------------------------------------------------
# maxpool_backward
# ------------------------------------------------------------

def maxpool_backward(dout, cache):
    X, pool_size, stride = cache
    N, C, H, W = X.shape
    _, _, H_out, W_out = dout.shape
    dX = np.zeros_like(X)

    for h in range(H_out):
        for w in range(W_out):
            window = X[:, :,
                       h * stride : h * stride + pool_size,
                       w * stride : w * stride + pool_size]

            # Find the flat index of the max within each (pool_size x pool_size) window.
            # Reshape to (N, C, pool_size*pool_size) so argmax gives a scalar per (n, c).
            flat      = window.reshape(N, C, -1)
            max_idx   = flat.argmax(axis=2)            # shape: (N, C)

            # Convert flat index back to 2D (row, col) within the window.
            max_row   = max_idx // pool_size
            max_col   = max_idx %  pool_size

            # Build index arrays to scatter dout into dX.
            n_idx = np.arange(N)[:, None]              # (N, 1)
            c_idx = np.arange(C)[None, :]              # (1, C)

            dX[n_idx, c_idx,
               h * stride + max_row,
               w * stride + max_col] += dout[:, :, h, w]

    return dX


# ------------------------------------------------------------
# Provided layers  (do not modify)
# ------------------------------------------------------------

def relu_forward(X):
    return np.maximum(0, X), X

def relu_backward(dout, cache):
    return dout * (cache > 0)

def fc_forward(X, W, b):
    return X @ W + b, (X, W)

def fc_backward(dout, cache):
    X, W = cache
    return dout @ W.T, X.T @ dout, dout.sum(axis=0)

def softmax_loss(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp     = np.exp(shifted)
    probs   = exp / exp.sum(axis=1, keepdims=True)
    N       = logits.shape[0]
    loss    = -np.log(probs[np.arange(N), y] + 1e-8).mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N
    return loss, dlogits


# ------------------------------------------------------------
# Parameter initialisation  (do not modify)
# ------------------------------------------------------------

def init_params():
    p = {}
    p['W1'] = np.random.randn(8,  3,  3, 3).astype(np.float32) * np.sqrt(2 / (3*3*3))
    p['b1'] = np.zeros(8,  dtype=np.float32)
    p['W2'] = np.random.randn(16, 8,  3, 3).astype(np.float32) * np.sqrt(2 / (3*3*8))
    p['b2'] = np.zeros(16, dtype=np.float32)
    # After conv1+pool: 32→16, after conv2+pool: 16→8. Flat size: 16*8*8 = 1024
    p['W3'] = np.random.randn(1024, 10).astype(np.float32) * np.sqrt(2 / 1024)
    p['b3'] = np.zeros(10, dtype=np.float32)
    return p


# ------------------------------------------------------------
# Forward + backward  (do not modify)
# ------------------------------------------------------------

def forward_backward(X, y, params):
    # Block 1
    c1, cc1 = conv_forward(X, params['W1'], params['b1'])
    r1, cr1 = relu_forward(c1)
    p1, cp1 = maxpool_forward(r1)
    # Block 2
    c2, cc2 = conv_forward(p1, params['W2'], params['b2'])
    r2, cr2 = relu_forward(c2)
    p2, cp2 = maxpool_forward(r2)
    # FC
    flat         = p2.reshape(X.shape[0], -1)
    logits, cfc  = fc_forward(flat, params['W3'], params['b3'])
    loss, dlogits = softmax_loss(logits, y)

    # Backward
    dflat, dW3, db3 = fc_backward(dlogits, cfc)
    dp2  = dflat.reshape(p2.shape)
    dr2  = maxpool_backward(dp2, cp2)
    dc2  = relu_backward(dr2, cr2)
    dp1, dW2, db2 = conv_backward(dc2, cc2)
    dr1  = maxpool_backward(dp1, cp1)
    dc1  = relu_backward(dr1, cr1)
    _,   dW1, db1 = conv_backward(dc1, cc1)

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    return loss, grads


# ------------------------------------------------------------
# Training loop  (do not modify)
# ------------------------------------------------------------

def train(lr=1e-3, batch_size=64, epochs=20):
    train_loader, test_loader = get_dataloaders(batch_size)
    params = init_params()

    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for Xb, yb in train_loader:
            Xb = Xb.numpy()
            yb = yb.numpy()
            loss, grads = forward_backward(Xb, yb, params)
            for k in params:
                params[k] -= lr * grads[k]
            total_loss += loss
            n_batches  += 1

        acc = evaluate(test_loader, params)
        print(f"Epoch {epoch+1}/{epochs}  "
              f"loss: {total_loss/n_batches:.4f}  "
              f"test acc: {acc:.3f}")

    return params


def evaluate(loader, params):
    correct, total = 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.numpy(), yb.numpy()
        c1, _ = conv_forward(Xb, params['W1'], params['b1'])
        r1, _ = relu_forward(c1)
        p1, _ = maxpool_forward(r1)
        c2, _ = conv_forward(p1, params['W2'], params['b2'])
        r2, _ = relu_forward(c2)
        p2, _ = maxpool_forward(r2)
        flat      = p2.reshape(Xb.shape[0], -1)
        logits, _ = fc_forward(flat, params['W3'], params['b3'])
        correct  += (logits.argmax(axis=1) == yb).sum()
        total    += len(yb)
    return correct / total


if __name__ == '__main__':
    train()
