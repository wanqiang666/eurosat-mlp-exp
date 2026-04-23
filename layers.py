import numpy as np


class Linear:
    """
    全连接层:
        out = x @ W + b

    输入:
        x: (batch_size, in_features)
    输出:
        out: (batch_size, out_features)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        # Xavier 初始化，适合 tanh/sigmoid；用于 ReLU 也能先跑通
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: 上游梯度, shape = (batch_size, out_features)

        returns:
            dx: 对输入 x 的梯度, shape = (batch_size, in_features)
        """
        if self.x is None:
            raise ValueError("Must call forward() before backward().")

        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        return dx

    def parameters(self):
        return [self.W, self.b]

    def gradients(self):
        return [self.dW, self.db]


class ReLU:
    """
    ReLU 激活:
        out = max(0, x)
    """

    def __init__(self) -> None:
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise ValueError("Must call forward() before backward().")

        dx = dout * (self.x > 0)
        return dx.astype(np.float32)

    def parameters(self):
        return []

    def gradients(self):
        return []


class Sigmoid:
    """
    Sigmoid 激活:
        out = 1 / (1 + exp(-x))
    """

    def __init__(self) -> None:
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise ValueError("Must call forward() before backward().")

        dx = dout * self.out * (1.0 - self.out)
        return dx.astype(np.float32)

    def parameters(self):
        return []

    def gradients(self):
        return []


class Tanh:
    """
    Tanh 激活:
        out = tanh(x)
    """

    def __init__(self) -> None:
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise ValueError("Must call forward() before backward().")

        dx = dout * (1.0 - self.out ** 2)
        return dx.astype(np.float32)

    def parameters(self):
        return []

    def gradients(self):
        return []


class Sequential:
    """
    简单的顺序容器，方便把多层串起来。
    """

    def __init__(self, *layers) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def gradients(self):
        grads = []
        for layer in self.layers:
            if hasattr(layer, "gradients"):
                grads.extend(layer.gradients())
        return grads

    def get_linear_layers(self):
        """
        返回所有 Linear 层，便于做 L2 正则或参数更新。
        """
        return [layer for layer in self.layers if isinstance(layer, Linear)]