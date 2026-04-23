import numpy as np
from layers import Linear


class SoftmaxCrossEntropyLoss:
    """
    Softmax + CrossEntropy 合并实现
    输入:
        logits: (batch_size, num_classes)
        y_true: (batch_size,)
    """

    def __init__(self) -> None:
        self.probs = None
        self.y_true = None
        self.batch_size = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        数值稳定版 softmax + cross entropy
        """
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.probs = probs
        self.y_true = y_true
        self.batch_size = logits.shape[0]

        correct_class_probs = probs[np.arange(self.batch_size), y_true]
        loss = -np.log(correct_class_probs + 1e-12)
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        """
        返回对 logits 的梯度
        shape = (batch_size, num_classes)
        """
        if self.probs is None or self.y_true is None:
            raise ValueError("Must call forward() before backward().")

        dlogits = self.probs.copy()
        dlogits[np.arange(self.batch_size), self.y_true] -= 1.0
        dlogits /= self.batch_size
        return dlogits.astype(np.float32)
    

def compute_l2_loss(model, weight_decay: float) -> float:
    """
    计算模型中所有 Linear 层权重的 L2 正则项
    注意：通常只对 W 做正则，不对 b 做正则
    """
    l2_loss = 0.0
    for layer in model.get_linear_layers():
        l2_loss += np.sum(layer.W ** 2)
    return 0.5 * weight_decay * float(l2_loss)


def add_l2_gradients(model, weight_decay: float) -> None:
    """
    把 L2 正则项对应的梯度加到各层 dW 上
    d/dW [0.5 * lambda * ||W||^2] = lambda * W
    """
    for layer in model.get_linear_layers():
        layer.dW += weight_decay * layer.W


def compute_accuracy(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y_true)
    return float(acc)