import numpy as np
from layers import Linear, ReLU, Sigmoid, Tanh, Sequential


class MLPClassifier:
    """
    三层神经网络分类器（按层数看：输入层 -> 隐藏层 -> 输出层）
    
    结构:
        Linear(input_dim, hidden_dim)
        Activation
        Linear(hidden_dim, num_classes)

    支持:
        - hidden_dim 自定义
        - activation 在 relu / sigmoid / tanh 之间切换
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str = "relu",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.activation_name = activation.lower()

        activation_layer = self._build_activation(self.activation_name)

        self.net = Sequential(
            Linear(input_dim, hidden_dim),
            activation_layer,
            Linear(hidden_dim, num_classes),
        )

    def _build_activation(self, activation: str):
        if activation == "relu":
            return ReLU()
        elif activation == "sigmoid":
            return Sigmoid()
        elif activation == "tanh":
            return Tanh()
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose from ['relu', 'sigmoid', 'tanh']."
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.net.forward(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.net.backward(dout)

    def parameters(self):
        return self.net.parameters()

    def gradients(self):
        return self.net.gradients()

    def get_linear_layers(self):
        return self.net.get_linear_layers()

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        preds = np.argmax(logits, axis=1)
        return preds

    def save_weights(self, path: str) -> None:
        """
        保存模型参数到 .npz 文件
        """
        linear_layers = self.get_linear_layers()
        save_dict = {}

        for idx, layer in enumerate(linear_layers):
            save_dict[f"W{idx+1}"] = layer.W
            save_dict[f"b{idx+1}"] = layer.b

        save_dict["input_dim"] = np.array([self.input_dim], dtype=np.int64)
        save_dict["hidden_dim"] = np.array([self.hidden_dim], dtype=np.int64)
        save_dict["num_classes"] = np.array([self.num_classes], dtype=np.int64)

        np.savez(path, **save_dict)

    def load_weights(self, path: str) -> None:
        """
        从 .npz 文件加载模型参数
        """
        checkpoint = np.load(path)
        linear_layers = self.get_linear_layers()

        for idx, layer in enumerate(linear_layers):
            layer.W = checkpoint[f"W{idx+1}"].astype(np.float32)
            layer.b = checkpoint[f"b{idx+1}"].astype(np.float32)

    def summary(self) -> None:
        print("MLPClassifier Summary")
        print(f"  Input dim   : {self.input_dim}")
        print(f"  Hidden dim  : {self.hidden_dim}")
        print(f"  Num classes : {self.num_classes}")
        print(f"  Activation  : {self.activation_name}")