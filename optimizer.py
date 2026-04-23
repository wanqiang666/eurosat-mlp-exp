class SGD:
    """
    手写 SGD 优化器，支持：
    - learning rate
    - weight decay（更推荐在 loss.py 里显式加到 dW，这里保留接口）
    - lr decay
    """

    def __init__(
        self,
        model,
        lr: float = 1e-2,
        lr_decay: float = 1.0,
        min_lr: float = 1e-6,
    ) -> None:
        self.model = model
        self.lr = lr
        self.initial_lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr

    def step(self) -> None:
        params = self.model.parameters()
        grads = self.model.gradients()

        if len(params) != len(grads):
            raise ValueError("The number of parameters and gradients must match.")

        for param, grad in zip(params, grads):
            param -= self.lr * grad

    def zero_grad(self) -> None:
        """
        这里主要是把各层缓存的梯度清零。
        因为你的 backward 每次都会重新写 dW / db，
        实际上不是必须，但保留这个接口会更规范。
        """
        for layer in self.model.get_linear_layers():
            layer.dW.fill(0.0)
            layer.db.fill(0.0)

    def decay_lr(self) -> None:
        """
        每个 epoch 后调用一次，例如:
            new_lr = old_lr * lr_decay
        """
        self.lr = max(self.lr * self.lr_decay, self.min_lr)

    def set_lr(self, new_lr: float) -> None:
        self.lr = max(new_lr, self.min_lr)

    def get_lr(self) -> float:
        return self.lr