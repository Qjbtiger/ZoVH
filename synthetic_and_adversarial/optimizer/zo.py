import torch
from typing import Iterator, Tuple
from model.synthetic_functions import SyntheticFunction

class ZerothOrderOptimizer(torch.optim.Optimizer):
    """
    Base class for zeroth-order optimizers.
    """
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not epsilon >= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not num_queries > 0:
            raise ValueError(f"Invalid number of queries: {num_queries}")
        if not mu >= 0.0:
            raise ValueError(f"Invalid mu value: {mu}")

        self.num_queries = num_queries
        self.mu = mu
        self.update_rule = update_rule
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.epsilon = epsilon

        super().__init__(params, defaults=dict())

    def _generate_noise(
        self,
        closure: SyntheticFunction,
        num_queries: int = 1,
        orthogonal: bool = False,
        previous_noise = None,
    ) -> torch.Tensor:
        if orthogonal:
            if previous_noise is None:
                num_all = num_queries
            else:
                num_all = previous_noise.shape[0] + num_queries
                previous_noise = previous_noise.t()
            _num = num_queries
            while _num > 0:
                noise = torch.randn(num_queries, closure.dim, device=closure.x.device).t()
                if previous_noise is not None:
                    noise = torch.cat([previous_noise, noise], dim=1)
                previous_noise, _ = torch.linalg.qr(noise)
                _num = num_all - previous_noise.shape[1]
            noise = previous_noise[:,-num_queries:].t()
            noise.mul_(closure.dim ** 0.5)
            # noise.div_(torch.norm(noise, dim=1, keepdim=True) + 1e-10)
        else:
            noise = torch.randn(num_queries, closure.dim, device=closure.x.device)
            # noise = torch.randn(num_queries, closure.dim, device=closure.x.device) / (closure.dim ** 0.5)

        return noise

    def _perturb_params(
        self,
        closure: SyntheticFunction,
        noise: torch.Tensor,
        mu: float
    ):
        closure.x.add_(mu * noise)

    def estimate_gradient(self,
        closure: SyntheticFunction
    ) -> torch.Tensor:
        """
        Estimate the gradient using finite differences.
        """
        
        raise NotImplementedError("This method should be implemented in subclasses.")

    @torch.no_grad()
    def step(
        self,
        closure: SyntheticFunction
    ) -> torch.Tensor:
        assert closure is not None, "Closure function is required for zeroth order optimization"

        loss = self.estimate_gradient(closure)
        param = closure.x
        grad = param.grad

        if self.update_rule == 'sgd':
            param.add_(-self.lr * grad)
        else: # adam radazo
            state = self.state[param]

            if len(state) == 0:
                state['step'] = 0
                state['m'] = torch.zeros_like(param)
                state['v'] = torch.zeros_like(param)

            m, v = state['m'], state['v']
            lr = self.lr
            beta1, beta2 = self.beta1, self.beta2
            epsilon = self.epsilon

            state['step'] += 1

            m.mul_(beta1).add_((1 - beta1) * grad)
            if self.update_rule == 'radazo':
                v.mul_(beta2).add_((1 - beta2) * (m ** 2))
            else:
                v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
            # m_hat = m / (1 - beta1 ** state['step'])
            # v_hat = v / (1 - beta2 ** state['step'])
            m_hat = m
            v_hat = v
            param.add_(-lr * m_hat / (v_hat.sqrt() + epsilon))

        return loss

class ZoVanilla(ZerothOrderOptimizer):
    def estimate_gradient(self, closure):
        """
        Estimate the gradient using finite differences.
        """
        loss = closure()

        noises = self._generate_noise(closure, num_queries=self.num_queries)
        fs = []
        for q in range(self.num_queries):
            self._perturb_params(closure, noises[q], self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h.item())
            self._perturb_params(closure, noises[q], -self.mu)

        fs = torch.tensor(fs, device=loss.device)
        # fs_baseline = torch.mean(fs) # a large mu should use this one, while a smaller one should use the following one
        fs_baseline = loss
        ws = (fs - fs_baseline) / self.mu  # shape: (num_queries,)
        
        closure.x.grad = torch.sum(ws.unsqueeze(-1) * noises, dim=0)

        closure.x.grad.div_(self.num_queries - 1) # ZO algorithm divide by (the number of queries - 1)

        # print(f"Estimated hyper_grad_x norm: {torch.norm(closure.x.grad).item():.4f}")
        # print("=" * 20)

        return loss

class ZoAR(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.99),
        epsilon: float = 1e-8,
        num_queries: int = 10,
        mu: float = 0.01,
        update_rule: str = 'radazo',
        num_histories: int = 5,
    ):
        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.num_histories = num_histories
        self.past = []

    def estimate_gradient(
        self,
        closure: SyntheticFunction
    ) -> torch.Tensor:
        """
            Estimate the gradient using finite differences with history.
        """

        loss = closure()

        new_noise = self._generate_noise(closure, num_queries=self.num_queries)
        for q in range(self.num_queries):
            self._perturb_params(closure, new_noise[q], self.mu)
            reward = closure()
            self._perturb_params(closure, new_noise[q], -self.mu)

            self.past.append([new_noise[q], reward])

        if len(self.past) > (self.num_histories + 1) * self.num_queries:
            self.past = self.past[-(self.num_histories + 1) * self.num_queries:]
        
        noises = [p[0] for p in self.past]
        noises = torch.stack(noises)  # shape: (num_queries * (num_histories + 1), param_shape)
        rewards = [p[1] for p in self.past]
        rewards = torch.stack(rewards)
        rewards = rewards - rewards.mean()
        
        ws = rewards / self.mu  # shape: (num_queries * (num_histories + 1),)

        closure.x.grad = torch.sum(ws.unsqueeze(-1) * noises, dim=0)

        closure.x.grad.div_(len(rewards) - 1) # rl w/ history algorithm divide by the number of rewards

        return loss