import math
import torch
from typing import Iterator, Tuple
from optimizer.zo import ZerothOrderOptimizer
from model.synthetic_functions import SyntheticFunction

class ZoHessVanilla(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        num_queries: int = 10,
        mu: float = 0.01,
        lambda_hess: float = 1.0,
        orthogonal: bool = False
    ):
        super().__init__(params, lr, num_queries=num_queries, mu=mu, update_rule='sgd')

        self.lambda_hess = lambda_hess
        self.orthogonal = orthogonal

    def estimate_gradient(
        self,
        closure: SyntheticFunction
    ) -> torch.Tensor:
        """
        Estimate the gradient using finite differences.
        """

        loss = closure()

        noises = self._generate_noise(closure, num_queries=self.num_queries, orthogonal=self.orthogonal)
        fs = []
        for q in range(self.num_queries):
            self._perturb_params(closure, noises[q], self.mu)
            f_x_plus_h = closure()
            fs.append(f_x_plus_h)
            self._perturb_params(closure, noises[q], -self.mu)

        fs = torch.tensor(fs, device=loss.device)
        noises_norm_sq = torch.sum(noises ** 2, dim=1)  # shape: (num_queries,)
        fs_baseline = torch.mean(fs)
        # fs_baseline = loss

        ws_grad = (fs - fs_baseline) / self.mu  # shape: (num_queries)
        grad_estimate = torch.sum(ws_grad.unsqueeze(-1) * noises, dim=0)
        grad_estimate.div_(self.num_queries - 1)
        
        ws_hess = (fs - fs_baseline) / (self.mu ** 2)  # shape: (num_queries)
        inv_hess_grad_coffs = ws_hess  / (self.lambda_hess * (self.num_queries - 1) + ws_hess * noises_norm_sq) # shape: (num_queries,)

        grad_noises = (torch.mv(noises, grad_estimate) * (self.num_queries - 1) - ws_grad * noises_norm_sq) / (self.num_queries - 2)
        term = torch.sum((inv_hess_grad_coffs * grad_noises).unsqueeze(1) * noises, dim=0)
        
        closure.x.grad = (grad_estimate - term) / self.lambda_hess


        return loss

class ZoARHess(ZerothOrderOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.001,
        num_queries: int = 10,
        mu: float = 0.01,
        lambda_hess: float = 1.0,
        orthogonal: bool = False,
        num_histories: int = 5,
    ):
        super().__init__(params, lr, num_queries=num_queries, mu=mu, update_rule='sgd')
        self.num_histories = num_histories
        self.lambda_hess = lambda_hess
        self.orthogonal = orthogonal
        self.past = []

    def estimate_gradient(
        self,
        closure: SyntheticFunction
    ) -> torch.Tensor:
        """
            Estimate the gradient using finite differences with history.
        """

        loss = closure()

        # make sure past size is within max capacity (self.num_histories + 1) * self.num_queries
        if len(self.past) > self.num_histories * self.num_queries:
            self.past = self.past[-self.num_histories * self.num_queries:]

        if len(self.past) > 0:
            previous_noises = [p[0] for p in self.past]
            previous_noises = torch.stack(previous_noises)  # shape: (num_queries * num_histories, param_shape)
        else:
            previous_noises = None
        new_noise = self._generate_noise(closure, num_queries=self.num_queries, orthogonal=self.orthogonal, previous_noise=previous_noises)

        for q in range(self.num_queries):
            self._perturb_params(closure, new_noise[q], self.mu)
            reward = closure()
            self._perturb_params(closure, new_noise[q], -self.mu)

            self.past.append([new_noise[q], reward])

        noises = torch.cat([previous_noises, new_noise], dim=0) if previous_noises is not None else new_noise
        noises_norm_sq = torch.sum(noises ** 2, dim=1)  # shape: (num_queries * (num_histories + 1),)
        rewards = [p[1] for p in self.past] # shape: (num_queries * (num_histories + 1),)
        rewards = torch.stack(rewards)
        
        ws_grad = (rewards - rewards.mean()) / self.mu  # shape: (num_queries * (num_histories + 1),)

        grad_estimate = torch.sum((self.mu * ws_grad).unsqueeze(-1) * noises, dim=0)
        grad_estimate.div_(len(rewards) - 1)

        ws_hess = (rewards - rewards.mean()) / (self.mu ** 2)  # shape: (num_queries * (num_histories + 1),)
        inv_hess_grad_coffs = ws_hess / (self.lambda_hess * (len(rewards) - 1) + ws_hess * noises_norm_sq) # shape: (num_queries * (num_histories + 1),)

        grad_noises = (torch.mv(noises, grad_estimate) * (len(rewards) - 1) - ws_grad * noises_norm_sq) / (len(rewards) - 2)
        term = torch.sum((inv_hess_grad_coffs * grad_noises).unsqueeze(1) * noises, dim=0)


        closure.x.grad = (grad_estimate - term) / self.lambda_hess


        return loss

    




class HiZOO(ZerothOrderOptimizer):
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
        assert (num_queries - 1) % 2 == 0, "HiZOO requires num_queries to be even."
        num_queries = (num_queries - 1) // 2

        super().__init__(params, lr, betas, epsilon, num_queries, mu, update_rule)

        self.hess_smooth = 1e-8
        # self.hess_smooth = 1.0

    def _perturb_params(
        self,
        closure: SyntheticFunction,
        noise: torch.Tensor,
        mu: float,
        hess_diag: torch.Tensor
    ):
        closure.x.add_(mu * noise / hess_diag)

    def estimate_gradient(
        self,
        closure: SyntheticFunction
    ) -> torch.Tensor:
        """
        Estimate the gradient using finite differences.
        """
        if not hasattr(self, 'hess_diag'):
            self.hess_diag = torch.ones_like(closure.x)
        loss = closure()

        noises = self._generate_noise(closure, num_queries=self.num_queries)
        fs_plus = []
        fs_minus = []
        for q in range(self.num_queries):
            self._perturb_params(closure, noises[q], self.mu, self.hess_diag)
            fs_plus.append(closure().item())
            self._perturb_params(closure, noises[q], -2 * self.mu, self.hess_diag)
            f_x_minus_h = closure()
            fs_minus.append(f_x_minus_h.item())
            self._perturb_params(closure, noises[q], self.mu, self.hess_diag)

        fs_plus = torch.tensor(fs_plus, device=loss.device)
        fs_minus = torch.tensor(fs_minus, device=loss.device)
        fs_baseline = loss.item()

        ws = torch.abs(fs_plus + fs_minus - 2 * fs_baseline) / (2 * self.mu ** 2)  # shape: (num_queries,)
        hess_diag_current = torch.sum(ws.unsqueeze(-1) * self.hess_diag.unsqueeze(0) * (noises ** 2), dim=0) / self.num_queries  # shape: (param_shape,)

        self.hess_diag.mul_(1 - self.hess_smooth).add_(self.hess_smooth * hess_diag_current)
        
        ws = (fs_plus - fs_minus) / (2 * self.mu)  # shape: (num_queries,)

        closure.x.grad = torch.sum(ws.unsqueeze(-1) * self.hess_diag.unsqueeze(0) * noises, dim=0)

        closure.x.grad.div_(self.num_queries) # ZO algorithm divide by the number of queries

        return loss


