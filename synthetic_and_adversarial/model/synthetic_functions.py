import torch
import torch.nn as nn

class SyntheticFunction(nn.Module):
    def __init__(
        self,
        x_init: torch.Tensor
    ):
        super(SyntheticFunction, self).__init__()
        
        assert len(x_init.shape) == 1, "x_init must be a 1D tensor"

        self.dim = x_init.shape[0]
        self.x = torch.nn.Parameter(x_init)

    # def grad(self) -> torch.Tensor:
    #     raise NotImplementedError("This method should be implemented in subclasses.")
    

class Levy(SyntheticFunction):
    def forward(self) -> torch.Tensor:
        x = self.x
        w = 1 + (x - 1) / 4

        term1 = torch.sin(torch.pi * w[0]) ** 2
        term2 = ((w[-1] - 1) ** 2) * (1 + torch.sin(2 * torch.pi * w[-1]) ** 2)
        term3 = torch.sum((w[:-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[:-1] + 1) ** 2))

        return term1 + term2 + term3
    
    def grad(self) -> torch.Tensor:
        x = self.x
        w = 1 + (x - 1) / 4

        grad = torch.zeros_like(x)
        dw_dx = 0.25

        # middle terms for i = 0..d-2 (vectorized)
        w_mid = w[:-1]
        wm1   = w_mid - 1
        s_mid = torch.sin(torch.pi * w_mid + 1)
        c_mid = torch.cos(torch.pi * w_mid + 1)

        B_mid  = 1 + 10 * s_mid ** 2
        dB_mid = 20 * torch.pi * s_mid * c_mid

        d_mid_dw = 2 * wm1 * B_mid + wm1 ** 2 * dB_mid
        grad[:-1] = d_mid_dw * dw_dx

        # add term1 contribution for i = 0
        d_term1_dw0 = torch.pi * torch.sin(2 * torch.pi * w[0])
        grad[0] += d_term1_dw0 * dw_dx

        # last term for i = d-1
        w_last  = w[-1]
        wl1     = w_last - 1
        s_last2 = torch.sin(2 * torch.pi * w_last)
        c_last2 = torch.cos(2 * torch.pi * w_last)

        d_last_dw = (
            2 * wl1 * (1 + s_last2 ** 2)
            + 4 * torch.pi * wl1 ** 2 * s_last2 * c_last2
        )
        grad[-1] = d_last_dw * dw_dx

        return grad

class Rosenbrock(SyntheticFunction):
    def forward(self) -> torch.Tensor:
        x = self.x

        term1 = 100 * (x[1:] - x[:-1] ** 2) ** 2
        term2 = (1 - x[:-1]) ** 2

        return torch.sum(term1 + term2)
    
    def grad(self) -> torch.Tensor:
        x = self.x
        grad = torch.zeros_like(x)

        # internal indices 1..d-2
        grad[1:-1] = (
            200 * (x[1:-1] - x[:-2] ** 2)
            - 400 * x[1:-1] * (x[2:] - x[1:-1] ** 2)
            - 2 * (1 - x[1:-1])
        )

        # i = 0
        grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])

        # i = d-1
        grad[-1] = 200 * (x[-1] - x[-2] ** 2)

        return grad


class Ackley(SyntheticFunction):

    def forward(self) -> torch.Tensor:
        x = self.x
        a, b, c = 20, 0.2, 2 * torch.pi
        d = self.dim

        term1 = - a * torch.exp(-b * torch.sqrt(torch.sum(x ** 2) / d))
        term2 = - torch.exp(torch.sum(torch.cos(c * x)) / d)

        return term1 + term2 + a + torch.e
    
    def grad(self) -> torch.Tensor:
        x = self.x
        a, b, c = 20, 0.2, 2 * torch.pi
        d = self.dim

        sum_sq = torch.sum(x ** 2)
        sum_cos = torch.sum(torch.cos(c * x))

        # avoid division by zero when x == 0
        if sum_sq == 0:
            coeff1 = 0.0
        else:
            r = torch.sqrt(sum_sq / d)
            coeff1 = (a * b / d) * torch.exp(-b * r) / r

        coeff2 = (c / d) * torch.exp(sum_cos / d)

        # fully vectorized
        grad = coeff1 * x + coeff2 * torch.sin(c * x)

        return grad


class Quadratic(SyntheticFunction):
    def forward(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.x ** 2)
    
    def grad(self) -> torch.Tensor:
        return self.x
    
    def hess(self) -> torch.Tensor:
        return torch.eye(self.dim, device=self.x.device)

def get_synthetic_funcs(
    name: str,
    x_init: torch.Tensor
) -> SyntheticFunction:
    
    all_functions = {
        "ackley": Ackley,
        "levy": Levy,
        "rosenbrock": Rosenbrock,
        "quadratic": Quadratic,
    }

    assert name in all_functions, f"Function {name} not found. Available functions: {list(all_functions.keys())}"
    
    return all_functions[name](x_init)

def test_code():
    torch.set_default_dtype(torch.float64)
    d = 10
    seed = 42
    tol = 1e-5
    FunctionClasses = [Ackley, Levy, Rosenbrock, Quadratic]

    for FunctionClass in FunctionClasses:
        print(f"\n=== Testing problem: {FunctionClass.__name__} ===")
        x_init = torch.randn(d, requires_grad=True)

        torch.manual_seed(seed)
        func = FunctionClass(x_init)

        # ---------- (1) Evaluate gradients ----------
        func.zero_grad()
        value = func()
        value.backward()
        grad_autograd = func.x.grad.detach().clone()
        grad_analytic = func.grad().detach()

        err_x = torch.max(torch.abs(grad_autograd - grad_analytic)).item()

        print(f"max |∇_x f(auto) - ∇_x f(analytic)| = {err_x:.3e}")

        if max(err_x, 0) < tol:
            print("All checks within tolerance.")
        else:
            print("WARNING: some checks exceed tolerance.")

if __name__ == "__main__":
    test_code()