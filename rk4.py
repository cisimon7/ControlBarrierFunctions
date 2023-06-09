import sys
import torch as th
from torch import Tensor
from typing import Callable

OdeFunc = Callable[[Tensor, Tensor], Tensor]


class RungeKutta4thOrder:
    def __init__(self, step_sz: float, ode_func: OdeFunc):
        self.step_sz = step_sz
        self.ode_func = ode_func

    def step(self, x0: Tensor, t0: Tensor) -> Tensor:
        k1 = th.mul(self.ode_func(t0, x0), self.step_sz)
        k2 = th.mul(self.ode_func(t0 + th.mul(self.step_sz, 0.5), x0 + th.mul(k1, 0.5)), self.step_sz)
        k3 = th.mul(self.ode_func(t0 + th.mul(self.step_sz, 0.5), x0 + th.mul(k2, 0.5)), self.step_sz)
        k4 = th.mul(self.ode_func(th.add(t0, self.step_sz), x0 + k3), self.step_sz)

        return x0 + th.mul(k1, 1/6) + th.mul(k2, 1/3) + th.mul(k3, 1/3) + th.mul(k4, 1/6)

    def steps(self, count: int, x0: Tensor, t0: float) -> Tensor:
        return th.vstack([x0 := self.step(x0, t0) for t0 in th.arange(t0, self.step_sz*count, self.step_sz)])

    def interval(self, x0: Tensor, t0: float, tf: float) -> Tensor:
        return th.vstack([x0 := self.step(x0, t0) for t0 in th.arange(t0, tf, self.step_sz)])