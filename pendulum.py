import torch as th
from torch import tensor, Tensor
from functools import cached_property


class PendulumAffineModel:
    def __init__(self, mass=1, length=1, damper_coeffs=0, gravity=9.81):
        self.mass, self.length, self.b = mass, length, damper_coeffs
        self.gv = gravity

    def f(self, state):
        theta, dtheta = state
        return (
            tensor([
                dtheta,
                - (self.gv*th.sin(theta))/self.length
                - (self.b*dtheta)/(self.mass * th.pow(th.tensor(self.length), 2))
            ])[:, None]
        )

    def g(self, state):
        return tensor([0, 1/(self.mass * th.pow(th.tensor(self.length), 2))])[:, None]

    def dynamics(self, state, control):
        return self.f(state) + self.g(state)@control


class PendulumLinearModel:
    def __init__(self, mass=1, length=1, damper_coeffs=0, gravity=9.81):
        self.mass, self.length, self.b = mass, length, damper_coeffs
        self.g = gravity

    @cached_property
    def state_matrix(self):
        return tensor([
            [0, 1],
            [-self.g/self.length, 0]
        ])

    @cached_property
    def control_matrix(self):
        return tensor([[0], [1/(self.mass*(self.length**2))]])

    def dynamics(self, state, control):
        return (self.state_matrix @ state) + (self.control_matrix @ control)
