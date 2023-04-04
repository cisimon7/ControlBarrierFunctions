import sys
import torch as th
from qp_solve import qp_solve
from time import perf_counter
import matplotlib.pyplot as plt
from torch import tensor, Tensor
from rk4 import RungeKutta4thOrder
from pendulum import PendulumAffineModel


def get_control(x, time, solver="cvxpy"):

    system = PendulumAffineModel()

    V   = lambda x: x.T @ x
    dV  = lambda x: 2 * x.T
    LfV = lambda x: dV(x) @ system.f(x)
    LgV = lambda x: dV(x) @ system.g(x)

    def control_law(x, x_d=th.tensor([th.pi, 0.])[:, None], k=th.tensor([1., 1.])):
        # Feedback linearizing control law
        m, g, L, b = system.mass, system.gv, system.length, system.b
        x1, x2 = x
        return - m * L**2 * (k @ (x - x_d)) + (m * g * L * th.sin(x1)) + (b * x2)

    u_abs, lamda, p = 10., 1., 1
    u_min, u_max, u_ref = - tensor([u_abs]), + tensor([u_abs]), control_law(x)
    H = tensor([[1.]])

    Q_qp = th.block_diag(H, tensor(p))
    p_qp = tensor([-2 * u_ref @ H, 0])
    A_qp = tensor([
        [LgV(x), -1],
        [     1,  0],
        [    -1,  0]
    ])
    b_qp = tensor([
        - LfV(x) - lamda * V(x),
        u_max,
        -u_min
    ])

    u_safe, delta = th.from_numpy(
        qp_solve(Q_qp.numpy(), p_qp.numpy(), A_qp.numpy(), b_qp.numpy(), lib=solver)
    )
    u_safe = u_safe[:, None].float()

    integrator = RungeKutta4thOrder(
        step_sz=0.01,
        ode_func=lambda time, state: system.dynamics(state, u_safe)
    )
    x_next = integrator.step(x, time)
    return x_next, u_safe


def plot(states, controls):

    fig = plt.figure(layout="constrained")
    sub_figs = fig.subfigures(nrows=2, ncols=1, wspace=0.07, height_ratios=[0.5, 0.5])

    axs0 = sub_figs[0].subplots(nrows=1, ncols=2)
    axs0[0].plot(states[0, :], states[1, :], color="g")
    axs0[0].set_title("Phase Plot")
    axs0[0].set_xlabel("position")
    axs0[0].set_ylabel("velocity")

    axs0[1].plot(controls.flatten(), color="r")
    axs0[1].set_title("Control Plot")
    axs0[1].set_xlabel("time")
    axs0[1].set_ylabel("control")

    axs1 = sub_figs[1].subplots(nrows=1, ncols=2)
    axs1[0].plot(states[0, :])
    axs1[0].set_title("Position Plot")
    axs1[0].set_xlabel("time")
    axs1[0].set_ylabel("position")

    axs1[1].plot(states[1, :])
    axs1[1].set_title("Velocity Plot")
    axs1[1].set_xlabel("time")
    axs1[1].set_ylabel("velocity")

    plt.show()


if __name__ == "__main__":

    states = []
    controls = []

    x = tensor([0., 0.])[:, None]
    states.append(x)

    time_start = perf_counter()
    for time in th.arange(1_000):
        x_next, u_safe = get_control(x, time, solver="casadi")
        states.append(x_next)
        controls.append(u_safe)
        x = x_next
    time_end = perf_counter()
    print(round(time_end - time_start, 4), "seconds")

    states = th.hstack(states).numpy()
    controls = th.hstack(controls).numpy()

    plot(states, controls)
