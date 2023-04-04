import torch as th
from qp_solve import qp_solve
from torch import tensor, Tensor
from pendulum import PendulumLinearModel


if __name__ == "__main__":

    system = PendulumLinearModel()
    A, B = system.state_matrix, system.control_matrix
    u_abs, delta, p = 1., 0.4, 1

    u_min, u_max, u_ref = -u_abs, +u_abs, 0.
    x = tensor([0., 0.])[:, None]
    u = tensor([0.])[:, None]

    H = tensor(1)
    Q_qp = th.block_diag(H, tensor(p))
    p_qp = tensor([-2*u_ref*H, 0])
    A_qp = tensor([
        [(2*x.mT@B) + (delta*x.mT@x), 0],
        [                          1, 0],
        [                         -1, 0]
    ])
    b_qp = tensor([2*x.mT@A@x, u_max, u_min])
    u_best = qp_solve(Q_qp.numpy(), p_qp.numpy(), A_qp.numpy(), b_qp.numpy())

    print(u_best)
    