import torch as th
from numpy import asarray


"""
finds a vector x that minimizes the quadratic objective 1/2x.Q.x + p.x,
subject to the linear inequality constraints A.x <= b
"""
def qp_solve(Q, p, A, b, lib="cvxpy"):
    if lib=="cvxpy":
        import cvxpy as cp
        x = cp.Variable(Q.shape[0])
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(x, Q) + p.T @ x),
            [A @ x <= b]
        )
        prob.solve()
        status = prob.solution.status
        if status == "infeasible":
            raise Exception("Optimization problem not feasible")

        result = x.value
        return asarray(result)[:, None]
    if lib=="casadi":
        from casadi import DM, conic
        H, g, A, uba = DM(Q), DM(p), DM(A), DM(b)
        prob = conic("S", "qpoases", dict(h=H.sparsity(), a=A.sparsity()))
        sol = prob(h=H, g=g, a=A, uba=uba)

        return asarray(sol["x"])
