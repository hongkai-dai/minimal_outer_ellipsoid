import numpy as np
import pytest  # noqa
import pydrake.solvers as solvers
import pydrake.symbolic as sym

import minimum_outer_ellipsoid.ellipsoid as mut


def test_add_containment_constraint1():
    """
    Test if an ellipsoid contains another ellipsoid.
    """
    prog = solvers.MathematicalProgram()
    dim = 2
    x = prog.NewIndeterminates(dim, "x")
    x_set = sym.Variables(x)
    S = prog.NewSymmetricContinuousVariables(dim, "S")
    b = prog.NewContinuousVariables(dim, "b")
    c = prog.NewContinuousVariables(1, "c")[0]

    S_inner = np.diag(np.array([2.0, 3.0]))
    b_inner = np.array([1.0, 2.0])
    c_inner = b_inner.dot(np.linalg.solve(S_inner, b_inner)) / 4 - 0.5
    inner_polynomials = np.array(
        [
            sym.Polynomial(
                x.dot(S_inner @ x) + b_inner.dot(x) + c_inner,
                x_set,
            )
        ]
    )
    inner_poly_lagrangians_degrees = [0]
    inner_poly_lagrangians = mut.add_containment_constraint(
        prog, x, inner_polynomials, S, b, c, inner_poly_lagrangians_degrees
    )
    assert inner_poly_lagrangians.shape == (1,)
    assert inner_poly_lagrangians[0].TotalDegree() == 0
    result = solvers.Solve(prog)
    assert result.is_success()
    inner_poly_lagrangians_sol = result.GetSolution(
        inner_poly_lagrangians[0]
    ).Evaluate({})
    assert inner_poly_lagrangians_sol >= 0

    # Now sample many points, if the point is in the inner ellipsoid, then it
    # has to be in the outer ellipsoid.
    S_sol = result.GetSolution(S)
    b_sol = result.GetSolution(b)
    c_sol = result.GetSolution(c)

    x_samples = np.random.random((100, dim))
    is_in_inner = mut.in_ellipsoid(S_inner, b_inner, c_inner, x_samples)
    is_in_outer = mut.in_ellipsoid(S_sol, b_sol, c_sol, x_samples)
    assert all(np.array(is_in_outer)[is_in_inner])


def test_add_containment_constraint2():
    """
    Test an ellipsoid contains a tetrahedron.
    """
    dim = 3
    prog = solvers.MathematicalProgram()
    x = prog.NewIndeterminates(dim, "x")
    S = prog.NewSymmetricContinuousVariables(dim, "S")
    b = prog.NewContinuousVariables(dim, "b")
    c = prog.NewContinuousVariables(1, "c")[0]

    inner_polynomials = np.array(
        [
            sym.Polynomial(-x[0]),
            sym.Polynomial(-x[1]),
            sym.Polynomial(-x[2]),
            sym.Polynomial(x[0] + x[1] + x[2] - 1),
        ]
    )
    inner_poly_lagrangians = mut.add_containment_constraint(
        prog,
        x,
        inner_polynomials,
        S,
        b,
        c,
        inner_poly_lagrangians_degrees=[2] * 4,
    )
    assert inner_poly_lagrangians.shape == (4,)
    assert all([poly.TotalDegree() == 2 for poly in inner_poly_lagrangians])

    result = solvers.Solve(prog)
    assert result.is_success()

    # Sample many points. If the point is inside the tetrahedron, then it is
    # inside the ellipsoid.
    x_samples = np.random.random((500, dim))
    in_tetrahedron = np.logical_and(
        np.all(x_samples >= 0, axis=1), np.sum(x_samples, axis=1) <= 1
    )
    S_sol = result.GetSolution(S)
    b_sol = result.GetSolution(b)
    c_sol = result.GetSolution(c)
    in_outer_ellipsoid = mut.in_ellipsoid(S_sol, b_sol, c_sol, x_samples)
    assert all(np.array(in_outer_ellipsoid)[in_tetrahedron])


def test_in_ellipsoid():
    center = np.array([0, 1, 1.5])
    A = np.diag(np.array([1, 2, 3]))
    # Build an ellipsoid {A*(x+center) | |x|<=1 }
    A_inv = np.linalg.inv(A)
    S = A_inv.T @ A_inv
    b = -2 * A_inv @ center
    c = center.dot(center) - 1

    assert mut.in_ellipsoid(S, b, c, A @ center)
    assert mut.in_ellipsoid(S, b, c, A @ (center + np.array([0.5, 0.2, 0.3])))
    assert not mut.in_ellipsoid(S, b, c, A @ (center + np.array([1.1, 0, 0])))
    assert mut.in_ellipsoid(
        S,
        b,
        c,
        (center + np.array([[0.5, 0, 0.2], [1.1, 0, 0], [1.2, 0, -1]])) @ A.T,
    ) == [True, False, False]
