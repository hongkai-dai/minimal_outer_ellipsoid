import numpy as np
import pytest  # noqa
import pydrake.solvers as solvers
import pydrake.symbolic as sym

import minimal_outer_ellipsoid.ellipsoid as mut


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
    inner_ineq_polynomials = np.array(
        [
            sym.Polynomial(
                x.dot(S_inner @ x) + b_inner.dot(x) + c_inner,
                x_set,
            )
        ]
    )
    inner_eq_polynomials = np.empty(())
    inner_ineq_poly_lagrangians_degrees = [0]
    inner_eq_poly_lagrangians_degrees = []
    (
        inner_ineq_poly_lagrangians,
        inner_eq_poly_lagrangians,
    ) = mut.add_containment_constraint(
        prog,
        x,
        inner_ineq_polynomials,
        inner_eq_polynomials,
        S,
        b,
        c,
        inner_ineq_poly_lagrangians_degrees,
        inner_eq_poly_lagrangians_degrees,
    )
    assert inner_ineq_poly_lagrangians.shape == (1,)
    assert inner_ineq_poly_lagrangians[0].TotalDegree() == 0
    assert inner_eq_poly_lagrangians.size == 0
    result = solvers.Solve(prog)
    assert result.is_success()
    inner_ineq_poly_lagrangians_sol = result.GetSolution(
        inner_ineq_poly_lagrangians[0]
    ).Evaluate({})
    assert inner_ineq_poly_lagrangians_sol >= 0

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

    inner_ineq_polynomials = np.array(
        [
            sym.Polynomial(-x[0]),
            sym.Polynomial(-x[1]),
            sym.Polynomial(-x[2]),
            sym.Polynomial(x[0] + x[1] + x[2] - 1),
        ]
    )
    inner_eq_polynomials = np.empty(())
    (
        inner_ineq_poly_lagrangians,
        inner_eq_poly_lagrangians,
    ) = mut.add_containment_constraint(
        prog,
        x,
        inner_ineq_polynomials,
        inner_eq_polynomials,
        S,
        b,
        c,
        inner_ineq_poly_lagrangians_degrees=[2] * 4,
        inner_eq_poly_lagrangians_degrees=[],
    )
    assert inner_ineq_poly_lagrangians.shape == (4,)
    assert all(
        [poly.TotalDegree() == 2 for poly in inner_ineq_poly_lagrangians]
    )
    assert inner_eq_poly_lagrangians.size == 0

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


def test_add_minimize_volume_cost():
    """
    Find the smallest outer ellipsoid that covers a given ellipsoid. The
    smallest outer ellipsoid is this inner ellipsoid.
    """

    prog = solvers.MathematicalProgram()
    dim = 3
    S = prog.NewSymmetricContinuousVariables(dim, "S")
    b = prog.NewContinuousVariables(dim, "b")
    c = prog.NewContinuousVariables(1, "c")[0]

    S_inner = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5.0]])
    S_inner = S_inner.T @ S_inner + 10 * np.eye(dim)
    b_inner = np.array([2, 4, 5.0])
    c_inner = b_inner.dot(np.linalg.solve(S_inner, b_inner)) / 4 - 0.5

    # By Schur complement, the condition that ellipsoid {x | x'Sx+b'x+c<=0}
    # contains {x | x'S_innerx+b_inner'x+c_inner<=0} is that there exists
    # γ>=0, such that the matrix
    # [γS_inner-S       (γb_inner - b)/2] is psd.
    # [(γb_inner-b)'/2       γc_inner-c ]
    gamma = prog.NewContinuousVariables(1, "gamma")[0]
    prog.AddBoundingBoxConstraint(0, np.inf, gamma)
    psd_mat = np.empty((dim + 1, dim + 1), dtype=object)
    psd_mat[:dim, :dim] = gamma * S_inner - S
    psd_mat[:dim, -1] = (gamma * b_inner - b) / 2
    psd_mat[-1, :dim] = (gamma * b_inner - b) / 2
    psd_mat[-1, -1] = gamma * c_inner - c
    prog.AddPositiveSemidefiniteConstraint(psd_mat)

    t = mut.add_minimize_volume_cost(prog, S, b, c)

    result = solvers.Solve(prog)
    assert result.is_success()
    S_sol = result.GetSolution(S)
    b_sol = result.GetSolution(b)
    c_sol = result.GetSolution(c)

    ratio = c_sol / c_inner
    np.testing.assert_allclose(S_sol, S_inner * ratio)
    np.testing.assert_allclose(b_sol, b_inner * ratio)
    t_sol = result.GetSolution(t)
    np.testing.assert_almost_equal(
        t_sol, b_sol.dot(np.linalg.solve(S_sol, b_sol)) / 4 - c_sol
    )


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


def test_to_affine_ball():
    def check(S, b, c):
        A, d = mut.to_affine_ball(S, b, c)

        # The ellipsoid is {x | xᵀA⁻ᵀA⁻¹x − 2dᵀA⁻¹x + dᵀd−1 ≤ 0}
        ratio = (d.dot(d) - 1) / c
        np.testing.assert_allclose(np.linalg.inv(A @ A.T), S * ratio)
        np.testing.assert_allclose(-2 * np.linalg.solve(A.T, d), b * ratio)

    check(np.eye(2), np.zeros(2), 1)
    check(np.diag(np.array([1.0, 2.0, 3.0])), np.array([2.0, 3.0, 4.0]), -10)
