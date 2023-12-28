from typing import List, Optional, Tuple, Union

import numpy as np
import pydrake.solvers as solvers
import pydrake.symbolic as sym


def add_containment_constraint(
    prog: solvers.MathematicalProgram,
    x: np.ndarray,
    inner_ineq_polynomials: np.ndarray,
    inner_eq_polynomials: np.ndarray,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
    inner_ineq_poly_lagrangians_degrees: List[int],
    inner_eq_poly_lagrangians_degrees: List[int],
    ellipsoid_lagrangian: Optional[sym.Polynomial] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add the constraint that a basic semi-algebraic set
    { x | pᵢ(x) <= 0, i=0, ..., N, qⱼ(x) = 0, j=0,...,M} is contained in
    the ellipsoid {x | xᵀSx+bᵀx+c≤ 0 }.

    A sufficient condition for the containment is
    −(1+β(x))*(xᵀSx+bᵀx+c) + ∑ᵢ₌₀ᴺλᵢ(x)pᵢ(x) + ∑ⱼ₌₀ᴹσⱼ(x)qⱼ(x) is sos
    β(x) is sos, λᵢ(x) is sos, i=1,..., N

    Args:
      prog: The mathematical program to which the constraint is added.
      x: The symbolic variables.
      inner_ineq_polynomials: An array of symbolic polynomials that defines the
        inequality constraint in the inner basic semi-algebraic set.
        inner_ineq_polynomials[i] is pᵢ(x) in the documentation.
      inner_eq_polynomials: An array of symbolic polynomials that defines the
      equality constraints in the inner basic semi-algebraic set.
      S: A symmetric matrix of symbolic variables. The parameters of the
        ellipsoid.
      b: A vector of symbolic variables. The parameters of the ellipsoid.
      c: A symbolic variable. The parameter of the ellipsoid.
      inner_ineq_poly_lagrangians_degrees:
        inner_ineq_poly_lagrangians_degrees[i] is the degree of the polynomial
        λᵢ(x).
      inner_eq_poly_lagrangians_degrees:
        inner_eq_poly_lagrangians_degrees[i] is the degree of the polynomial
        σᵢ(x).
      ellipsoid_lagrangian: The sos polynomial β(x) in the documentation. If
        None, then we use β(x)=0.

    Returns:
      inner_ineq_poly_lagrangians: An array of symbolic polynomials.
        inner_ineq_poly_lagrangians[i] is λᵢ(x) in the documentation.
      inner_eq_poly_lagrangians: An array of symbolic polynomials.
        inner_eq_poly_lagrangians[i] is σᵢ(x) in the documentation.
    """
    if ellipsoid_lagrangian is None:
        ellipsoid_lagrangian = sym.Polynomial()
    x_set = sym.Variables(x)
    ellipsoid = sym.Polynomial(x.dot(S @ x) + b.dot(x) + c, x_set)
    leading_poly = -(1 + ellipsoid_lagrangian) * ellipsoid
    assert (
        len(inner_ineq_poly_lagrangians_degrees) == inner_ineq_polynomials.size
    )
    inner_ineq_poly_lagrangians = np.empty(
        (inner_ineq_polynomials.size,), dtype=object
    )
    for i in range(inner_ineq_polynomials.size):
        if inner_ineq_poly_lagrangians_degrees[i] == 0:
            inner_ineq_poly_lagrangians_var = prog.NewContinuousVariables(1)[0]
            prog.AddBoundingBoxConstraint(
                0, np.inf, inner_ineq_poly_lagrangians_var
            )
            inner_ineq_poly_lagrangians[i] = sym.Polynomial(
                {
                    sym.Monomial(): sym.Expression(
                        inner_ineq_poly_lagrangians_var
                    )
                }
            )
        else:
            inner_ineq_poly_lagrangians[i], _ = prog.NewSosPolynomial(
                x_set, inner_ineq_poly_lagrangians_degrees[i]
            )
    inner_eq_poly_lagrangians = np.array(
        [
            prog.NewFreePolynomial(x_set, degree)
            for degree in inner_eq_poly_lagrangians_degrees
        ]
    )

    sos_condition = leading_poly
    if inner_ineq_poly_lagrangians.size != 0:
        sos_condition += inner_ineq_poly_lagrangians.dot(
            inner_ineq_polynomials
        )
    if inner_eq_poly_lagrangians.size != 0:
        sos_condition += inner_eq_poly_lagrangians.dot(inner_eq_polynomials)
    prog.AddSosConstraint(sos_condition)
    return inner_ineq_poly_lagrangians, inner_eq_poly_lagrangians


def add_minimize_volume_cost(
    prog: solvers.MathematicalProgram,
    S: np.ndarray,
    b: np.ndarray,
    c: sym.Variable,
) -> sym.Variable:
    """
    Add the cost to minimize the volume of the ellipsoid
    {x | xᵀSx + bᵀx + c <= 0}.

    As explained in the docs/formulation.pdf, we can minimize the volume with
    the following cost/constraint

    min t
    s.t ⌈ c+t  bᵀ/2⌉ ≽ 0
        ⌊ b/2     S⌋
        log(det(S)) >= 0

    Args:
      prog: The mathematical program to which the cost/constraint are added.
      S: A symmetric matrix of symbolic variables. The parameters of the
        ellipsoid.
      b: A vector of symbolic variables. The parameters of the ellipsoid.
      c: A symbolic variable. The parameter of the ellipsoid.

    Return:
      t: The slack variable representing the volume of the ellipsoid.
    """
    t = prog.NewContinuousVariables(1, "t")[0]
    dim = S.shape[0]
    psd_mat = np.empty((dim + 1, dim + 1), dtype=object)
    psd_mat[0, 0] = c + t
    psd_mat[0, 1:] = b.T / 2
    psd_mat[1:, 0] = b / 2
    psd_mat[1:, 1:] = S
    prog.AddPositiveSemidefiniteConstraint(psd_mat)
    prog.AddLogDeterminantLowerBoundConstraint(S, 0)
    prog.AddLinearCost(t)
    return t


def in_ellipsoid(
    S: np.ndarray, b: np.ndarray, c: float, pts: np.ndarray
) -> Union[bool, List[bool]]:
    """
    Return if `pts` is/are in the ellipsoid {x | x'*S*x+b'*x+c <= 0}.

    Args:
      pts: A single point or a group of points. Each row is a point.
    Return:
      flag: If pts is a 1-D array, then we return a single boolean. If pts is a
        2D array, then flag[i] indicates whether pts[i] is in the ellipsoid.
    """
    dim = S.shape[0]
    if pts.shape == (dim,):
        return pts.dot(S @ pts) + b.dot(pts) + c <= 0
    else:
        assert pts.shape[1] == dim
        return (np.sum(pts * (pts @ S), axis=1) + pts @ b + c <= 0).tolist()


def to_affine_ball(
    S: np.ndarray, b: np.ndarray, c: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the ellipsoid in the form {x | xᵀSx+bᵀx+c ≤ 0} to an alternative
    form as an affine transformation of a ball {A(y+d) | |y|₂ ≤ 1 }

    Args:
      S: A symmetric matrix that parameterizes the ellipsoid.
      b: A vector that parameterizes the ellipsoid.
      c: A float that parameterizes the ellipsoid.
    Returns:
      A: The affine transformation of the ball.
      d: The center of the ellipsoid.
    """

    # The ellipsoid can be written as
    # {x | xᵀA⁻ᵀA⁻¹x − 2dᵀA⁻¹x + dᵀd−1 ≤ 0}. To match it with xᵀSx+bᵀx+c ≤ 0,
    # we introduce a scalar k, with the constraint
    # A⁻ᵀA⁻¹ = kS
    # −2A⁻ᵀd = kb
    # dᵀd−1 = kc
    # To solve these equations, I define A̅ = A * √k, d̅ = d/√k
    # Hence I know A̅⁻ᵀA̅⁻¹ = S
    bar_A_inv = np.linalg.cholesky(S).T
    bar_A = np.linalg.inv(bar_A_inv)
    # −2A̅⁻ᵀd̅=b
    bar_d = bar_A.T @ b / -2
    # kd̅ᵀd̅−1 = kc
    k = 1 / (bar_d.dot(bar_d) - c)
    sqrt_k = np.sqrt(k)
    A = bar_A / sqrt_k
    d = bar_d * sqrt_k
    return (A, d)
