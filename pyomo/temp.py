import pyomo.environ as pyo 
SOLVER = pyo.SolverFactory('ipopt')

from IPython.display import Markdown, HTML
import pyomo.environ as pyo
import numpy as np
from scipy.stats import norm

# set our risk threshold and risk levels (sometimes you may get an infeasible
# problem if the chance constraint becomes too tight!)
alpha = 0.6
beta = 0.3

# specify the initial capital, the risk-free return the number of risky assets,
# their expected returns, and their covariance matrix.
C = 1
R = 1.25
n = 3
mu = np.array([1.25, 1.15, 1.35])
Sigma = np.array([[1.5, 0.5, 2], [0.5, 2, 0], [2, 0, 5]])

# Check how dramatically the optimal solution changes if we assume i.i.d.
# deviations for the returns. # Sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# To change covariance matrix, make sure you input a semi-definite positive one.
# The easiest way to generate a random covariance matrix is first generating
# a random m x m matrix A and then taking the matrix A^T A (which is always
# semi-definite positive)
# m = 3
# A = np.random.rand(m, m)
# Sigma = A.T @ A


def markowitz_chanceconstraints(alpha, beta, mu, Sigma):
    model = pyo.ConcreteModel("Markowitz portfolio problem with chance constraints")

    model.x = pyo.Var(range(n), domain=pyo.NonNegativeReals)
    model.xtilde = pyo.Var(domain=pyo.NonNegativeReals)

    @model.Objective(sense=pyo.maximize)
    def objective(m):
        a = mu @ m.x + R * m.xtilde
        # if a < pyo.value(m.x[0]):
        return a


    @model.Constraint()
    def chance_constraint(m):
        # we use the inverse CDF of the standard normal norm.ppf() in the library scipy.stats
        return norm.ppf(1 - beta) * (m.x @ (Sigma @ m.x)) <= (mu @ m.x - alpha)

    @model.Constraint()
    def total_assets(m):
        return m.xtilde + sum(m.x[i] for i in range(n)) == C

    return model


model = markowitz_chanceconstraints(alpha, beta, mu, Sigma)
result = SOLVER.solve(model)

print(
    f"Solver status: {result.solver.status}, Termination condition: {result.solver.termination_condition}"
)
print(f"Solution: ", end="")
print(
    f"xtilde = {model.xtilde.value:.3f}, x_1 = {model.x[0].value:.3f}, x_2 = {model.x[1].value:.3f}, x_3 = {model.x[2].value:.3f}"
)
print(f"Maximum objective value: {model.objective():.2f}")