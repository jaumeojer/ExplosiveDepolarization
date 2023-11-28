import numpy as np
import scipy as sp
from scipy.integrate import ode, odeint

def vode_solver(func, x0, t_max, parameters):
    t0 = 0

    backend = "vode"
    method = "bdf"

    solver = ode(func).set_integrator(backend, method=method)

    sol = []
    t = []

    solver.set_initial_value(x0, t0).set_f_params(*parameters)

    while solver.successful() and solver.t < t_max:
        solver.integrate(t_max, step=True)
        t.append(solver.t)
        sol.append(solver.y)

    t = np.array(t)

    sol = np.array(sol)

    return t, sol.T


def vode_solver_cutoff(func, x0, t_max, pre, parameters):
    t0 = 0

    backend = "vode"
    method = "bdf"

    solver = ode(func).set_integrator(backend, method=method)

    sol = []
    t = []

    solver.set_initial_value(x0, t0).set_f_params(*parameters)
    old = x0
    err = np.ones(len(x0))*pre

    while solver.successful() and solver.t < t_max:
        solver.integrate(t_max, step=True)
        t.append(solver.t)
        sol.append(solver.y)
        if (np.absolute(solver.y - old) < err).all() == True:
            break
        old = solver.y

    t = np.array(t)

    sol = np.array(sol)

    return t, sol.T


def vode_solver_final_state(func, x0, t_max, parameters):
    t0 = 0

    backend = "vode"
    method = "bdf"

    solver = ode(func).set_integrator(backend, method=method)

    solver.set_initial_value(x0, t0).set_f_params(*parameters)

    while solver.successful() and solver.t < t_max:
        solver.integrate(t_max, step=True)

    return solver.t, solver.y.T


def vode_solver_final_state_cutoff(func, x0, t_max, pre, parameters):
    t0 = 0

    backend = "vode"
    method = "bdf"

    solver = ode(func).set_integrator(backend, method=method)

    solver.set_initial_value(x0, t0).set_f_params(*parameters)
    old = x0
    err = np.ones(len(x0))*pre

    while solver.successful() and solver.t < t_max:
        solver.integrate(t_max, step=True)
        if (np.absolute(solver.y - old) < err).all() == True:
            break
        old = solver.y

    return solver.t, solver.y.T


def vode_solver_uniform(func, x0, t_max, dt, parameters):
    t0 = 0

    backend = "vode"
    method = "bdf"

    solver = ode(func).set_integrator(backend, nsteps=500, method=method)

    sol = []
    t = []

    solver.set_initial_value(x0, t0).set_f_params(*parameters)

    while solver.successful() and solver.t < t_max:
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        sol.append(solver.y)

    t = np.array(t)

    sol = np.array(sol)

    return t, sol.T


def odeint_solver(func, x0, t_max, dt, args=()):

    t = np.arange(0, t_max, dt)

    sol = odeint(func, x0, t, args=args)

    return t, sol.T
