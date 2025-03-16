import numpy as np


# Initial conditions for DJ
def J(X, Y, u_0, y_min=0.4, y_max=0.6, h=0.005):
    return u_0 / 2 * (
        np.tanh((Y - y_min) / h) - np.tanh((Y - y_max) / h) - 1
    ), np.zeros_like(Y)


def d_1(X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
    return (
        2
        * L_box
        / h**2
        * (
            (Y - y_max) * np.exp(-((Y - y_max) ** 2) / h**2)
            + (Y - y_min) * np.exp(-((Y - y_min) ** 2) / h**2)
        )
        * (
            np.sin(8 * np.pi * X / L_box)
            + np.sin(24 * np.pi * X / L_box)
            + np.sin(6 * np.pi * X / L_box)
        )
    )


def d_2(X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
    return (
        np.pi
        * (np.exp(-((Y - y_max) ** 2) / h**2) + np.exp(-((Y - y_min) ** 2) / h**2))
        * (
            8 * np.cos(8 * np.pi * X / L_box)
            + 24 * np.cos(24 * np.pi * X / L_box)
            + 6 * np.cos(6 * np.pi * X / L_box)
        )
    )


def D(X, Y, u_0, y_min, y_max, h, L_box):
    d1 = d_1(X, Y, y_min, y_max, h, L_box)
    d2 = d_2(X, Y, y_min, y_max, h, L_box)
    delta = u_0 / (40 * np.max(np.sqrt(d1**2 + d2**2)))
    return delta * d1, delta * d2


def initial_fields(L=1, n=10, y_min=0.4, y_max=0.6, h=1 / 200, u_max=1):
    #L = 1
    N = 2 ** n
    dx = 1/N

    # create 2D grid
    x = np.linspace(0, L - dx, N)
    y = np.linspace(0, L - dx, N)
    Y, X = np.meshgrid(y, x)

    # load initial conditions for DJ
    U, V = J(X, Y, u_max, y_min, y_max, h)
    dU, dV = D(X, Y, u_max, y_min, y_max, h, L)
    U = U + dU
    V = V + dV

    return U, V
