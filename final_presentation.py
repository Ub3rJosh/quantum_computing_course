import numpy as np
from scipy.sparse import diags


###   ###   ###   constants   ###   ###   ###
hbar = c = 1  # as god intended
mass = 50  # example
omega = 1  # example

n = 128  # number of elements (2^7 = 128, nice for qubits later)


###   ###   ###   handy functions   ###   ###   ###
def harmonic_oscillator_potential(x,
                                  omega=omega,
                                  m=mass):
    V_vector = 0.5 * m * omega ** 2 * x ** 2  # calculate the potential
    V_matrix = diags(V_vector).toarray()  # operator-ify that vector
    return V_matrix

###   ###   ###   the actual code   ###   ###   ###
x, dx = np.linspace(-5, 5, n, retstep=True)  # real-space (it's where it's at!)
# finite differencing 2nd derivative (good enough for toy problem):
d2_by_dx2 = diags([1., -2., 1.], [-1, 0, 1], shape=(n, n)).toarray() / (dx ** 2)

T = - (1 / (2 * mass)) * d2_by_dx2    # kinetic energy
V = harmonic_oscillator_potential(x)  # potential energy
H = T + V  # define hamiltonian

# H psi = E psi  <- omg an eigenvalue problem!
eigenvalues, eigenvectors = np.linalg.eigh(H)

