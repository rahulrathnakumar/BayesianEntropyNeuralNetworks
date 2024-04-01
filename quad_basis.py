import numpy as np

def quad_basis(x):
    """
    Quadratic basis functions
    """
    phi = np.zeros((3,x.shape[0]))
    phi[0] = 1.0
    phi[1] = x
    phi[2] = x**2
    return np.transpose(phi)

if __name__ == '__main__':
    x = np.array([0.7417, 1.7319, 0.3711, 0.8068])
    phi = quad_basis(x)
    print(phi)