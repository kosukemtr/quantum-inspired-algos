
import numpy as np
from sample_matrix import MatrixSampler

def qi_solver(A, b, d, n_iter):
    """
    Solves for x in the least squares problem: argmin_x ||Ax - b|| based on arXiv: 2103.10309.
    
    This function seeks to find a vector 'y' such that 'x = A^T y' provides the solution. 
    
    Parameters:
    - A: Matrix involved in the least squares problem.
    - b: Vector representing the target values in the least squares problem.
    - d: The number of samples used for approximating inner products. 
         Increasing 'd' can improve the approximation accuracy.
         d = O(kappa_F^2/epsilon^2) where kappa_F = ||A||_F/||A^+|| and epsilon is error is recommended in the reference.
    - n_iter: The number of iterations to perform in the algorithm.
              More iterations can lead to a more accurate solution but take longer to compute.
              n = O(kappa_F^2 log(1/epsilon)) is recommended.
    """
    sampler = MatrixSampler(A)
    y0 = np.zeros(A.shape[0])
    y_history = [y0]

    for i in range(n_iter):
        row_index = sampler.sample_row_index()
        e = np.zeros_like(y0)
        e[row_index] = 1.
        if d != np.inf:
            col_indices = sampler.sample_col_index(d)
            y_new = y_history[-1] + \
                1/sampler.row_norms[row_index] * \ 
            (
                b[row_index] - \
                (
                    np.sum(
                        A[row_index, col_indices]*
                        y_history[-1].dot(A[:,col_indices])*
                        sampler.frobenius_norm / sampler.col_norms[col_indices]
                    )
                ) / d
            ) * e
        else:
            y_new = y_history[-1] + \
                1/sampler.row_norms[row_index] * \
            (
                b[row_index] - 
                        A[row_index, :].dot(np.transpose(A) @ y_history[-1])
            ) * e
        # print(row_index, y_new)
        y_history.append(y_new)

    return y_history

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Step 1: Create test matrix A and vector b
    # np.random.seed(0)  # for reproducibility
    rng = np.random.default_rng(seed=1)
    A = rng.random((3, 2))  # example dimensions
    print(A)
    b = rng.random(3)

    # Step 2: Find the true solution using pseudo-inverse
    x_true = np.linalg.pinv(A) @ b

    # Step 3: Run qi_solver to get approximate solutions
    d = 100000 # Example parameter, adjust as needed
    n_iter = 1000  # Example number of iterations
    y_history = qi_solver(A, b, d, n_iter)

    # Step 4: Calculate errors over iterations
    errors = []
    for y in y_history:
        x_approx = np.transpose(A) @ y
        error = np.linalg.norm(x_true - x_approx)
        errors.append(error)
    print(x_true)
    print(np.transpose(A) @ y_history[-1])
    # Step 5: Plot the error evolution
    plt.plot(range(len(y_history)), errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig("test.png")