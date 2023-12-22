
import numpy as np
from sample_matrix import MatrixSampler

def qi_lstsq_solver(A, b, d, n_iter):
    """
    Solves for x in the least squares problem: argmin_x ||Ax - b|| based on arXiv: 2103.10309.
    
    This function seeks to find a vector 'y' such that 'x = A^T y' provides the solution and returns history of y.
    
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
        y_history.append(y_new)

    return y_history

def sample_lstsq_solution(A, b, d, n_iter, seed=None):
    """
    Solves for x in the least squares problem: argmin_x ||Ax - b|| based on arXiv: 2103.10309.
    
    This function implements SQ(x), L2-norm sampling of the solution vector x,
    i.e., it returns a random index from the probability distribution proportional to |x[i]|^2
    
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
    y = qi_lstsq_solver(A, b, d, n_iter)[-1]
    def sampler(vector):
        """ Sampler for a vector, returns a random index based on |vector[i]|^2 """
        probabilities = np.abs(vector) ** 2
        probabilities /= np.sum(probabilities)
        return lambda: np.random.choice(len(vector), p=probabilities)
    sampler_list = [sampler(A[i]) for i in range(len(A))]
    return _lcv(y, A, np.linalg.norm(A, axis=1), sampler_list, seed)


def _lcv(coef_list, v_list, v_norm_list, v_sampler_list, seed=None):
    """
    Linear combination of vectors.
    Samples from u = \sum_i c_i v_i using rejection sampling, where c is coef and v is a vector.
    More spcifically, this function outputs random indices according to probability distribution |u[i]|^2
    Paramters:
        coef_list: coeffcient list
        v_list: list of vectors
        v_norm_list: list of norms of vectors
        v_samler_list: samplers of v. They need to be callables v_sampler() which samples a random index from a probability distribution |v[i]|^2.
    """
    if len(coef_list) != len(v_list):
        raise ValueError()
    while True:
        rng = np.random.default_rng(seed)
        coef_index = rng.choice(len(coef_list), p= coef_list ** 2 * v_norm_list / np.sum(coef_list ** 2 * v_norm_list))
        index = v_sampler_list[coef_index]()
        accept_prob = \
            np.sum(coef_list*v_list[:,index])**2 / len(coef_list) /\
                  np.linalg.norm(coef_list*v_list[:,index])**2
        if accept_prob > 1:
            raise ValueError()
        accept = rng.uniform() < accept_prob
        if accept:
            return index

