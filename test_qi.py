
from qi import _lcv, qi_lstsq_solver, sample_lstsq_solution
import numpy as np
import matplotlib.pyplot as plt
import datetime

def test_qi_lstsq_solver():
    # Step 1: Create test matrix A and vector b
    # np.random.seed(0)  # for reproducibility
    rng = np.random.default_rng(seed=1)
    A = rng.random((4, 10))  # example dimensions
    b = rng.random(4)
    plt.figure(figsize=(10,6))

    # Step 2: Find the true solution using pseudo-inverse
    x_true = np.linalg.pinv(A) @ b
    kappa_F = np.linalg.norm(A)/np.min(np.linalg.svd(A, compute_uv=False))
    print(f"kappa_F: {kappa_F}")
    # Step 3: Run qi_solver to get approximate solutions
    d = np.inf # Example parameter, adjust as needed
    n_iter = 1000  # Example number of iterations
    y_history = qi_lstsq_solver(A, b, d, n_iter)

    # Step 4: Calculate errors over iterations
    errors = []
    init_error = np.linalg.norm(x_true - np.transpose(A) @ y_history[0])**2
    for y in y_history:
        x_approx = np.transpose(A) @ y
        error = np.linalg.norm(x_true - x_approx)**2
        errors.append(error)
    print("true solution", x_true)
    print("qi single trial solution", np.transpose(A) @ y_history[-1])
    # Step 5: Plot the error evolution
    plt.plot(range(len(y_history)), errors, label="single trial")

    n_trials = 1
    y_history_list = np.zeros((n_trials, len(y_history), len(y_history[0])))
    for i in range(n_trials):
        y_history_list[i] = np.array(qi_lstsq_solver(A,b,d,n_iter))
    averaged_y_history = np.sum(y_history_list, axis=0)/n_trials
    averaged_error_history = np.zeros(len(y_history))
    for i in range(n_trials):
        for j, y in enumerate(y_history_list[i]):
            averaged_error_history[j] += np.linalg.norm(x_true - np.transpose(A) @ y)**2
    averaged_error_history /= n_trials
    print(averaged_error_history)
    # Step 4: Calculate errors over iterations
    errors = []
    for y in averaged_y_history:
        x_approx = np.transpose(A) @ y
        error = np.linalg.norm(x_true - x_approx)**2
        errors.append(error)
    print(f"qi {n_trials} trials solution", np.transpose(A) @ averaged_y_history[-1])
    # Step 5: Plot the error evolution
    plt.plot(range(len(y_history)), errors, label=f"average over {n_trials} trials")
    plt.plot(range(len(y_history)), averaged_error_history, label="$\|x-x_*\|^2$"+f" averaged over {n_trials} trials")
    plt.plot(range(len(y_history)), 100*(1-1/kappa_F**2)**np.arange(len(y_history))*init_error, label="theoretical bound for $\|x-x_*\|^2$ which 99% of x should satisfy")
    plt.savefig("test.png")
    # plt.clf()

    # Perform Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(A)

    # Truncate the singular values to create a low-rank matrix
    # Let's truncate to rank 2
    rank = 2
    S_truncated = np.zeros_like(A)
    S_truncated[:rank, :rank] = np.diag(S[:rank])

    # Reconstruct the low-rank matrix
    B = U @ S_truncated @ VT

    # Step 2: Find the true solution using pseudo-inverse
    x_true = np.linalg.pinv(B) @ b

    # Step 3: Run qi_solver to get approximate solutions
    y_history = qi_lstsq_solver(B, b, d, n_iter)

    # Step 4: Calculate errors over iterations
    errors = []
    for y in y_history:
        x_approx = np.transpose(B) @ y
        error = np.linalg.norm(x_true - x_approx)**2
        errors.append(error)
    print("true solution", x_true)
    print("qi single trial solution", np.transpose(B) @ y_history[-1])
    # Step 5: Plot the error evolution
    plt.plot(range(len(y_history)), errors, label="single trial low rank", linestyle="dashed")

    y_history_list = np.zeros((n_trials, len(y_history), len(y_history[0])))
    for i in range(n_trials):
        y_history_list[i] = np.array(qi_lstsq_solver(B,b,d,n_iter))
    averaged_y_history = np.sum(y_history_list, axis=0)/n_trials
    # Step 4: Calculate errors over iterations
    errors = []
    for y in averaged_y_history:
        x_approx = np.transpose(B) @ y
        error = np.linalg.norm(x_true - x_approx)**2
        errors.append(error)
    print(f"qi {n_trials} trials solution", np.transpose(B) @ averaged_y_history[-1])
    # Step 5: Plot the error evolution
    plt.plot(range(len(y_history)), errors, label=f"average over {n_trials} trials low rank", linestyle="dashed")
    plt.xlabel('Iteration')
    plt.ylabel('$\|x-x_*\|^2$')
    plt.yscale("log")
    plt.legend()
    plt.savefig("test_qi_lstsq_solver.png")

def test_lcv():
    def v_sampler(vector):
        """ Sampler for a vector, returns a random index based on |vector[i]|^2 """
        probabilities = np.abs(vector) ** 2
        probabilities /= np.sum(probabilities)
        return lambda: np.random.choice(len(vector), p=probabilities)

    # Test data
    num_vectors = 3
    vector_size = 5
    rng = np.random.default_rng(seed=1)
    coef_list = rng.random(num_vectors)
    v_list = rng.random((num_vectors, vector_size))
    v_norm_list = np.linalg.norm(v_list, axis=1)**2
    v_sampler_list = [v_sampler(v) for v in v_list]

    # Perform the sampling multiple times
    num_samples = 10000
    n_trials = 20
    all_sampled_distributions = np.zeros((n_trials, vector_size))

    for trial in range(n_trials):
        sampled_indices = [_lcv(coef_list, v_list, v_norm_list, v_sampler_list) for _ in range(num_samples)]
        sampled_distribution = np.bincount(sampled_indices, minlength=vector_size) / num_samples
        all_sampled_distributions[trial, :] = sampled_distribution

    # Calculate the true distribution
    true_distribution = np.zeros(vector_size)
    for i in range(num_vectors):
        true_distribution += coef_list[i] * v_list[i] 
    true_distribution = true_distribution**2 / np.sum(true_distribution**2)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(vector_size), true_distribution, color='red', label='True Distribution')
    parts = plt.violinplot(all_sampled_distributions, positions=np.arange(vector_size), showmeans=True)
    parts['bodies'][0].set_alpha(0.8)
    parts['bodies'][0].set_zorder(2)
    
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.legend()
    # Get current time and format it as a string
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Set the plot title with the current time
    plt.title(f"Plot generated on {formatted_time}")
    plt.savefig("test_lcv_violin.png")

def test_sample_lstsq_solution():
    # Step 1: Create test matrix A and vector b
    # np.random.seed(0)  # for reproducibility
    rng = np.random.default_rng(seed=1)
    A = rng.random((10, 5))  # example dimensions
    b = rng.random(10)

    # Perform Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(A)

    # Truncate the singular values to create a low-rank matrix
    # Let's truncate to rank 2
    rank = 2
    S_truncated = np.zeros_like(A)
    S_truncated[:rank, :rank] = np.diag(S[:rank])

    # Reconstruct the low-rank matrix
    B = U @ S_truncated @ VT

    # Step 2: Find the true solution using pseudo-inverse
    x_true = np.linalg.pinv(B) @ b

    # Step 3: Run qi_solver to get approximate solutions
    d = np.inf # Example parameter, adjust as needed
    n_iter = 30  # Example number of iterations
    n_samples = 100
    n_trials = 30
    vector_size = len(x_true)
    all_sampled_distributions = np.zeros((n_trials, vector_size))
    import tqdm
    for trial in tqdm.tqdm(range(n_trials)):
        sampled_indices = [sample_lstsq_solution(B, b, d, n_iter) for _ in range(n_samples)]
        sampled_distribution = np.bincount(sampled_indices, minlength=vector_size) / n_samples
        all_sampled_distributions[trial, :] = sampled_distribution

    # Calculate the true distribution
    true_distribution = x_true**2 / np.sum(x_true**2)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(vector_size), true_distribution, color='red', label='True Distribution')
    parts = plt.violinplot(all_sampled_distributions, positions=np.arange(vector_size), showmeans=True)
    parts['bodies'][0].set_alpha(0.8)
    parts['bodies'][0].set_zorder(2)
    
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig("test_sample_lstsq_solution.png")


if __name__=="__main__":
    # test_lcv()
    test_qi_lstsq_solver()
    # test_sample_lstsq_solution()