import numpy as np

class MatrixSampler:
    def __init__(self, matrix, seed=None):
        self.matrix = np.array(matrix)
        self.elements_squared_matrix = np.array(matrix)**2
        self.m, self.n = self.matrix.shape
        self.row_norms = np.linalg.norm(self.matrix, axis=1)**2
        self.col_norms = np.linalg.norm(self.matrix, axis=0)**2
        self.frobenius_norm = np.sum(self.row_norms)
        self.rng = np.random.default_rng(seed)

    def sample_row_index(self, num_samples=1):
        return self.rng.choice(self.m, size=num_samples, p=self.row_norms / self.frobenius_norm)
    
    def sample_col_index(self, num_samples=1):
        return self.rng.choice(self.n, size=num_samples, p=self.col_norms / self.frobenius_norm)
    
    def sample_col_index_given_row(self, index, num_samples=1):
        return self.rng.choice(self.n, size=num_samples, 
                               p=self.elements_squared_matrix[index,:] / self.row_norms[index])

    def query_entry(self, i, j):
        return self.matrix[i, j]

    def query_norm(self, i):
        return self.row_norms[i]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Test code
    matrix = np.array([[0,0,0],[1,1,1],[2,2,2]]) 
    sampler = MatrixSampler(matrix)  # Initialize the MatrixSampler with the matrix

    num_samples = 10000  # Number of samples to draw
    samples = sampler.sample_row_index(num_samples)  # Draw samples

    # Count the frequency of each row being sampled
    sample_counts = np.bincount(samples, minlength=sampler.m)
    normalized_sample_counts = sample_counts / np.sum(sample_counts)
    normalized_row_norms = np.array([0,3,12])/np.sum([0,3,12])

    # Plotting both normalized sample counts and row norms
    plt.figure(figsize=(12, 7))
    plt.bar(range(sampler.m), normalized_sample_counts, width=0.4, label='Normalized Sample Counts', align='center')
    plt.bar(np.arange(sampler.m) + 0.4, normalized_row_norms, width=0.4, label='Normalized Row Norms', align='center')
    plt.xlabel('Row Index')
    plt.ylabel('Normalized Values')
    plt.title('Comparison of Normalized Sampling Distribution and Row Norms')
    plt.legend()
    plt.savefig("test.png")