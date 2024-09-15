import numpy as np
from scipy.stats import invgamma, norm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def create_gmm(gaussians, inv_gammas, num_samples=3):
    """
    Create a Gaussian Mixture Model (GMM) from predefined Gaussian distributions and Inverse-Gamma distributions.
    """
    means = []
    variances = []
    
    for _ in range(num_samples):
        for gaussian, inv_gamma in zip(gaussians, inv_gammas):
            variance = inv_gamma.rvs()
            variances.append(variance)
            mean = gaussian.rvs()
            means.append(mean)
    
    gmm = GaussianMixture(n_components=num_samples)
    gmm.means_ = np.array(means).reshape(-1, 1)
    gmm.covariances_ = np.array(variances).reshape(-1, 1, 1)
    gmm.weights_ = np.ones(num_samples) / num_samples
    gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)) for cov in gmm.covariances_]) # For efficient computation
    
    return gmm

def plot_gmm(gmm):
    """
    Plot the Gaussian Mixture Model (GMM) with individual components overlayed.
    """
    x = np.linspace(gmm.means_.min() - 3, gmm.means_.max() + 3, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, '-k', label='GMM')

    for i in range(pdf_individual.shape[1]):
        plt.plot(x, pdf_individual[:, i], '--', label=f'GMM Component {i+1}')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model with Inidivual Components')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Define Gaussian distributions for means
    gaussians = [norm(loc=5, scale=1.5)]
    
    # Define Inverse-Gamma distributions for variances
    inv_gammas = [invgamma(a=2.5, scale=1.2)]
    
    # Create GMM
    gmm = create_gmm(gaussians, inv_gammas, num_samples=3)
    
    # Print the means and covariances of the created GMM
    print("GMM Means:", gmm.means_)
    print("GMM Covariances:", gmm.covariances_)

    # Plot the GMM
    plot_gmm(gmm)