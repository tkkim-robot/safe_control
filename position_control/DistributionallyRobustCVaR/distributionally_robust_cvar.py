import numpy as np
from scipy.stats import norm, invgamma
from gmm_creation import create_gmm
import matplotlib.pyplot as plt


class DistributionallyRobustCVaR:
    def __init__(self, gmm):
        self.gmm = gmm

    def calculate_var(self, mu, sigma, alpha=0.95):
        """
        Calculate Value at Risk (VaR) for a normal distribution.
        """
        var = mu + sigma * norm.ppf(alpha)
        return var

    def calculate_cvar(self, mu, sigma, alpha=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) for a normal distribution.
        """
        cvar = mu + sigma * (norm.pdf(norm.ppf(alpha)) / (1 - alpha))
        return cvar

    def compute_dr_cvar(self, alpha=0.95):
        """
        Compute the infimum of CVaR values from the GMM components.
        """
        cvar_values = []
        for mean, cov in zip(self.gmm.means_, self.gmm.covariances_):
            mu = mean[0]
            sigma = np.sqrt(cov[0, 0])
            cvar = self.calculate_cvar(mu, sigma, alpha)
            cvar_values.append(cvar)
        dr_cvar = np.min(cvar_values)
        dr_cvar_index = np.argmin(cvar_values)
        return dr_cvar, cvar_values, dr_cvar_index

    def is_within_boundary(self, boundary, alpha=0.95):
        """
        Check if the Distributionally Robust CVaR is within the specified boundary.
        """
        dr_cvar, _, _ = self.compute_dr_cvar(alpha)
        return dr_cvar <= boundary


def plot_gmm_with_cvar(gmm, cvar_values, dr_cvar_index):
    """
    Plot the GMM with individual components, CVaR boundaries, and DR_CVaR line.
    """
    x = np.linspace(gmm.means_.min() - 3, gmm.means_.max() + 3, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, '-k', label='GMM')

    for i, cvar in enumerate(cvar_values):
        color = 'red'
        linestyle = '-'
        linewidth = 1 if i != dr_cvar_index else 2.5
        plt.axvline(cvar, color=color, linestyle=linestyle, linewidth=linewidth, label=f'Component {i+1} CVaR')
        if i == dr_cvar_index:
            plt.annotate('DR_CVaR', xy=(cvar, 0.0), xytext=(cvar - 0.15, 0.03),
                            arrowprops=dict(facecolor=color, shrink=0.05),
                            horizontalalignment='right')

    for i in range(pdf_individual.shape[1]):
        plt.plot(x, pdf_individual[:, i], '--', label=f'GMM Component {i+1}')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model with Individual Components and CVaR Boundaries')
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

    # Initialize the Distributionally Robust CVaR filter
    cvar_filter = DistributionallyRobustCVaR(gmm)

    # Define a boundary for the CVaR
    boundary = 10

    # Compute the Distributionally Robust CVaR
    dr_cvar, cvar_values, dr_cvar_index = cvar_filter.compute_dr_cvar(alpha=0.95)
    print(f"Distributionally Robust CVaR: {dr_cvar}")

    # Check if the Distributionally Robust CVaR is within the specified boundary
    within_boundary = cvar_filter.is_within_boundary(boundary, alpha=0.95)
    print(f"Within Boundary: {within_boundary}")

    # Plot the GMM with individual components and CVaR boundaries
    plot_gmm_with_cvar(gmm, cvar_values, dr_cvar_index)