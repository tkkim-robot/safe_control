from scipy.stats import norm, invgamma
from gmm_creation import create_gmm, plot_gmm
from distributionally_robust_cvar import DistributionallyRobustCVaR, plot_gmm_with_cvar

def main():
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

if __name__ == "__main__":
    main()