# Import necessary libraries
from itc_classes import * # Assuming this is a custom library for ITC data analysis

import numpy as np
from random import choices
from scipy.optimize import root
from scipy import stats
import pocomc as pc # Likely for Bayesian inference
import corner # For plotting corner plots (marginal distributions)
import matplotlib.pyplot as plt
import yaml # For reading configuration files
import os # For interacting with the operating system (e.g., creating directories)
import torch # Likely for numerical computations, possibly GPU acceleration
import shutil # For copying files
import corner # Duplicate import, can be removed

R = 0.001987 # Gas constant in kcal/(mol*K)

def main():
    # Get the configuration file path from environment variables
    config_file='./config_bayesian_ITC_one_to_one.yaml'  #os.getenv("CONFIG_FILE") 

    # Load configuration from the YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Get run name and create path
    run_name=config['run_name']
    path = './'+ run_name +'/'

    # Check if continuing a previous run
    continue_run=config['continue_run']

    # If continuing, load the config file from the existing run directory
    if continue_run:
        with open(path+config_file, "r") as f:
            config = yaml.safe_load(f)

    # Extract various parameters from the configuration
    filenames = config['isotherms'] # List of ITC isotherm data files
    filtering = config['filtering'] # Filtering options
    prior_shape = config['prior_shape'] # Shape of the prior distribution
    width = config['width'] # Width of the prior
    jeffreys_sigma = config['jeffreys_sigma'] # Sigma for Jeffreys prior
    h0_auto = config['h0_auto'] # Automatic determination of h0
    prior_width_bounds = config['prior_width_bounds'] # Bounds for prior width
    order_iso = config['order_isotherms'] # Order of isotherms

    # Parameters for using a posterior from a previous run
    posterior = config['posterior']
    posterior_path = config['posterior_path']
    posterior_indices = config['posterior_indices']

    # Parameters for the Bayesian sampler
    n_effective = config['n_effective'] # Effective number of samples
    n_total=config['n_total'] # Total number of samples

    # Number of CPUs to use
    n_cpus = config['kernels']

    # Set environment variables for OMP and Torch to limit threads
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Create the run directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Copy the config file to the run directory if it's a new run
    if not continue_run:
        shutil.copy(config_file, path)

    # Extract binding state information from config
    binding_states=np.array(config['binding_states'])
    degeneracy=np.array(config['degeneracy_micro_macro'])
    components=np.array(config['components_binding_states'])
    names_components=config['names_components']
    dd_names=np.array(config['dd_names'])

    # Extract bounds and combinations for analysis
    dd_combinations=np.array(config['dd_combinations'])
    ddg_bounds=np.array(config['ddg_bounds'])
    ddh_bounds=np.array(config['ddh_bounds'])
    bounds=np.array(config['bounds'])
    # Initialize the binding structure object
    binding=binding_structure(components,names_components, degeneracy,binding_states)

    # Create isotherm objects for each data file
    isotherms=[]
    for file in filenames:
        isotherms.append(isotherm(binding,file))

    # Initialize the bayesian run object
    run=bayesian_run(binding,isotherms,path,prior_width_bounds)

    # Get the prior distribution based on configuration
    if posterior:
        # Load posterior data if specified
        posterior_data=np.loadtxt(posterior_path, delimiter=',')
        prior=run.get_prior(bounds=bounds, dd_combinations=dd_combinations, ddg_bounds=ddg_bounds, ddh_bounds=ddh_bounds, posterior=posterior, posterior_indices=posterior_indices, posterior_data=posterior_data, prior_shape=prior_shape, filtering=filtering, prior_width=width, jeffreys_sigma=jeffreys_sigma,  h0_auto=h0_auto, prior_width_bounds=prior_width_bounds,nuisance_bounds = np.array([[-10.0, 10.0],[0.001, 10.0]]),order_iso=order_iso)
    else:
        prior=run.get_prior(bounds=bounds, dd_combinations=dd_combinations, ddg_bounds=ddg_bounds, ddh_bounds=ddh_bounds, posterior=posterior, prior_shape=prior_shape, filtering=filtering, prior_width=width, jeffreys_sigma=jeffreys_sigma,  h0_auto=h0_auto, prior_width_bounds=prior_width_bounds,nuisance_bounds = np.array([[-10.0, 10.0],[0.001, 10.0]]),order_iso=order_iso)


    # Run the Bayesian sampler
    if continue_run:
        print('Continuing run ' + path)
        samples, logl, logz=run.run(prior, n_effective, n_total, n_cpus, continue_run)

    else:
        # Plot corner plot of the prior if it's a new run
        figures=run.plot_corner(prior.rvs(10000), prior.bounds_plot([item for sublist in width for item in (sublist if isinstance(sublist, list) else [sublist])]), name='corner_prior')
        print('Starting run ' + path)
        # Start the Bayesian sampling run
        samples, logl, logz=run.run(prior, n_effective, n_total, n_cpus)

    # Compute the average prior width from samples
    prior_width=np.average(samples[:,-(run.n_dim-run.cumulative_indices[-1]):],axis=0)
    # Plot various corner plots of the posterior samples
    figures=run.plot_corner(samples, prior.bounds_plot(prior_width), name='corner_full')
    figures=run.plot_corner([prior.rvs(10000),samples], prior.bounds_plot(prior_width), name='corner_full_prior')
    figures=run.plot_corner(samples, prior.bounds_plot(prior_width), name='corner_zoom', indices=list(range(2*len(binding_states))))
    figures=run.plot_corner(samples, prior.bounds_plot(prior_width), name='corner_error', indices=list(range(run.cumulative_indices[-1], run.n_dim)))

    # Plot the fitted isotherms
    figures=run.plot_isotherms(samples)

    # Plot histograms of the posterior samples
    run.plot_histograms(samples, prior.bounds_plot(prior_width), name='hist')

    # Plot corner plots for concentrations of each isotherm
    for i in range(len(isotherms)):
        figures=run.plot_corner(samples, prior.bounds_plot(prior_width), name='corner_conc_'+str(i+1), indices=list(range(run.cumulative_indices[1+i],run.cumulative_indices[2+i]-2)))

    # Compute and plot results for differential properties (ddG, ddH) if specified
    if len(dd_combinations)>=1:
        dd_samples,dd_labels=run.compute_dd_samples(samples, dd_combinations, labels=dd_names)

        # Ensure bounds for ddG and ddH match the number of combinations
        if len(ddg_bounds) == 1:
            ddg_bounds = np.array(list(ddg_bounds)*len(dd_combinations))

        if len(ddh_bounds) == 1:
            ddh_bounds = np.array(list(ddh_bounds)*len(dd_combinations))

        # Plot corner plots and histograms for differential properties
        run.plot_corner(dd_samples, np.concatenate([ddg_bounds,ddh_bounds]), name='corner_dd',labels=dd_labels)
        run.plot_histograms(dd_samples, np.concatenate([ddg_bounds,ddh_bounds]), name='hist', labels=dd_labels)




if __name__ == '__main__':
    main()