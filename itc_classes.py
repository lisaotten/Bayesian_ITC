import numpy as np
from scipy.optimize import root
from scipy.spatial.distance import cdist
from scipy import stats
import pocomc as pc
import corner
import matplotlib.pyplot as plt
import yaml
import os
import torch
import shutil
import re
import emcee
import h5py
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import pytensor
pytensor.config.mode == 'NUMBA'
import multiprocessing as mp
mp.set_start_method('fork', force=True)

R = 0.001987 # Define the gas constant R

class LogPosterior:
    """
    Represents the logarithm of the posterior probability for a Bayesian model.

    Attributes:
        prior: An object representing the prior distribution with a `logpdf` method.
        log_like: A callable representing the log-likelihood function.
    """
    def __init__(self, prior, log_like):
        self.prior = prior
        self.log_like = log_like

    def __call__(self, x):
        """
        Calculates the log-posterior probability for a given parameter vector x.

        Args:
            x (ndarray): The parameter vector.

        Returns:
            float: The log-posterior probability, or -np.inf if parameters are outside prior bounds.
        """
        if np.any((x < self.prior.bounds[:, 0]) | (x > self.prior.bounds[:, 1])):
            return -np.inf # Return negative infinity if parameters are out of bounds
        return float(self.prior.logpdf(x) + self.log_like(x)) # Calculate and return log-posterior

class inverse_x_distribution:
    """
    Represents an inverse-x distribution with a given range [a, b].

    Attributes:
        a (float): The lower bound of the distribution.
        b (float): The upper bound of the distribution.
        log_norm (float): The normalization constant in log-space.
    """
    def __init__(self, a, b):
        if a <= 0 or b <= 0 or a >= b:
            raise ValueError("Require 0 < a < b.") # Validate input bounds
        self.a = a
        self.b = b
        self.log_norm = np.log(np.log(b / a))  # Normalization constant in log-space

    def logpdf(self, x):
        """
        Log-probability density function.

        Args:
            x (ndarray or scalar): The input value(s).

        Returns:
            ndarray or scalar: The log-probability density for the input value(s).
        """
        if np.isscalar(x):  # Handle scalar input
            if x < self.a or x > self.b:
                return -np.inf # Return negative infinity if out of bounds
            return -np.log(x) - self.log_norm
        else:  # Handle vectorized input
            x = np.asarray(x)
            log_pdf = -np.log(x) - self.log_norm
            log_pdf[(x < self.a) | (x > self.b)] = -np.inf # Set log-pdf to negative infinity for out-of-bounds values
            return log_pdf
    def rvs(self, size=1, random_state=None):
        """
        Generate random samples from the distribution.

        Args:
            size (int): The number of samples to generate.
            random_state (int or RandomState): A seed or RandomState object for reproducibility.

        Returns:
            ndarray: An array of random samples.
        """
        rng = np.random.default_rng(random_state)
        u = rng.uniform(0, 1, size) # Generate uniform random numbers
        return self.a * (self.b / self.a) ** u # Transform uniform samples to inverse-x samples


def geometric_medoid(points):
    """
    Computes the geometric medoid for a set of points.
    The medoid is the point in the input set that minimizes the sum of distances
    to all other points.

    Parameters:
        points (ndarray): Input points of shape (N, d) or (N,) for 1D data.

    Returns:
        medoid (ndarray or scalar): The medoid point (one of the input points).
                                    Returns a scalar if the input is 1D.
    """
    # Check if the input is 1D
    if points.ndim == 1:
        # Directly handle 1D array
        distance_matrix = cdist(points[:, None], points[:, None], metric='seuclidean')
    else:
        # Handle multi-dimensional array
        distance_matrix = cdist(points, points, metric='seuclidean')

    # Sum distances for each point
    total_distances = np.sum(distance_matrix, axis=1)

    # Find the index of the point with the minimum sum of distances
    medoid_index = np.argmin(total_distances)

    # Return the medoid
    return points[medoid_index]

        
class prior_isotherm:
    """
    Represents the prior distribution for isotherm parameters in a Bayesian model.

    Attributes:
        iso: An object containing isotherm data and properties.
        prior_shape (str): The shape of the prior distribution (e.g., 'gaussian', 'uniform', 'lognormal').
        filtering (bool): Whether to apply filtering based on concentration ratio.
        prior_width (float or list): The width of the prior distribution(s).
        nuisance_bounds (ndarray): Bounds for nuisance parameters (h0 and sigma).
        jeffreys_sigma (bool): Whether to use a Jeffreys prior for sigma.
        h0_auto (bool): Whether to automatically determine the h0 bound.
        dim (int): The dimensionality of the parameter space.
        bounds (ndarray): The overall bounds for all parameters.
        prior_nuisance_list (list): A list of distributions for nuisance parameters.
        filter_ind_c1 (int): Index for the first component in filtering.
        filter_ind_c2 (int): Index for the second component in filtering.
        factor (float): The ratio factor used in filtering.
    """

    def __init__(self, iso, prior_shape, filtering, prior_width=1, nuisance_bounds=np.array([[-50.0, 50.0],[0.001, 10.0]]), jeffreys_sigma=False, h0_auto=True):
        """
        Initializes the prior_isotherm object.

        Args:
            iso: Isotherm data object.
            prior_shape (str): Shape of the prior ('gaussian', 'uniform', 'lognormal').
            filtering (bool): Enable filtering.
            prior_width (float or list): Prior width.
            nuisance_bounds (ndarray): Bounds for h0 and sigma.
            jeffreys_sigma (bool): Use Jeffreys prior for sigma.
            h0_auto (bool): Automatically determine h0 bound.
        """

        self.iso=iso # Store the isotherm data object
        self.dim = int(self.iso.number_components + 2) # Calculate the dimensionality of the parameter space (components + 2 nuisance parameters)
        self.prior_shape = prior_shape # Store the prior shape
        self.prior_width = prior_width # Store the prior width

        # Determine the bounds for the parameters
        if h0_auto:
            # Automatically determine the h0 bound based on isotherm data
            self.bounds = np.array([[0 , np.inf]]*self.iso.number_components + [[self.iso.dq_list[self.iso.skipped_mask][0],2*self.iso.dq_list[self.iso.skipped_mask][-1]-self.iso.dq_list[self.iso.skipped_mask][0]]]+[list(nuisance_bounds[1])])
        else:
            # Use provided nuisance bounds for h0
            self.bounds = np.array([[0 , np.inf]]*self.iso.number_components + list(nuisance_bounds))

        # Initialize a list to store nuisance parameter priors
        self.prior_nuisance_list=[]
        # Add a uniform prior for h0 based on determined bounds
        self.prior_nuisance_list.append(stats.uniform(loc=self.bounds[self.iso.number_components][0], scale=self.bounds[self.iso.number_components][1]-self.bounds[self.iso.number_components][0]))

        # Add a prior for sigma
        if jeffreys_sigma:
            # Use an inverse-x distribution (Jeffreys prior) for sigma
            self.prior_nuisance_list.append(inverse_x_distribution(a=nuisance_bounds[1,0], b=nuisance_bounds[1,1]))
        else:
            # Use a uniform prior for sigma
            self.prior_nuisance_list.append(stats.uniform(loc=nuisance_bounds[1,0], scale=nuisance_bounds[1,1]-nuisance_bounds[1,0]))

        # Configure filtering if enabled
        if filtering:
            self.filtering = True
            # Find indices for filtering based on filter_prior and in_syringe properties
            self.filter_ind_c1= next(i for i, x in enumerate(self.iso.filter_prior) if x>0 and self.iso.in_syringe[i])
            self.filter_ind_c2= next(i for i, x in enumerate(self.iso.filter_prior) if x>0 and not self.iso.in_syringe[i])
            # Calculate the ratio factor for filtering
            self.factor= self.iso.filter_prior[self.filter_ind_c1]/self.iso.filter_prior[self.filter_ind_c2]
        else:
            self.filtering = False # Disable filtering

    def bounds_plot(self, prior_width=None):
        """
        Calculates bounds for plotting based on prior width.

        Args:
            prior_width (float or list, optional): Prior width for this calculation. Defaults to None.

        Returns:
            ndarray: Array of bounds for plotting.
        """
        if prior_width is None:
            prior_width=self.prior_width

        prior_width=list(np.atleast_1d(prior_width))

        if len(prior_width)==1:
            prior_width=prior_width*self.iso.number_components

        # Calculate and return bounds for plotting
        return np.array([[est*max(1-2*prior_width[i],0),est*(1+2*prior_width[i])] for i,est in enumerate(self.iso.concentration_estimate)] + list(self.bounds[-2:]))

    def return_simple_priors(self, prior_width):
        """
        Returns a list of simple prior distributions based on prior shape and width.

        Args:
            prior_width (ndarray): Array of prior widths for each component.

        Returns:
            list: A list of lists, where each inner list contains prior distributions for a component.
        """

        prior_width = np.atleast_2d(prior_width)
        if prior_width.shape[0] == 1:
            prior_width = np.repeat(prior_width, self.iso.number_components, axis=0)

        concentration = np.array(self.iso.concentration_estimate)[:, None] * np.ones_like(prior_width)

        # Create prior distributions based on the specified shape
        if self.prior_shape == 'gaussian':
            a = -1 / prior_width
            b = np.inf
            loc = concentration
            scale = concentration * prior_width
            priors = [
                    [stats.truncnorm(a[i, j], b, loc[i, j], scale[i, j]) for j in range(prior_width.shape[1])]
                    for i in range(prior_width.shape[0])
                    ]

        elif self.prior_shape == 'uniform':
            loc = concentration * np.maximum(1 - prior_width, 0)
            scale = concentration * (1 + prior_width - np.maximum(1 - prior_width, 0))
            priors = [
                [stats.uniform(loc[i, j], scale[i, j]) for j in range(prior_width.shape[1])]
                    for i in range(prior_width.shape[0])
                    ]

        elif self.prior_shape == 'lognormal':
            s = np.log(1 + prior_width)
            priors = [
                [stats.lognorm(s[i, j], scale=concentration[i, j]) for j in range(prior_width.shape[1])]
                        for i in range(prior_width.shape[0])
                        ]
        return priors



    def ratio_func(self,ratio):
        """
        Filters based on the concentration ratio.

        Args:
            ratio (float): The calculated concentration ratio.

        Returns:
            float: 0 if the ratio is within the acceptable range, -np.inf otherwise.
        """
        if self.factor>1:
            if ratio < self.factor-0.5:
                return -np.inf
            else:
                return 0
        else:
            if ratio > 1./(1./self.factor-0.5):
                return -np.inf
            else:
                return 0

    def ratio_func_mask(self,ratio):
        """
        Returns a boolean mask based on the concentration ratio.

        Args:
            ratio (float): The calculated concentration ratio.

        Returns:
            bool: True if the ratio is within the acceptable range, False otherwise.
        """
        if self.factor>1:
            if ratio < self.factor-0.5:
                return False
            else:
                return True
        else:
            if ratio > 1./(1./self.factor-0.5):
                return False
            else:
                return True

    def logpdf(self, x , prior_width=None):
        """
        Calculates the log-probability density function for the prior.

        Args:
            x (ndarray): The parameter vector(s).
            prior_width (ndarray, optional): Array of prior widths. Defaults to None.

        Returns:
            ndarray or scalar: The log-probability density.
        """

        shape_return=np.isscalar(x.T[0])
        x = np.atleast_2d(x)

        if prior_width is None:
            prior_list=self.return_simple_priors(self.prior_width)
            return_value=sum(dist[0].logpdf(x.T[i]) for i, dist in enumerate(prior_list))
            #prior_width = np.full(len(x.T[0]), self.prior_width)

        else:
            prior_width = np.atleast_2d(prior_width)
            if prior_width.shape[0] == 1:
                prior_width = np.repeat(prior_width, self.iso.number_components, axis=0)
            if prior_width.shape[1] < x.shape[0]:
                prior_width = prior_width * np.ones(x.shape[0])
            concentration = np.array(self.iso.concentration_estimate)[:, None] * np.ones_like(prior_width)

            # Calculate log-pdf based on the specified prior shape
            if self.prior_shape == 'gaussian':
                a = -1 / prior_width
                b = np.inf
                loc = concentration
                scale = concentration * prior_width
                return_value=np.sum( [
                        [stats.truncnorm.logpdf(x.T[0:self.iso.number_components][i,j],a[i, j], b, loc[i, j], scale[i, j]) for j in range(prior_width.shape[1])]
                        for i in range(prior_width.shape[0])
                        ],axis=0)

            elif self.prior_shape == 'uniform':
                loc = concentration * np.maximum(1 - prior_width, 0)
                scale = concentration * (1 + prior_width - np.maximum(1 - prior_width, 0))
                return_value=np.sum( [
                    [stats.uniform.logpdf(x.T[0:self.iso.number_components][i,j],loc[i, j], scale[i, j]) for j in range(prior_width.shape[1])]
                        for i in range(prior_width.shape[0])
                        ],axis=0)

            elif self.prior_shape == 'lognormal':
                s = np.log(1 + prior_width)
                return_value=np.sum( [
                    [stats.lognorm.logpdf(x.T[0:self.iso.number_components][i,j],s[i, j], scale=concentration[i, j]) for j in range(prior_width.shape[1])]
                        for i in range(prior_width.shape[0])
                        ],axis=0)

        # Add log-pdf of nuisance parameters
        return_value+=sum(dist.logpdf(x.T[i+self.iso.number_components]) for i, dist in enumerate(self.prior_nuisance_list))

        # Apply filtering if enabled
        if self.filtering:
            h0 = np.atleast_1d(x.T[-2])
            # Calculate inflection point and corresponding volume
            ind_inflection=np.argmin(np.abs(self.iso.dq_list[self.iso.skipped_mask][:,np.newaxis]-(h0+self.iso.dq_list[self.iso.skipped_mask][0])/2),axis=0)
            vol_inflection=np.array([sum(entry) for entry in [self.iso.inj_list[:i] for i in np.argmin(np.abs(self.iso.dq_list-self.iso.dq_list[self.iso.skipped_mask][ind_inflection][:, np.newaxis]), axis=1)]])
            # Calculate ratio and apply ratio function
            ratio=x.T[self.filter_ind_c1]*vol_inflection/x.T[self.filter_ind_c2]/self.iso.V0
            return_value+=np.array(list(map(self.ratio_func, ratio)))

        return np.squeeze(return_value) if shape_return else return_value

    def rvs(self, size=1, prior_width=None):
        """
        Generates random samples from the prior distribution.

        Args:
            size (int): The number of samples to generate.
            prior_width (ndarray, optional): Array of prior widths. Defaults to None.

        Returns:
            ndarray: An array of random samples.
        """

        if prior_width is None:
            prior_width = np.array([self.prior_width]*size).T

        prior_list=np.array(self.return_simple_priors(prior_width))

        # Generate samples from simple priors and nuisance priors
        return_vector = np.array([list(map(lambda d: d.rvs(size=1)[0], dist)) for dist in prior_list]+[dist.rvs(size=size) for dist in self.prior_nuisance_list])

        # Apply filtering and redraw samples if necessary
        if self.filtering:
            ind_inflection=np.argmin(np.abs(self.iso.dq_list[self.iso.skipped_mask][:,np.newaxis]-(return_vector[-2]+self.iso.dq_list[self.iso.skipped_mask][0])/2),axis=0)
            vol_inflection=np.array([sum(entry) for entry in [self.iso.inj_list[:i] for i in np.argmin(np.abs(self.iso.dq_list-self.iso.dq_list[self.iso.skipped_mask][ind_inflection][:, np.newaxis]), axis=1)]])
            ratio=return_vector[self.filter_ind_c1]*vol_inflection/return_vector[self.filter_ind_c2]/self.iso.V0
            check=np.array(list(map(self.ratio_func_mask, ratio)))
            counter=0
            while not np.all(check):
                counter+=1
                # Update only the entries in return_vector where check is False
                indices_to_update = np.where(~check)[0]  # Get indices where check is False

                return_vector[np.ix_([self.filter_ind_c1, self.filter_ind_c2], indices_to_update)] = np.array([list(map(lambda d: d.rvs(size=1)[0], dist[indices_to_update])) for dist in prior_list[[self.filter_ind_c1, self.filter_ind_c2]]])
                if counter%10000==0:
                    print('Redrawing',len(indices_to_update),'h0 sample(s)')
                    return_vector[np.ix_([-2], indices_to_update)]= np.array(self.prior_nuisance_list[0].rvs(size=len(indices_to_update)))
                    ind_inflection[indices_to_update]=np.argmin(np.abs(self.iso.dq_list[self.iso.skipped_mask][:,np.newaxis]-(return_vector[-2][indices_to_update]+self.iso.dq_list[self.iso.skipped_mask][0])/2),axis=0)
                    vol_inflection[indices_to_update]=np.array([sum(entry) for entry in [self.iso.inj_list[:i] for i in np.argmin(np.abs(self.iso.dq_list-self.iso.dq_list[self.iso.skipped_mask][ind_inflection[indices_to_update]][:, np.newaxis]), axis=1)]])

                # Recalculate ratio for updated entries only
                ratio = (return_vector[self.filter_ind_c1][indices_to_update] * vol_inflection[indices_to_update] /
                         (return_vector[self.filter_ind_c2][indices_to_update] * self.iso.V0))

                # Update the check array only for these entries
                check[indices_to_update] = np.array(list(map(self.ratio_func_mask, ratio)))

        return return_vector.T

class full_estimate:
    """
    Represents a full estimate for binding parameters, incorporating bounds and differential enthalpy/free energy constraints.

    Attributes:
        binding: An object containing binding data and properties.
        bounds (ndarray): Bounds for all parameters.
        bounds_plot (ndarray): Bounds for plotting.
        dimv (int): Dimensionality related to the number of stages.
        dim (int): Total dimensionality of the parameter space.
        dd_combinations (ndarray): Combinations for calculating differential values.
        ddg_bounds (ndarray): Bounds for differential free energy.
        ddh_bounds (ndarray): Bounds for differential enthalpy.
        last_non_zero_indices_dd (list): Indices of the last non-zero elements in dd_combinations.
        posterior_dist (gaussian_kde): Kernel density estimate of the posterior distribution.
    """
    def __init__(self, binding, bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior_data):

        self.binding=binding
        self.bounds = np.copy(bounds)
        self.bounds_plot = np.copy(bounds)
        self.dimv= binding.number_stages
        self.dim = len(self.bounds)
        self.dd_combinations = np.array(dd_combinations,dtype=int)

        # Ensure ddg_bounds has the correct shape
        if len(ddg_bounds) == 1:
            self.ddg_bounds = np.array(list(ddg_bounds)*len(self.dd_combinations))
        else:
            self.ddg_bounds = np.array(ddg_bounds)

        # Ensure ddh_bounds has the correct shape
        if len(ddh_bounds) == 1:
            self.ddh_bounds = np.array(list(ddh_bounds)*len(self.dd_combinations))
        else:
            self.ddh_bounds = np.array(ddh_bounds)

        # Find the index of the last non-zero element for each differential combination
        self.last_non_zero_indices_dd = [max([i for i, x in enumerate(sublist) if x != 0]) for sublist in self.dd_combinations]

        # Create a kernel density estimate of the posterior data
        self.posterior_dist=stats.gaussian_kde(posterior_data.T)

    def logpdf(self, x):
        """
        Calculates the log-probability density function.

        Args:
            x (ndarray): The parameter vector(s).

        Returns:
            ndarray: The log-probability density.
        """
        # Calculate log-pdf based on uniform priors within bounds
        return_vector = np.sum([np.where((x.T[i] >= self.bounds[i, 0]) & (x.T[i] <= self.bounds[i, 1]), -np.log(self.bounds[i, 1] - self.bounds[i, 0]) if np.isfinite(self.bounds[i, 1] - self.bounds[i, 0]) else 0.0, -np.inf) for i in range(self.dim)], axis=0)
        # Add log-pdf from the posterior distribution KDE
        temp=self.posterior_dist.logpdf(x.T)
        if len(temp)==1:
            temp=temp[0]
        return_vector+=temp

        # Add log-pdf contributions from differential free energy and enthalpy bounds
        for i, combination in enumerate(self.dd_combinations):
            sum_total=sum(combination[k]*x.T[k] for k in range(len(combination)))
            return_vector+=stats.uniform.logpdf(sum_total,loc=self.ddg_bounds[i,0],scale=self.ddg_bounds[i,1]-self.ddg_bounds[i,0])
            sum_total=sum(combination[k]*x.T[k+self.dimv] for k in range(len(combination)))
            return_vector+=stats.uniform.logpdf(sum_total,loc=self.ddh_bounds[i,0],scale=self.ddh_bounds[i,1]-self.ddh_bounds[i,0])

        return return_vector


    def rvs(self, size=1):
        """
        Generates random samples from the distribution.

        Args:
            size (int): The number of samples to generate.

        Returns:
            ndarray: An array of random samples.
        """

        invalid_indices=np.arange(size, dtype=int)
        return_vector = np.empty((size,self.dim))

        # Generate samples until all satisfy the bounds and differential constraints
        while len(invalid_indices)>0:
            within_bounds = np.ones(size, dtype=int)
            # Resample from the posterior distribution KDE
            return_vector[invalid_indices] = self.posterior_dist.resample(len(invalid_indices)).T
            # Check if samples are within the defined bounds
            for i, bound in enumerate(self.bounds[:self.dimv]):

                indx=[k for k, y in enumerate(self.last_non_zero_indices_dd) if y == i]

                lower_bound=[bound[0]]
                upper_bound=[bound[1]]
                for ind in indx:

                    sum_total = sum(self.dd_combinations[ind][k]*return_vector.T[k] for k in range(i))

                    if self.dd_combinations[ind][i] > 0:
                        lower_bound=np.maximum(lower_bound,(self.ddg_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])
                        upper_bound=np.minimum(upper_bound,(self.ddg_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                    else:
                        lower_bound=np.maximum(lower_bound,(self.ddg_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                        upper_bound=np.minimum(upper_bound,(self.ddg_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])

                within_bounds &= ((return_vector.T[i] >= lower_bound) & (return_vector.T[i] <= upper_bound)).astype(int)

            for i, bound in enumerate(self.bounds[self.dimv:2*self.dimv]):

                indx=[k for k, y in enumerate(self.last_non_zero_indices_dd) if y == i]
                lower_bound=[bound[0]]
                upper_bound=[bound[1]]
                for ind in indx:
                    sum_total = sum(self.dd_combinations[ind][k]*return_vector.T[k+self.dimv] for k in range(i))
                    if self.dd_combinations[ind][i] > 0:
                        lower_bound=np.maximum(lower_bound,(self.ddh_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])
                        upper_bound=np.minimum(upper_bound,(self.ddh_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                    else:
                        lower_bound=np.maximum(lower_bound,(self.ddh_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                        upper_bound=np.minimum(upper_bound,(self.ddh_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])

                within_bounds &= ((return_vector.T[i+self.dimv] >= lower_bound) & (return_vector.T[i+self.dimv] <= upper_bound)).astype(int)

            for i, bound in enumerate(self.bounds[2*self.dimv:]):
                within_bounds &= ((return_vector.T[i+2*self.dimv] >= bound[0]) & (return_vector.T[i+2*self.dimv] <= bound[1])).astype(int)

            invalid_indices = np.where(within_bounds == 0)[0]

        return return_vector

class prior_combined:
    """
    Represents a combined prior distribution for binding structure and isotherm parameters.

    Attributes:
        dim_prior_prior_width (int): Dimensionality related to prior width bounds.
        bounds (ndarray): Overall bounds for all parameters.
        dim (int): Total dimensionality of the parameter space.
        cumulative_indices (list): Cumulative indices for slicing parameter vectors.
        prior_prior_width (list): List of prior distributions for prior widths.
        order_iso (list): Order of isotherms.
        structure (list): Structure of nested prior width bounds.
        prior_binding_structure: Prior distribution for binding structure.
        prior_isotherms (list): List of prior distributions for isotherms.
        name_to_order (dict): Mapping of component names to their order in isotherms.
        fixed_samples (bool): Whether to use fixed samples.
        fixed_samples_set (ndarray): Set of fixed samples if fixed_samples is True.
    """

    def __init__(self, prior_binding_structure, prior_isotherms, cumulative_indices, prior_width_bounds=[],order_iso=[],fixed_samples=None):

        # Flatten the prior width bounds and determine its dimensionality
        flattened_prior_width_bounds = [item for sublist in prior_width_bounds for item in (sublist if isinstance(sublist[0], list) else [sublist])]
        self.dim_prior_prior_width=len(flattened_prior_width_bounds)

        # Combine bounds from binding structure, isotherms, and prior width bounds
        self.bounds = np.array(list(prior_binding_structure.bounds) + [bound for prior in prior_isotherms for bound in prior.bounds] + [bound for bound in flattened_prior_width_bounds])
        # Calculate total dimensionality
        self.dim = prior_binding_structure.dim + sum(prior.dim for prior in prior_isotherms) + self.dim_prior_prior_width
        self.cumulative_indices = cumulative_indices

        # Create uniform prior distributions for prior widths
        self.prior_prior_width=[stats.uniform(loc=bound[0], scale=bound[1]-bound[0]) for bound in flattened_prior_width_bounds]

        self.order_iso=order_iso

        # Determine the structure of nested prior width bounds
        self.structure = []
        for item in prior_width_bounds:
            if isinstance(item, list):
                # Check if item is a list and if its elements are also lists
                if isinstance(item[0], list):  # Nested list
                    # Count nested list lengths
                    self.structure.append(len(item))
                else:
                    # Single-level list, just count 1
                    self.structure.append(1)

        self.prior_binding_structure = prior_binding_structure
        self.prior_isotherms = prior_isotherms

        # Create a mapping from component names to their order in isotherms
        self.name_to_order = dict(zip(self.prior_binding_structure.binding.names_components, self.order_iso))

        # Check if fixed samples are provided
        if fixed_samples is None:
            self.fixed_samples=False
        else:
            self.fixed_samples=True
            self.fixed_samples_set=fixed_samples

    def recreate_nested_list(self,flattened_list):
        """
        Recreates the nested list structure from a flattened list.

        Args:
            flattened_list (list): The flattened list.

        Returns:
            list: The recreated nested list.
        """
        result = []
        start_idx = 0

        for group_size in self.structure:
            # Create groups of the appropriate size
            group = []
            for _ in range(group_size):
                group.append(flattened_list[start_idx])
                start_idx += 1
            # For the group that contains 2 elements, group them together as a sublist

            result.append(group)

        return result

    def bounds_plot(self,prior_width=None):
        """
        Calculates bounds for plotting based on prior width.

        Args:
            prior_width (float or list, optional): Prior width for this calculation. Defaults to None.

        Returns:
            ndarray: Array of bounds for plotting.
        """

        return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for prior in self.prior_isotherms for bound in prior.bounds_plot(prior_width)]

        if self.dim_prior_prior_width>0:
            if self.dim_prior_prior_width>1:
                name_to_width = dict(zip(self.prior_binding_structure.binding.names_components, self.recreate_nested_list(prior_width)))

                return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for i,prior in enumerate(self.prior_isotherms) for bound in prior.bounds_plot(np.array([name_to_width[name][self.name_to_order[name][i]] for name in prior.iso.names_components]))]
            else:
                return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for prior in self.prior_isotherms for bound in prior.bounds_plot(prior_width)]
            return_bounds+=list(self.bounds[-self.dim_prior_prior_width:])
        else:
            return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for prior in self.prior_isotherms for bound in prior.bounds_plot(prior_width)]

        return np.array(return_bounds)

    def logpdf(self, x):
        """
        Calculates the log-probability density function for the combined prior.

        Args:
            x (ndarray): The parameter vector(s).

        Returns:
            ndarray: The log-probability density.
        """

        # Calculate log-pdf for binding structure parameters
        return_vector=self.prior_binding_structure.logpdf(x.T[:self.cumulative_indices[1]].T)

        # Add log-pdf contributions from isotherm parameters and prior widths
        if self.dim_prior_prior_width>0:
            if self.dim_prior_prior_width>1:

                name_to_width = dict(zip(self.prior_binding_structure.binding.names_components, self.recreate_nested_list(x.T[self.cumulative_indices[-1]:])))

                for i, prior in enumerate(self.prior_isotherms):
                    return_vector+=prior.logpdf(x.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T,np.array([name_to_width[name][self.name_to_order[name][i]] for name in prior.iso.names_components])[:,None] if x.ndim==1 else np.array([name_to_width[name][self.name_to_order[name][i]] for name in prior.iso.names_components]))

            else:
                return_vector+=sum(prior.logpdf(x.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T,x.T[self.cumulative_indices[-1]]) for i, prior in enumerate(self.prior_isotherms))

            return_vector+=sum(prior.logpdf(x.T[self.cumulative_indices[-1]+i].T) for i, prior in enumerate(self.prior_prior_width))

        else:
            return_vector+=sum(prior.logpdf(x.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T) for i, prior in enumerate(self.prior_isotherms))

        return return_vector

    def rvs(self, size=1):
        """
        Generates random samples from the combined prior distribution.

        Args:
            size (int): The number of samples to generate.

        Returns:
            ndarray: An array of random samples.
        """

        # If fixed samples are provided, return a random selection from them
        if self.fixed_samples:
            ind_random=np.random.choice(range(len(self.fixed_samples_set)),size,replace=False)

            return self.fixed_samples_set[ind_random]

        # Otherwise, generate random samples from the individual prior components
        else:

            return_vector_prior_width=[prior.rvs(size) for prior in self.prior_prior_width]

            return_vector=list(self.prior_binding_structure.rvs(size).T)

            if self.dim_prior_prior_width>0:
                if self.dim_prior_prior_width>1:
                    name_to_width = dict(zip(self.prior_binding_structure.binding.names_components, self.recreate_nested_list(return_vector_prior_width)))

                    for i,prior in enumerate(self.prior_isotherms):
                        return_vector+= list(prior.rvs(size, np.array([name_to_width[name][self.name_to_order[name][i]] for name in prior.iso.names_components])).T)
                else:
                    for prior in self.prior_isotherms:
                        return_vector+= list(prior.rvs(size, return_vector_prior_width[0]).T)
            else:
                for prior in self.prior_isotherms:
                        return_vector+= list(prior.rvs(size).T)

            return np.array(return_vector+return_vector_prior_width).T



class prior_thermo:
    """
    Represents a prior distribution for thermodynamic parameters, potentially incorporating posterior information.

    Attributes:
        binding: An object containing binding data and properties.
        bounds (ndarray): Bounds for all parameters.
        bounds_plot (ndarray): Bounds for plotting.
        dim (int): Total dimensionality of the parameter space.
        dimv (int): Dimensionality related to the number of stages (half of total).
        dd_combinations (ndarray): Combinations for calculating differential values.
        ddg_bounds (ndarray): Bounds for differential free energy.
        ddh_bounds (ndarray): Bounds for differential enthalpy.
        last_non_zero_indices_dd (list): Indices of the last non-zero elements in dd_combinations.
        posterior (bool): Whether posterior information is used.
        posterior_dist (gaussian_kde): Kernel density estimate of the posterior distribution if posterior is True.
        posterior_indices (list): Indices of parameters informed by the posterior.
        posterior_data (ndarray): Posterior data if posterior is True.
    """

    def __init__(self, binding, bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior, posterior_indices=None, posterior_data=None):

        self.binding=binding
        self.bounds = np.copy(bounds)
        self.bounds_plot = np.copy(bounds)
        self.dim = len(self.bounds)
        self.dimv= int(len(self.bounds)/2)
        self.dd_combinations = np.array(dd_combinations,dtype=int)

        # Ensure ddg_bounds has the correct shape
        if len(ddg_bounds) == 1:
            self.ddg_bounds = np.array(list(ddg_bounds)*len(self.dd_combinations))
        else:
            self.ddg_bounds = np.array(ddg_bounds)

        # Ensure ddh_bounds has the correct shape
        if len(ddh_bounds) == 1:
            self.ddh_bounds = np.array(list(ddh_bounds)*len(self.dd_combinations))
        else:
            self.ddh_bounds = np.array(ddh_bounds)

        # Find the index of the last non-zero element for each differential combination
        self.last_non_zero_indices_dd = [max([i for i, x in enumerate(sublist) if x != 0]) for sublist in self.dd_combinations]

        # Configure posterior information if provided
        if posterior:
            self.posterior=True
            self.posterior_dist=stats.gaussian_kde(posterior_data.T)#bw_method=len(posterior_data)**(-1./(len(posterior_data.T)+4)/2))
            self.posterior_indices=posterior_indices

        else:
            self.posterior=False
            self.posterior_indices=[]

    def logpdf(self, x):
        """
        Calculates the log-probability density function.

        Args:
            x (ndarray): The parameter vector(s).

        Returns:
            ndarray: The log-probability density.
        """

        # Calculate log-pdf based on uniform priors for parameters not informed by the posterior
        return_vector=sum(stats.uniform.logpdf(x.T[i],loc=self.bounds[i,0],scale=self.bounds[i,1]-self.bounds[i,0]) for i in range(self.dim) if i not in self.posterior_indices)

        # Add log-pdf from the posterior distribution KDE if used
        if self.posterior:
            temp=self.posterior_dist.logpdf(x.T[self.posterior_indices])
            if len(temp)==1:
                temp=temp[0]
            return_vector+=temp

        # Add log-pdf contributions from differential free energy and enthalpy bounds
        for i, combination in enumerate(self.dd_combinations):
            sum_total=sum(combination[k]*x.T[k] for k in range(len(combination)))
            return_vector+=stats.uniform.logpdf(sum_total,loc=self.ddg_bounds[i,0],scale=self.ddg_bounds[i,1]-self.ddg_bounds[i,0])
            sum_total=sum(combination[k]*x.T[k+self.dimv] for k in range(len(combination)))
            return_vector+=stats.uniform.logpdf(sum_total,loc=self.ddh_bounds[i,0],scale=self.ddh_bounds[i,1]-self.ddh_bounds[i,0])

        return return_vector

    def rvs(self, size=1):
        """
        Generates random samples from the prior distribution.

        Args:
            size (int): The number of samples to generate.

        Returns:
            ndarray: An array of random samples.
        """

        return_vector=[]

        # If posterior is used, sample from the posterior KDE
        if self.posterior:
            ind_random=np.random.choice(range(len(self.posterior_dist.dataset.T)),size)

        # Generate samples for free energy parameters, considering bounds and differential constraints
        for i, bound in enumerate(self.bounds[:self.dimv]):

            if i in self.posterior_indices:

                return_vector.append(self.posterior_dist.dataset.T[ind_random,self.posterior_indices.index(i)])

                continue

            indx=[k for k, y in enumerate(self.last_non_zero_indices_dd) if y == i]

            lower_bound=[bound[0]]
            upper_bound=[bound[1]]
            for ind in indx:

                sum_total = sum(self.dd_combinations[ind][k]*return_vector[k] for k in range(len(return_vector)))

                if self.dd_combinations[ind][i] > 0:
                    lower_bound=np.maximum(lower_bound,(self.ddg_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])
                    upper_bound=np.minimum(upper_bound,(self.ddg_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                else:
                    lower_bound=np.maximum(lower_bound,(self.ddg_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                    upper_bound=np.minimum(upper_bound,(self.ddg_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])

            return_vector.append(np.random.uniform(lower_bound,upper_bound,size=size))

        # Generate samples for enthalpy parameters, considering bounds and differential constraints
        for i, bound in enumerate(self.bounds[self.dimv:]):

            if i + self.dimv in self.posterior_indices:

                return_vector.append(self.posterior_dist.dataset.T[ind_random,self.posterior_indices.index(i + self.dimv)])
                continue

            indx=[k for k, y in enumerate(self.last_non_zero_indices_dd) if y == i]
            lower_bound=[bound[0]]
            upper_bound=[bound[1]]
            for ind in indx:
                sum_total = sum(self.dd_combinations[ind][k]*return_vector[k+self.dimv] for k in range(len(return_vector)-self.dimv))
                if self.dd_combinations[ind][i] > 0:
                    lower_bound=np.maximum(lower_bound,(self.ddh_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])
                    upper_bound=np.minimum(upper_bound,(self.ddh_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                else:
                    lower_bound=np.maximum(lower_bound,(self.ddh_bounds[ind][1]-sum_total)/self.dd_combinations[ind][i])
                    upper_bound=np.minimum(upper_bound,(self.ddh_bounds[ind][0]-sum_total)/self.dd_combinations[ind][i])

            return_vector.append(np.random.uniform(lower_bound,upper_bound,size=size))

        return np.array(return_vector).T
        
class bayesian_run:
    """
    Manages Bayesian inference runs for binding and isotherm data using different samplers.

    Attributes:
        binding: Binding data object.
        list_isotherms (list): List of isotherm data objects.
        number_isotherms (int): Number of isotherms.
        cumulative_indices (ndarray): Cumulative indices for slicing parameter vectors.
        n_dim (int): Total dimensionality of the parameter space.
        labels (ndarray): Labels for the parameters.
        path (str): Output path for results.
    """

    def __init__(self, binding, list_isotherms, path, prior_width_bounds=[]):
        """
        Initializes the bayesian_run object.

        Args:
            binding: Binding data object.
            list_isotherms (list): List of isotherm data objects.
            path (str): Output path for results.
            prior_width_bounds (list, optional): Bounds for prior widths. Defaults to [].
        """

        self.binding=binding # Store binding data
        self.list_isotherms=list_isotherms # Store list of isotherms
        self.number_isotherms=len(list_isotherms) # Store number of isotherms
        # Calculate cumulative indices for parameter slicing
        self.cumulative_indices = np.cumsum([self.binding.number_stages]*2+[iso.number_components+2 for iso in self.list_isotherms])

        # Flatten prior width bounds and calculate its dimensionality
        flattened_prior_width_bounds = [item for sublist in prior_width_bounds for item in (sublist if isinstance(sublist[0], list) else [sublist])]
        prior_prior_width=len(flattened_prior_width_bounds)
        # Calculate total dimensionality of the parameter space
        self.n_dim = self.cumulative_indices[-1]+prior_prior_width

        # Generate labels for the parameters
        self.labels = np.concatenate((np.char.add('g',self.binding.binding_states),np.char.add('h',self.binding.binding_states)))
        for i, isotherm in enumerate(self.list_isotherms):
            self.labels=np.concatenate((self.labels,np.char.add('iso'+str(i+1)+'_',np.char.add(isotherm.names_components,np.where(isotherm.in_syringe, '_s', '_c'))),['iso'+str(i+1)+'_dh0'],['iso'+str(i+1)+'_sigma']))
        if prior_prior_width==1:
            self.labels=np.concatenate((self.labels,['prior_width']))
        elif prior_prior_width>1:

            for i, sublist in enumerate(prior_width_bounds):
                if isinstance(sublist[0], list):
                    for j, subsublist in enumerate(sublist):
                        self.labels = np.concatenate(
                            (self.labels, [f"prior_width_{self.binding.names_components[i]}_{j+1}"])
                        )
                else:
                    self.labels = np.concatenate(
                        (self.labels, [f"prior_width_{self.binding.names_components[i]}"])
                    )

        # Create output directories if they don't exist
        if not os.path.exists(path):
            os.makedirs(path)
        self.path=path

        if not os.path.exists(self.path + 'figures/'):
            os.makedirs(self.path + 'figures/')

        if not os.path.exists(self.path + 'data/'):
            os.makedirs(self.path + 'data/')

    def log_like(self, parameters):
        """
        Calculates the log-likelihood of the data given the parameters.

        Args:
            parameters (ndarray): The parameter vector.

        Returns:
            float: The log-likelihood.
        """

        logl=0
        for i, iso in enumerate(self.list_isotherms):
            # Calculate predicted dq values for the current isotherm
            dq_list_calc=iso.get_dq_list(dg = parameters[:self.cumulative_indices[0]],
                                         dh = parameters[self.cumulative_indices[0]:self.cumulative_indices[1]],
                                         total_concentrations = parameters[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][:iso.number_components],
                                         dh_0 = parameters[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components])

            # Extract sigma for the current isotherm
            sigma = parameters[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components+1]

            # Calculate the difference between calculated and experimental dq values
            diff = dq_list_calc[self.list_isotherms[i].skipped_mask] - self.list_isotherms[i].dq_list[self.list_isotherms[i].skipped_mask]
            # Update log-likelihood based on the difference and sigma
            logl += -len(diff) * np.log(np.sqrt(2*np.pi)*sigma)
            logl += -0.5 * np.dot(diff, diff)/ sigma**2

        return logl

    def get_prior(self, bounds, dd_combinations, ddg_bounds, ddh_bounds,  posterior, prior_shape, filtering, prior_width=[1], nuisance_bounds = np.array([[-50.0, 50.0],[0.001, 10.0]]), posterior_indices=None, posterior_data=None, jeffreys_sigma=False,  h0_auto=True, prior_width_bounds=[],order_iso=[],fixed_samples=None):
        """
        Creates a combined prior distribution for the Bayesian run.

        Args:
            bounds (ndarray): Bounds for all parameters.
            dd_combinations (ndarray): Combinations for differential values.
            ddg_bounds (ndarray): Bounds for differential free energy.
            ddh_bounds (ndarray): Bounds for differential enthalpy.
            posterior (bool): Whether to use posterior information.
            prior_shape (str): Shape of the prior distribution.
            filtering (bool): Enable filtering.
            prior_width (list, optional): Prior width(s). Defaults to [1].
            nuisance_bounds (ndarray, optional): Bounds for nuisance parameters. Defaults to np.array([[-50.0, 50.0],[0.001, 10.0]]).
            posterior_indices (list, optional): Indices informed by posterior. Defaults to None.
            posterior_data (ndarray, optional): Posterior data. Defaults to None.
            jeffreys_sigma (bool, optional): Use Jeffreys prior for sigma. Defaults to False.
            h0_auto (bool, optional): Automatically determine h0 bound. Defaults to True.
            prior_width_bounds (list, optional): Bounds for prior widths. Defaults to [].
            order_iso (list, optional): Order of isotherms. Defaults to [].
            fixed_samples (ndarray, optional): Fixed samples. Defaults to None.

        Returns:
            prior_combined: A combined prior distribution object.
        """

        order_iso_prior=order_iso
        # Get prior for binding structure
        prior_binding=self.binding.get_prior(bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior, posterior_indices, posterior_data)

        # Create prior distributions for isotherms, considering prior width structure
        if len(prior_width)>1:

            if not order_iso:
                order_iso_prior=[[0] * len(self.list_isotherms) for _ in range(len(self.binding.names_components))]

            structure = []
            for item in prior_width:
                if isinstance(item, list):
                    structure.append(len(item))
                else:
                    structure.append(1)

            prior_width_flat=[item for sublist in prior_width for item in (sublist if isinstance(sublist, list) else [sublist])]

            prior_width_structure = []
            start_idx = 0

            for group_size in structure:
                # Create groups of the appropriate size
                group = []
                for _ in range(group_size):
                    group.append(prior_width_flat[start_idx])
                    start_idx += 1
                # For the group that contains 2 elements, group them together as a sublist

                prior_width_structure.append(group)

            name_to_width = dict(zip(self.binding.names_components, prior_width_structure))
            name_to_order = dict(zip(self.binding.names_components, order_iso_prior))

            prior_isotherms=[]
            for i,iso in enumerate(self.list_isotherms):
                prior_width_i=[name_to_width[name][name_to_order[name][i]] for name in iso.names_components]
                prior_isotherms.append(iso.get_prior(prior_shape=prior_shape, filtering=filtering,  prior_width=prior_width_i, nuisance_bounds = nuisance_bounds, jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto))

        else:
            prior_isotherms=[iso.get_prior(prior_shape=prior_shape, filtering=filtering,  prior_width=prior_width, nuisance_bounds = nuisance_bounds, jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto) for iso in self.list_isotherms]

        # Return a combined prior object
        return prior_combined(prior_binding, prior_isotherms, self.cumulative_indices, prior_width_bounds, order_iso_prior,fixed_samples)

    def get_full_distribution(self, bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior_data):
        """
        Creates a full estimate distribution object.

        Args:
            bounds (ndarray): Bounds for all parameters.
            dd_combinations (ndarray): Combinations for differential values.
            ddg_bounds (ndarray): Bounds for differential free energy.
            ddh_bounds (ndarray): Bounds for differential enthalpy.
            posterior_data (ndarray): Posterior data.

        Returns:
            full_estimate: A full estimate distribution object.
        """

        return full_estimate(self.binding, bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior_data)

    def run_pymc(self, prior, n_effective, n_total, n_step=100, continue_run=False):
        """
        Runs Bayesian inference using PyMC.

        Args:
            prior: The prior distribution.
            n_effective (int): Effective number of samples.
            n_total (int): Total number of steps.
            n_step (int, optional): Number of steps per save. Defaults to 100.
            continue_run (bool, optional): Whether to continue a previous run. Defaults to False.

        Returns:
            ndarray: Sampled parameters.
        """
        checkpoint_dir = os.path.join(self.path, "data")

        samples_file = os.path.join(checkpoint_dir, "samples_pymc")

        log_posterior = LogPosterior(prior, self.log_like)

        class Posterior(Op):
            itypes = [pt.dvector]
            otypes = [pt.dscalar]
            def perform(self, node, inputs, outputs):
                x = inputs[0]
                #print(f"Input Type: {type(x)}, Input Shape: {x.shape if isinstance(x, np.ndarray) else 'scalar'}")

                result = log_posterior(x)
                #print(f"Result Type: {type(result)}, Result: {result}")
                outputs[0][0] = np.asarray(result, dtype=np.float64)

        posterior_op = Posterior()

        samples = prior.rvs(n_effective)

        if continue_run:
            sample_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("samples_pymc_step_")]
            if sample_files:
                latest_file = max(sample_files, key=lambda x: int(x.split("_step_")[1].split(".csv")[0]))
                latest_path = os.path.join(checkpoint_dir, latest_file)
                samples = np.loadtxt(latest_path, delimiter=",")[-n_effective:]
                last_step = int(latest_file.split('_step_')[1].split('.csv')[0])
                start_step = last_step
            else:
                samples = prior.rvs(n_effective)
                start_step = 0
        else:
            samples = prior.rvs(n_effective)
            start_step = 0

        start_points = [{"x": np.asarray(s, dtype=np.float64)} for s in samples]
        prior_bounds=np.where(np.isinf(prior.bounds), 1, prior.bounds)
        #covariance_matrix = np.cov(samples, rowvar=False)
        #covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6
        # Calculate standard deviations for each parameter
        std_devs = np.std(samples, axis=0)
        print(std_devs)
        # Construct diagonal covariance matrix
        covariance_matrix = np.diag(std_devs ** 2)
        for i in range(start_step, n_total, n_step):
            draws = min(n_step, n_total - i)
            print(f"Sampling from {i + 1} to {i + draws}")

            with pm.Model() as model:
                x = pm.Uniform("x", lower=prior_bounds[:, 0], upper=prior_bounds[:, 1], shape=(self.n_dim,))
                pm.Potential("logp", posterior_op(x))
                step = pm.DEMetropolis(S=std_devs,scaling=0.001,lamb=0.3*2.38 / np.sqrt(2 * self.n_dim))



                try:
                    trace = pm.sample(
                        draws=draws,
                        tune=0,
                        step=step,
                        chains=len(start_points),
                        initvals=start_points,
                        return_inferencedata=False,
                        compute_convergence_checks=False,
                        progressbar='combined',
                        cores=1,
                        discard_tuned_samples=True,
                    )
                except Exception as e:
                    print(f"Sampling Error: {e}")
                    raise

                start_points = []
                chunk_samples = []
                for chain_idx in range(len(trace.chains)):
                    chain_vals = trace.get_values("x", chains=chain_idx)

                    chunk_samples.append(chain_vals)
                    last_sample = chain_vals[-1]

                    start_points.append({"x": last_sample})

                chunk_samples = np.column_stack(chunk_samples)
                print("Acceptance rate:", np.mean(trace.get_sampler_stats("accepted")))
                samples_filename = os.path.join(checkpoint_dir, f"samples_pymc_step_{i + draws}.csv")

                # Save to CSV
                with open(samples_filename, "ab") as f:
                    np.savetxt(f, chunk_samples, delimiter=",")

        return chunk_samples

    def run_emcee(self, prior, n_effective, n_total, n_step=100, continue_run=False):
        """
        Runs Bayesian inference using emcee.

        Args:
            prior: The prior distribution.
            n_effective (int): Effective number of samples.
            n_total (int): Total number of steps.
            n_step (int, optional): Number of steps per save. Defaults to 100.
            continue_run (bool, optional): Whether to continue a previous run. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - samples (ndarray): Sampled parameters.
                - logl (ndarray): Log-likelihoods.
        """

        log_posterior = LogPosterior(prior, self.log_like)

        backend_path = os.path.join(self.path + 'data/', 'emcee_backend.h5')

        # ---- Handle backend creation or loading ----
        backend = emcee.backends.HDFBackend(backend_path)

        if continue_run and os.path.exists(backend_path):
            print("Continuing from existing run.")
            initial_state = None  # sampler will use backend’s last state
            completed_steps = backend.iteration
        else:
            if os.path.exists(backend_path):
                os.remove(backend_path)
            backend = emcee.backends.HDFBackend(backend_path)
            backend.reset(n_effective, self.n_dim)
            starting_points = prior.rvs(n_effective)
            initial_state = starting_points
            completed_steps = 0
            print("Starting new run.")


        sampler = emcee.EnsembleSampler(
            n_effective,
            self.n_dim,
            log_posterior,
            backend=backend,
            #moves=emcee.moves.StretchMove(a=1.2),
            moves=emcee.moves.DEMove(sigma=1e-05, gamma0=0.4*2.38 / np.sqrt(2 * self.n_dim)),
        )

        for i in range(completed_steps, n_total, n_step):
            steps = min(n_step, n_total - i)
            sampler.run_mcmc(initial_state, steps, progress=True)
            initial_state = None  # Only use the initial state once
            accept_frac = np.mean(sampler.acceptance_fraction)
            print(f"Saved {i + steps} / {n_total} steps with acceptance rate {accept_frac:.3f}")
            samples = backend.get_chain(discard=i, flat=True)
            samples_filename = os.path.join(self.path + 'data/', f'samples_emcee_step_{i + steps}.csv')
            np.savetxt(samples_filename, samples, delimiter=',')
        # ---- Extract samples ----
        samples = backend.get_chain(flat=True)
        logl = backend.get_log_prob(flat=True)

        np.savetxt(os.path.join(self.path + 'data/', 'samples_emcee.csv'), samples, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', 'log_likelihood_emcee.csv'), logl, delimiter=',')

        return samples, logl

    def run(self, prior, n_effective, n_total, n_cpus=1 ,continue_run=False, from_distribution=False, full_distribution=None):
        """
        Runs Bayesian inference using Pocomc.

        Args:
            prior: The prior distribution.
            n_effective (int): Effective number of samples.
            n_total (int): Total number of steps.
            n_cpus (int, optional): Number of CPUs to use. Defaults to 1.
            continue_run (bool, optional): Whether to continue a previous run. Defaults to False.
            from_distribution (bool, optional): Whether to sample from a full distribution. Defaults to False.
            full_distribution (full_estimate, optional): The full estimate distribution. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - samples_reweighted (ndarray): Reweighted samples.
                - logl_reweighted (ndarray): Reweighted log-likelihoods.
                - logz (float): Log-evidence.
        """

        if not from_distribution:
            if n_cpus==1:
                sampler = pc.Sampler(n_effective=n_effective,
                                     n_active=n_effective//2,
                                     n_dim=self.n_dim,
                                     likelihood=self.log_like,
                                     prior=prior,
                                     output_dir = self.path,
                                     dynamic=False,
                                    )

            else:
                sampler = pc.Sampler(n_effective=n_effective,
                                     n_active=n_effective//2,
                                     n_dim=self.n_dim,
                                     likelihood=self.log_like,
                                     prior=prior,
                                     output_dir = self.path,
                                     pool=n_cpus,
                                     dynamic=False,
                                    )

        else:
            # Define a fake log-likelihood when sampling from a full distribution
            def fake_log_like(x):
                return self.log_like(x)+prior.logpdf(x)-full_distribution.logpdf(x)

            if n_cpus==1:
                sampler = pc.Sampler(n_effective=n_effective,
                                     n_active=n_effective//2,
                                     n_dim=self.n_dim,
                                     likelihood=fake_log_like,
                                     prior=full_distribution,
                                     output_dir = self.path,
                                     dynamic=False,
                                    )

            else:
                sampler = pc.Sampler(n_effective=n_effective,
                                     n_active=n_effective//2,
                                     n_dim=self.n_dim,
                                     likelihood=fake_log_like,
                                     prior=full_distribution,
                                     output_dir = self.path,
                                     pool=n_cpus,
                                     dynamic=False,
                                    )

        # Handle continuing a previous run
        if continue_run:

            if os.path.exists(os.path.join(self.path, "pmc_final.state")):

                print('Run already complete')
                sampler.run(n_total=n_total,n_evidence=n_total,save_every=2, resume_state_path = os.path.join(self.path, "pmc_final.state"))

            else:
                pmc_files = []

                # Iterate through files in the folder
                for file in os.listdir(self.path):
                    match = re.match(r"pmc_(\d+)\.state$", file)
                    if match:
                        pmc_files.append((int(match.group(1)), file))

                if not pmc_files:
                    sampler.run(n_total=n_total,n_evidence=n_total,save_every=2)

                else:
                    # Find the file with the highest number
                    path_continue=max(pmc_files, key=lambda x: x[0])[1]

                    sampler.run(n_total=n_total,n_evidence=n_total,save_every=2, resume_state_path = os.path.join(self.path, path_continue))

        else:
            sampler.run(n_total=n_total,n_evidence=n_total,save_every=2)

        # Get results and samples
        results = sampler.results
        samples, weights, logl, logp = sampler.posterior()
        samples_reweighted, logl_reweighted, logp_reweighted = sampler.posterior(resample=True)

        # Save results and samples
        np.save(os.path.join(self.path + 'data/',f'results_dict.npy'), results)

        np.savetxt(os.path.join(self.path + 'data/', f'samples.csv'), samples, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', f'weights.csv'), weights, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', f'log_likelihood.csv'), logl, delimiter=',')

        np.savetxt(os.path.join(self.path + 'data/', f'samples_reweighted.csv'), samples_reweighted, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', f'log_likelihood_reweighted.csv'), logl_reweighted, delimiter=',')

        logz,logzerr=sampler.evidence()
        np.savetxt(os.path.join(self.path + 'data/', f'logz.csv'), [logz,logzerr], delimiter=',')

        return samples_reweighted, logl_reweighted, logz

    def create_synthetic_data_from_samples(self, samples, path):
        """
        Creates synthetic isotherm data from sampled parameters.

        Args:
            samples (ndarray): Sampled parameters.
            path (str): Output path for synthetic data.
        """

        # Compute the geometric medoid of the samples
        medoid=geometric_medoid(samples)

        dg = medoid[:self.cumulative_indices[0]],
        dh = medoid[self.cumulative_indices[0]:self.cumulative_indices[1]],

        # Create and save synthetic isotherms for each original isotherm
        for i,iso in enumerate(self.list_isotherms):

            total_concentrations = medoid[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][:iso.number_components],
            dh_0 = medoid[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components]
            sigma = medoid[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components+1]

            iso_syn=self.binding.synthetic_isotherm(dg, dh, total_concentrations, dh_0, sigma, iso.names_components, iso.concentration_estimate, iso.in_syringe, iso.filter_prior, iso.inj_list[0]*1e6, iso.inj_list[-1]*1e6, len(iso.inj_list)+1, iso.Temp , iso.V0)
            iso_syn.save(path, 'iso'+str(i+1))

    def compute_dd_samples(self, samples, dd_combinations, labels=None):
        """
        Computes differential free energy and enthalpy samples.

        Args:
            samples (ndarray): Sampled parameters.
            dd_combinations (ndarray): Combinations for differential values.
            labels (list, optional): Labels for differential values. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - dd_samples (ndarray): Differential free energy and enthalpy samples.
                - dd_labels (ndarray): Labels for differential values.
        """

        N, total_dim = samples.shape
        k, d = dd_combinations.shape

        dd_samples = np.zeros((N, 2 * k))

        dd_labels = np.zeros(k, dtype='U100')

        # Calculate differential free energy and enthalpy for each combination
        for i in range(k):

            dd_samples[:, i] = np.dot(samples[:, :d], dd_combinations[i])

            dd_samples[:, k + i] = np.dot(samples[:, d:2*d], dd_combinations[i])

            dd_labels[i] = f"comb({str(dd_combinations[i])})"

        # Create labels for differential values
        dd_labels=dd_labels = np.concatenate([np.char.add('g', dd_labels),np.char.add('h', dd_labels)]) if labels is None else np.concatenate([np.char.add('g', labels),np.char.add('h', labels)])

        return dd_samples, dd_labels


    def rescale_prior(self, prior_shape, filtering, samples, prior_width_final, jeffreys_sigma=False, h0_auto=True, nuisance_bounds = np.array([[-50.0, 50.0],[0.001, 10.0]])):
        """
        Rescales the prior based on final prior widths.

        Args:
            prior_shape (str): Shape of the prior.
            filtering (bool): Enable filtering.
            samples (ndarray): Sampled parameters.
            prior_width_final (float or list): Final prior width(s).
            jeffreys_sigma (bool, optional): Use Jeffreys prior for sigma. Defaults to False.
            h0_auto (bool, optional): Automatically determine h0 bound. Defaults to True.
            nuisance_bounds (ndarray, optional): Bounds for nuisance parameters. Defaults to np.array([[-50.0, 50.0],[0.001, 10.0]]).

        Returns:
            ndarray: Rescaled weights.
        """

        # Create a mapping from component names to indices
        name_to_ind = dict(zip(self.binding.names_components, np.linspace(-self.n_dim +self.cumulative_indices[-1],-1,self.n_dim-self.cumulative_indices[-1],dtype=int)))

        # Get prior distributions for isotherms with original and final prior widths
        prior_isotherms=[iso.get_prior(prior_shape=prior_shape, filtering=filtering, nuisance_bounds=nuisance_bounds,
                                                jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto) for iso in self.list_isotherms]

        # Calculate the difference in log-prior probabilities
        diff=(-sum(prior.logpdf(samples[:,self.cumulative_indices[1+i]:self.cumulative_indices[2+i]],
                                prior_width=samples[:,[name_to_ind[name] for name in prior.iso.names_components]].T) for i, prior in enumerate(prior_isotherms))
              + sum(prior.logpdf(samples[:,self.cumulative_indices[1+i]:self.cumulative_indices[2+i]],
                                 prior_width=prior_width_final) for i, prior in enumerate(prior_isotherms)))

        # Calculate weights based on the difference
        weights=np.exp(diff)

        return weights

    def evaluate_prior_dependence(self, prior_shape, filtering, samples, prior_width_original, prior_width_min, bounds_plot, jeffreys_sigma=False, h0_auto=True, prior_width_max=None, truths=None, logz=0., name='prior_dependence', nuisance_bounds = np.array([[-50.0, 50.0],[0.001, 10.0]])):
        """
        Evaluates the dependence of results on the prior width.

        Args:
            prior_shape (str): Shape of the prior.
            filtering (bool): Enable filtering.
            samples (ndarray): Sampled parameters.
            prior_width_original (float or list): Original prior width(s).
            prior_width_min (float): Minimum prior width to evaluate.
            bounds_plot (ndarray): Bounds for plotting.
            jeffreys_sigma (bool, optional): Use Jeffreys prior for sigma. Defaults to False.
            h0_auto (bool, optional): Automatically determine h0 bound. Defaults to True.
            prior_width_max (float, optional): Maximum prior width to evaluate. Defaults to None.
            truths (ndarray, optional): True parameter values. Defaults to None.
            logz (float, optional): Log-evidence. Defaults to 0..
            name (str, optional): Name for output files. Defaults to 'prior_dependence'.
            nuisance_bounds (ndarray, optional): Bounds for nuisance parameters. Defaults to np.array([[-50.0, 50.0],[0.001, 10.0]]).

        Returns:
            tuple: A tuple containing:
                - figures (list): List of generated figures.
                - widtharray (ndarray): Array of prior widths evaluated.
                - logzarray (ndarray): Array of log-evidence values.
                - weights (ndarray): Array of weights.
        """

        if prior_width_max is None:
            prior_width_max = prior_width_original

        # Get prior distributions for isotherms with original prior width
        prior_isotherms_original=[iso.get_prior(prior_width=prior_width_original, prior_shape=prior_shape, filtering=filtering, nuisance_bounds=nuisance_bounds,jeffreys_sigma=jeffreys_sigma,h0_auto=h0_auto) for iso in self.list_isotherms]

        # Generate an array of prior widths to evaluate
        widtharray=np.linspace(prior_width_min,prior_width_max,50)
        logzarray=[None]*len(widtharray)
        percentiles=[None]*len(widtharray)
        weights=[None]*len(widtharray)

        # Calculate the difference in log-prior probabilities for the original prior
        diff0=-sum(prior.logpdf(samples.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T) for i, prior in enumerate(prior_isotherms_original))

        # Evaluate log-evidence and percentiles for each prior width
        for k,w in enumerate(widtharray):
            prior_isotherms_new = [iso.get_prior(prior_width=w, prior_shape=prior_shape, filtering=filtering, nuisance_bounds=nuisance_bounds,jeffreys_sigma=jeffreys_sigma,h0_auto=h0_auto) for iso in self.list_isotherms]
            diff=diff0+sum(prior.logpdf(samples.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T) for i, prior in enumerate(prior_isotherms_new))
            logzarray[k]=logz+np.logaddexp.reduce(diff)-np.log(len(samples))
            weights[k]=np.exp(diff)
            print(logzarray[k],weights[k])
            percentiles[k]=np.percentile(samples,(2.5,25,50,75,97.5),axis=0,weights=weights[k], method='inverted_cdf')

        percentiles=np.array(percentiles)

        # Save results
        np.savetxt(os.path.join(self.path + 'data/', 'widtharray.csv'), widtharray, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', 'logzarray.csv'), logzarray, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', 'weightsarray.csv'), weights, delimiter=',')

        figures=[]

        # Plot log-evidence vs prior width
        fig, ax = plt.subplots()
        ax.plot(widtharray,logzarray,color='k')
        ax.set_ylim(logzarray[-1]*1.1-np.max(logzarray)*0.1,np.max(logzarray)*1.1-0.1*logzarray[-1])
        ax.plot(prior_width_original, logz, 'o', color='k')
        plt.savefig(os.path.join(self.path +'figures/',name+'_logz.png'))
        plt.close(fig)

        figures.append(fig)

        # Plot percentiles vs prior width for each parameter
        for i in range(len(samples.T)):

            fig, ax = plt.subplots()
            ax.plot(widtharray,percentiles[:,0,i],color='k')
            ax.plot(widtharray,percentiles[:,1,i],color='k')
            ax.plot(widtharray,percentiles[:,2,i],color='k',linestyle='dotted')
            ax.plot(widtharray,percentiles[:,3,i],color='k')
            ax.plot(widtharray,percentiles[:,4,i],color='k')

            ax.set_ylim(bounds_plot[i])
            ax.set_xlim(widtharray[0],widtharray[-1])
            ax.fill_between(widtharray,percentiles[:,0,i],percentiles[:,4,i], color='gray', alpha=0.5)
            ax.fill_between(widtharray,percentiles[:,1,i],percentiles[:,3,i], color='gray', alpha=0.5)

            ax.set_title(self.labels[i], fontsize=20)  # Set title font size
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)

            if truths is not None:
                ax.axvline(truths[i], color='k')

            ax.axvline(widtharray[np.argmax(logzarray)], color='k', linestyle='dotted')
            plt.savefig(os.path.join(self.path +'figures/',name+'_'+self.labels[i]+'.png'))
            plt.close(fig)

            figures.append(fig)


        return figures, widtharray, logzarray, weights

    def plot_corner(self, samples, bounds_plot, name='corner', indices=None, weights=None, truths=None, labels=None, colors=['k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']):
        """
        Generates a corner plot of the posterior samples.

        Args:
            samples (ndarray or list): Posterior samples.
            bounds_plot (ndarray): Bounds for plotting.
            name (str, optional): Name for the output file. Defaults to 'corner'.
            indices (list, optional): Indices of parameters to plot. Defaults to None.
            weights (ndarray or list, optional): Weights for the samples. Defaults to None.
            truths (ndarray, optional): True parameter values. Defaults to None.
            labels (list, optional): Labels for the parameters. Defaults to None.
            colors (list, optional): Colors for the plots. Defaults to ['k','tab:blue',...].

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """

        if labels is None:
            labels=self.labels

        # Handle single or multiple sets of samples
        if not isinstance(samples, list):
            samples = [samples]
        if weights is None:
            weights = [None] * len(samples)
        if truths is None:
            truths = np.array([None] * len(samples[0].T))
        elif not isinstance(weights, list):
            weights = [weights] * len(samples)

        if indices is None:
            indices=[i for i in range(len(samples[0].T))]

        # Generate the corner plot
        fig=corner.corner(samples[0][:,indices],color=colors[0],weights=weights[0],range=bounds_plot[indices],truth_color='k',truths=truths[indices], bins=80,labels=labels[indices],plot_datapoints=False,hist_kwargs=dict(density=True),plot_density=True,hist2d_kwargs=dict(quiet=True))
        # Add additional sample sets to the plot
        for i in range(1,len(samples)):
            corner.corner(samples[i][:,indices],fig=fig,color=colors[i],weights=weights[i],range=bounds_plot[indices],truth_color='k',truths=truths[indices],bins=80,plot_datapoints=False,hist_kwargs=dict(density=True),plot_density=True,hist2d_kwargs=dict(quiet=True))

        # Save and close the figure
        plt.savefig(os.path.join(self.path +'figures/',name+'.png'))
        plt.close(fig)
        return fig

    def plot_histograms(self, samples, bounds_plot, name='hist', weights=None,truths=None, labels=None,colors=['k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']):
        """
        Generates histograms for each parameter.

        Args:
            samples (ndarray or list): Posterior samples.
            bounds_plot (ndarray): Bounds for plotting.
            name (str, optional): Name for the output files. Defaults to 'hist'.
            weights (ndarray or list, optional): Weights for the samples. Defaults to None.
            truths (ndarray, optional): True parameter values. Defaults to None.
            labels (list, optional): Labels for the parameters. Defaults to None.
            colors (list, optional): Colors for the histograms. Defaults to ['k','tab:blue',...].

        Returns:
            list: A list of generated figures.
        """

        if labels is None:
            labels=self.labels

        # Handle single or multiple sets of samples
        if not isinstance(samples, list):
            samples = [samples]
        if weights is None:
            weights = [None] * len(samples)
        elif not isinstance(weights, list):
            weights = [weights] * len(samples)

        figures = []

        # Generate and save a histogram for each parameter
        for i in range(len(samples[0].T)):

            fig, ax = plt.subplots()
            for j in range(len(samples)):
                ax.hist(samples[j][:,i], bins=80, range=bounds_plot[i],weights=weights[j], histtype='step', color=colors[j],density=True)
            ax.set_title(labels[i])
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.tick_params(axis='both', labelsize=12)
            if truths is not None:
                ax.axvline(truths[i], color='k')
            ax.set_xlim(bounds_plot[i][0], bounds_plot[i][1])
            plt.savefig(os.path.join(self.path +'figures/',name+'_'+labels[i]+'.png'))
            plt.close(fig)

            figures.append(fig)

        return figures

    def plot_isotherms(self, samples, name='iso', colors=['grey','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan'], format_save='.png', size_mini=False):
        """
        Plots the isotherms based on the posterior samples.

        Args:
            samples (ndarray or list): Posterior samples.
            name (str, optional): Name for the output files. Defaults to 'iso'.
            colors (list, optional): Colors for the plotted isotherms. Defaults to ['grey','tab:blue',...].
            format_save (str, optional): Format to save the figures. Defaults to '.png'.
            size_mini (boolean, optional): Optional smaller version of figures.
            
        Returns:
            list: A list of generated figures.
        """

        # Handle single or multiple sets of samples
        if not isinstance(samples, list):
            samples = [samples]

        figures = []
        # Select random indices from samples for plotting
        inds = np.random.randint(list(map(len,samples)), size=(1000,len(samples)))

        # Plot predicted isotherms for each original isotherm
        for j, iso in enumerate(self.list_isotherms):
            if size_mini:
                fig, ax = plt.subplots(figsize=(3, 2))
            else:
                fig, ax = plt.subplots()
            for k in range(len(samples)):
                for i, ind in enumerate(inds[:,k]):
                    parameters = samples[k][ind]

                    # Calculate predicted dq values
                    y_pred_i = iso.get_dq_list(dg = parameters[:self.cumulative_indices[0]],
                                                 dh = parameters[self.cumulative_indices[0]:self.cumulative_indices[1]],
                                                 total_concentrations = parameters[self.cumulative_indices[1+j]:self.cumulative_indices[2+j]][:iso.number_components],
                                                 dh_0 = parameters[self.cumulative_indices[1+j]:self.cumulative_indices[2+j]][iso.number_components])

                    # Plot predicted isotherm with transparency
                    if size_mini:
                        ax.plot(y_pred_i[1:], alpha=0.01, color=colors[k],linewidth=0.3)
                    else:
                        ax.plot(y_pred_i, alpha=0.01, color=colors[k])
                        
            # Plot original isotherm data
            if size_mini:
                ax.plot(iso.dq_list[1:], ls='None', color='black', marker='o',ms=2)
                ax.plot([x for x in iso.skipped_injections[1:]],iso.dq_list[[x - 1 for x in iso.skipped_injections[1:]]], ms=2,ls='None', color='r', marker='o')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(labelbottom=False, labelleft=False)
                ax.set_ylim(min(iso.dq_list[1:])-2,max(iso.dq_list[1:])+2)
            else:
                ax.plot(iso.dq_list, ls='None', color='black', marker='o')
                ax.plot([x - 1 for x in iso.skipped_injections],iso.dq_list[[x - 1 for x in iso.skipped_injections]], ls='None', color='r', marker='o')
                ax.set_ylim(min(iso.dq_list)-2,max(iso.dq_list)+2)
            # Set y-axis limits and title
            
            ax.set_title('iso'+str(j+1)+' '+' '.join(np.char.add(iso.names_components,np.where(iso.in_syringe, '_s', '_c'))))

            # Save and close the figure
            plt.savefig(os.path.join(self.path +'figures/',name+'_'+str(j+1)+format_save))
            plt.close(fig)

            figures.append(fig)

        return figures
        
class binding_structure:
    """
    Represents the binding structure of components and their interactions, including
    stoichiometry, degeneracy, and binding states.

    Attributes:
        components (ndarray): A 2D numpy array where each row represents a binding
                              stage and each column represents a component. The values
                              indicate the stoichiometry of each component in that stage.
        names_components (ndarray): A 1D numpy array of strings representing the names
                                    of the components.
        degeneracy (ndarray): A 1D numpy array representing the degeneracy of each
                              binding stage.
        binding_states (ndarray): A 1D numpy array of strings representing the names
                                  of the binding states (e.g., '1', '2', '1-2').
        number_stages (int): The total number of binding stages.
        number_components (int): The total number of components.
    """

    def __init__(self, components, names_components, degeneracy, binding_states):
        """
        Initializes a binding_structure object.

        Args:
            components (ndarray): Stoichiometry of components in each binding stage.
            names_components (ndarray): Names of the components.
            degeneracy (ndarray): Degeneracy of each binding stage.
            binding_states (ndarray): Names of the binding states.
        """

        self.components=np.copy(components)
        self.names_components=np.copy(names_components)
        self.degeneracy=np.copy(degeneracy)
        self.binding_states=np.copy(binding_states)

        self.number_stages=len(components)
        self.number_components=len(components.T)

    def solve_for_free_concentrations(self,free_concentrations,total_concentrations,k):
        """
        Calculates the residual for solving for free concentrations using the law of mass action.

        This function is used within a root-finding algorithm (like `scipy.optimize.root`)
        to find the free concentrations of components given the total concentrations
        and binding constants (k).

        Args:
            free_concentrations (ndarray): An array of initial guesses for the free
                                           concentrations of each component.
            total_concentrations (ndarray): An array of the total concentrations of
                                            each component.
            k (ndarray): An array of association binding constants for each binding stage.

        Returns:
            ndarray: An array of residuals. The root of these residuals corresponds to
                     the correct free concentrations.
        """

        return total_concentrations - np.abs(free_concentrations) - np.sum(np.prod(np.abs(free_concentrations)**self.components,axis=1)*self.degeneracy/k*self.components.T,axis=1)

    def jacobian_solve_for_free_concentrations(self, free_concentrations, total_concentrations, k):
        """
        Calculates the Jacobian matrix of the `solve_for_free_concentrations` function.

        This Jacobian is used by some root-finding algorithms to improve convergence.

        Args:
            free_concentrations (ndarray): An array of free concentrations.
            total_concentrations (ndarray): An array of total concentrations.
            k (ndarray): An array of association binding constants.

        Returns:
            ndarray: The Jacobian matrix.
        """

        free_concentrations=np.abs(free_concentrations)+1e-15 # Add a small value to avoid division by zero
        common = np.prod((free_concentrations)**self.components,axis=1)*self.degeneracy/k*self.components.T
        jacobian = np.eye(len(free_concentrations))-np.sum(common[:, None, :] * self.components.T[None, :, :], axis=2) / (free_concentrations[None, :])

        return jacobian

    def min_solve_for_free_concentrations(self, free_concentrations, total_concentrations, k):
        """
        Calculates the sum of squared residuals for solving for free concentrations.

        This function is used by minimization algorithms to find the free concentrations
        by minimizing the sum of squared differences between the total and calculated
        concentrations.

        Args:
            free_concentrations (ndarray): An array of free concentrations.
            total_concentrations (ndarray): An array of total concentrations.
            k (ndarray): An array of association binding constants.

        Returns:
            float: The sum of squared residuals.
        """
        return np.sum(self.solve_for_free_concentrations(free_concentrations,total_concentrations,k)**2)

    def jacobian_min_solve_for_free_concentrations(self, free_concentrations, total_concentrations, k):
        """
        Calculates the Jacobian of the `min_solve_for_free_concentrations` function.

        Args:
            free_concentrations (ndarray): An array of free concentrations.
            total_concentrations (ndarray): An array of total concentrations.
            k (ndarray): An array of association binding constants.

        Returns:
            ndarray: The Jacobian of the sum of squared residuals.
        """
        return np.sum(2*self.solve_for_free_concentrations(free_concentrations,total_concentrations,k)*self.jacobian_solve_for_free_concentrations(free_concentrations, total_concentrations, k),axis=0)

    def get_single_concentrations(self, dg, dh, initial_concentrations,Temp=298.15):
        """
        Calculates the equilibrium concentrations of free and bound species.

        This function uses a root-finding algorithm to solve the system of equations
        derived from the law of mass action and conservation of mass to find the
        free concentrations. Then, it calculates the concentrations of the bound species.

        Args:
            dg (ndarray): An array of standard Gibbs free energies of binding for each stage.
            dh (ndarray): An array of standard enthalpies of binding for each stage.
            initial_concentrations (ndarray): An array of the initial total concentrations
                                              of each component.
            Temp (float, optional): The temperature in Kelvin. Defaults to 298.15.

        Returns:
            ndarray: A 2D array where the first row contains the free concentrations
                     and subsequent rows contain the concentrations of each bound species.
        """

        k = np.exp(dg/(R*Temp)) # Calculate association binding constants from free energy

        concentrations_for_solver = initial_concentrations

        # Use different root-finding methods with increasing tolerance if initial attempts fail
        sol = root(self.solve_for_free_concentrations,x0=concentrations_for_solver,args=(concentrations_for_solver,k),method='hybr',options={'xtol':1e-4})


        if np.sum(np.abs(self.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver,k)))/np.sum(np.abs(sol.x))>1e-3:

            sol = root(self.solve_for_free_concentrations,x0=concentrations_for_solver,args=(concentrations_for_solver,k),jac=self.jacobian_solve_for_free_concentrations,method='hybr',options={'xtol':1e-10})
            if np.sum(np.abs(self.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver,k)))/np.sum(np.abs(sol.x))>1e-3:
                sol = root(self.solve_for_free_concentrations,x0=concentrations_for_solver,args=(concentrations_for_solver,k),method='lm',options={'xtol':1e-10})
                if np.sum(np.abs(self.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver,k)))/np.sum(np.abs(sol.x))>1e-3:
                    sol = root(self.solve_for_free_concentrations,x0=concentrations_for_solver,args=(concentrations_for_solver,k),jac=self.jacobian_solve_for_free_concentrations,method='lm',options={'xtol':1e-10})
                    if np.sum(np.abs(self.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver,k)))/np.sum(np.abs(sol.x))>1e-3:
                        sol = root(self.solve_for_free_concentrations,x0=concentrations_for_solver,args=(concentrations_for_solver,k),method='df-sane',options={'ftol':1e-15})

        fc_list=np.array([np.abs(sol.x)]) # Get free concentrations (ensure they are non-negative)

        # Calculate concentrations of bound species using free concentrations and binding constants
        conc_binding = np.prod(fc_list[:, None, :] ** self.components[None, :, :],axis=2)*self.degeneracy/k

        return np.concatenate((fc_list,conc_binding),axis=1) # Return free and bound concentrations

    def get_prior(self, bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior, posterior_indices=None, posterior_data=None):
        """
        Creates a prior distribution object for the thermodynamic parameters.

        Args:
            bounds (ndarray): Bounds for the thermodynamic parameters (free energies and enthalpies).
            dd_combinations (ndarray): Combinations for calculating differential values.
            ddg_bounds (ndarray): Bounds for differential free energy.
            ddh_bounds (ndarray): Bounds for differential enthalpy.
            posterior (bool): Whether to use posterior information from a previous run.
            posterior_indices (list, optional): Indices of parameters informed by the posterior. Defaults to None.
            posterior_data (ndarray, optional): Posterior data to inform the prior. Defaults to None.

        Returns:
            prior_thermo: A prior_thermo distribution object.
        """

        return prior_thermo(self, bounds, dd_combinations, ddg_bounds, ddh_bounds,  posterior, posterior_indices, posterior_data)

    def reduce(self, names_components):
        """
        Reduces the binding structure to a subset of components.

        This method creates a new binding structure that only includes the specified
        components and the binding stages that involve only these components.

        Args:
            names_components (list): A list of strings representing the names of the
                                     components to include in the reduced structure.

        Returns:
            tuple: A tuple containing:
                - binding_structure: The reduced binding structure object.
                - filter (ndarray): A boolean array indicating which original binding
                                    stages were kept.
        """

        iso_indexes_for_filter = [list(self.names_components).index(name) for name in names_components] # Get indices of specified components
        non_iso_indexes = [i for i, name in enumerate(self.names_components) if name not in names_components] # Get indices of components not in the list
        filter=(self.components[:, non_iso_indexes] == 0).all(axis=1) # Determine which stages only involve the specified components

        # Create and return the reduced binding structure and the filter array
        return binding_structure(self.components[filter][:,iso_indexes_for_filter], names_components, self.degeneracy[filter], binding_states=self.binding_states[filter]), filter

    def synthetic_isotherm(self, dg, dh, total_concentrations, dh_0, sigma, names_components_syn, concentration_estimate_syn, in_syringe_syn, filter_prior_syn=None, first_inj_vol= 2, inj_vol = 10, inj_count = 35, Temp=298.15 , V0 = 1.42e-3, show_concs=False, colors= [
                                                                                                                                                                                                                                                                    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                                                                                                                                                                                                                                                                    'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan',
                                                                                                                                                                                                                                                                    'darkblue', 'gold', 'lime', 'crimson', 'indigo',
                                                                                                                                                                                                                                                                    'maroon', 'teal', 'turquoise', 'lightgreen', 'darkorange',
                                                                                                                                                                                                                                                                    'navy', 'magenta', 'salmon', 'orchid', 'darkgrey'
                                                                                                                                                                                                                                                                    ]):
        """
        Generates synthetic isotherm data based on the binding structure and provided parameters.

        This method simulates an ITC experiment to produce a synthetic isotherm
        (a plot of differential heat per injection versus injection number)
        based on given binding parameters and experimental conditions.

        Args:
            dg (ndarray): Standard Gibbs free energies of binding.
            dh (ndarray): Standard enthalpies of binding.
            total_concentrations (ndarray): Total concentrations of components.
            dh_0 (float): Differential heat of injection before binding.
            sigma (float): Standard deviation of the noise in the simulated data.
            names_components_syn (list): Names of components for the synthetic isotherm.
            concentration_estimate_syn (ndarray): Estimated concentrations for the synthetic isotherm.
            in_syringe_syn (ndarray): Boolean array indicating which components are in the syringe.
            filter_prior_syn (ndarray, optional): Prior filtering information. Defaults to None.
            first_inj_vol (float, optional): Volume of the first injection in uL. Defaults to 2.
            inj_vol (float, optional): Volume of subsequent injections in uL. Defaults to 10.
            inj_count (int, optional): Number of injections after the first. Defaults to 35.
            Temp (float, optional): Temperature in Kelvin. Defaults to 298.15.
            V0 (float, optional): Initial volume of the cell in L. Defaults to 1.42e-3.
            show_concs (bool, optional): Whether to plot the concentrations of species during the simulation. Defaults to False.
            colors (list, optional): List of colors for plotting. Defaults to a predefined list.

        Returns:
            isotherm: An isotherm object containing the synthetic data.
        """

        # Define injection volumes
        inj_list_syn = [first_inj_vol*1e-6] + [inj_vol*1e-6]*inj_count

        # Define skipped injections (usually the first one)
        skipped_injections_syn = [1]

        # Get indices of components relevant to the synthetic isotherm
        iso_indexes = [i for i, name in enumerate(self.names_components) if name in names_components_syn]

        # Combine true parameters for potential use (e.g., plotting truths)
        truths_syn=list(dg)+list(dh)+list(total_concentrations[iso_indexes])+[dh_0]+[sigma]

        # Filter prior information if provided
        if filter_prior_syn is None:
            filter_prior=None
        else:
            filter_prior=filter_prior_syn[iso_indexes]

        # Create a synthetic isotherm object
        isotherm_syn=isotherm(self, inj_list_syn=inj_list_syn, skipped_injections_syn=skipped_injections_syn, concentration_estimate_syn=concentration_estimate_syn[iso_indexes], in_syringe_syn=in_syringe_syn[iso_indexes], filter_prior_syn=filter_prior, names_components_syn=names_components_syn,  truths_syn= truths_syn, Temp=Temp , V0=V0)
        # Plot the generated synthetic isotherm
        fig=isotherm_syn.plot(dg, dh, total_concentrations[iso_indexes], dh_0)
        fig.show()

        # Optionally show concentration profiles during the simulation
        if show_concs:
            concs = isotherm_syn.get_conc(dg, dh, total_concentrations[iso_indexes], dh_0) # Get concentrations of species
            labels = np.concatenate((isotherm_syn.binding.names_components, isotherm_syn.binding.binding_states)) # Create labels for species

            # Create a single figure with two subplots for linear and log scale concentration plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left plot: Linear scale
            for i in range(len(concs.T)):
                axes[0].plot(concs.T[i], label=labels[i], color=colors[i])
            axes[0].set_title('Linear Scale', fontsize=14)
            axes[0].set_ylabel('Concentration', fontsize=12)

            # Right plot: Log scale
            for i in range(len(concs.T)):
                axes[1].plot(concs.T[i], label=labels[i], color=colors[i])
            axes[1].set_title('Log Scale', fontsize=14)
            axes[1].set_yscale('log')

            # Combine legends and adjust placement
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc='lower center', ncol=12, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.05)
            )

            # Adjust layout and display the plot
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the bottom for the legend
            plt.show()

        return isotherm_syn # Return the synthetic isotherm object

class isotherm:
    """
    Represents an isotherm experiment, including experimental data and methods
    for calculating heat changes and species concentrations.

    Attributes:
        V0 (float): Initial volume of the cell in liters.
        Temp (float): Temperature of the experiment in Kelvin.
        dq_list (ndarray): Array of differential heat values per injection.
        inj_list (ndarray): Array of injection volumes in liters.
        skipped_injections (ndarray): Array of indices of skipped injections.
        concentration_estimate (ndarray): Array of estimated concentrations of components.
        in_syringe (ndarray): Boolean array indicating which components are in the syringe.
        names_components (ndarray): Array of names of the components.
        filter_prior (ndarray or None): Prior filtering information.
        truths (ndarray or None): True parameter values if known.
        number_components (int): Number of components.
        skipped_mask (ndarray): Boolean mask for non-skipped injections.
        conc_scaling (ndarray): Scaling factors for concentrations due to injections.
        binding (binding_structure): Reduced binding structure relevant to this isotherm.
        filter (ndarray): Boolean array indicating which binding stages are included.
        binding_syr (binding_structure): Binding structure for components in the syringe.
        filter_syr (ndarray): Boolean array for syringe components.
        reaction_in_syringe (bool): Indicates if a reaction occurs in the syringe.
        binding_diff (binding_structure): Binding structure for differential calculations.
        filter_diff (ndarray): Boolean array for differential calculations.
    """

    def __init__(self, binding, filepath=None, inj_list_syn=None, skipped_injections_syn=None, concentration_estimate_syn=None, in_syringe_syn=None, filter_prior_syn=None, names_components_syn=None, truths_syn=None, Temp=298.15 , V0 = 1.42e-3):
        """
        Initializes an isotherm object, either from a file or synthetic data.

        Args:
            binding (binding_structure): The overall binding structure.
            filepath (str, optional): Path to a file containing experimental data. Defaults to None.
            inj_list_syn (list, optional): List of injection volumes for synthetic data. Defaults to None.
            skipped_injections_syn (list, optional): List of skipped injection indices for synthetic data. Defaults to None.
            concentration_estimate_syn (ndarray, optional): Estimated concentrations for synthetic data. Defaults to None.
            in_syringe_syn (ndarray, optional): Boolean array for components in syringe for synthetic data. Defaults to None.
            filter_prior_syn (ndarray, optional): Prior filtering information for synthetic data. Defaults to None.
            names_components_syn (list, optional): Names of components for synthetic data. Defaults to None.
            truths_syn (ndarray, optional): True parameter values for synthetic data. Defaults to None.
            Temp (float, optional): Temperature in Kelvin. Defaults to 298.15.
            V0 (float, optional): Initial volume of the cell in liters. Defaults to 1.42e-3.
        """

        self.V0 = V0 # Store initial cell volume
        self.Temp = Temp # Store temperature

        if filepath is not None:
            # Load data from file
            inj_list = []
            dq_list = []
            skipped_injections = []
            concentration_estimate = []
            in_syringe = []
            filter_prior=[]
            names_components = []
            truths=[]

            with open(filepath) as f:
                for line in f:
                    values = line.strip('\n').replace(' ','').split(',')
                    if values[0].casefold() == 'skipped_inj':
                        for i in range(len(values)-1):
                            skipped_injections.append(int(values[i+1]))

                    elif values[0].casefold() == 'truths':
                        for i in range(len(values)-1):
                            truths.append(float(values[i+1]))
                            
                    elif values[0].casefold() == 'v0':
                        self.V0 = float(values[1]) # Overwrite initial value if additional information given

                    elif values[0].casefold() == 'concentration_estimate':
                        for i in range(len(values)-1):
                            concentration_estimate.append(float(values[i+1]))

                    elif values[0].casefold() == 'filter_prior':
                        for i in range(len(values)-1):
                            filter_prior.append(int(values[i+1]))

                    elif values[0].casefold() == 'components':
                        for i in range(len(values)-1):
                            names_components.append(values[i+1])

                    elif values[0].casefold() == 'in_syringe':
                        for i in range(len(values)-1):
                            in_syringe.append(values[i+1] == 'True')

                    elif values[0].casefold() == 't':
                        self.Temp = 273.15 + float(values[1]) # Convert Celsius to Kelvin

                    elif len(values) == 2:
                        dq_list.append(float(values[0]))
                        inj_list.append(float(values[1])*1e-6) # Convert uL to L

            self.dq_list = np.array(dq_list) # Differential heat list
            self.inj_list = np.array(inj_list) # Injection volume list
            self.skipped_injections = np.array(skipped_injections) # Skipped injections indices
            self.concentration_estimate = np.array(concentration_estimate) # Estimated concentrations
            self.in_syringe = np.array(in_syringe) # Components in syringe boolean
            self.names_components = np.array(names_components) # Component names

            if len(filter_prior):
                self.filter_prior = np.array(filter_prior) # Prior filter information

            else:
                self.filter_prior = None

            if len(truths):
                self.truths = np.array(truths) # True parameter values

            else:
                self.truths = None

        else:
            # Initialize with synthetic data
            self.dq_list = np.array([]) # Differential heat list (will be generated later)
            self.inj_list = np.array(inj_list_syn) # Injection volume list
            self.skipped_injections = np.array(skipped_injections_syn) # Skipped injections indices
            self.concentration_estimate = np.array(concentration_estimate_syn) # Estimated concentrations
            self.in_syringe = np.array(in_syringe_syn) # Components in syringe boolean
            self.names_components = np.array(names_components_syn) # Component names
            self.truths = np.array(truths_syn) # True parameter values

            if filter_prior_syn is None:
                self.filter_prior = None # Prior filter information

            else:
                self.filter_prior = np.array(filter_prior_syn)


        self.number_components = len(self.concentration_estimate) # Number of components

        # Create a mask for non-skipped injections
        self.skipped_mask = np.isin(np.arange(len(self.inj_list)), self.skipped_injections-1, invert=True)

        # Calculate concentration scaling factors due to injections
        self.conc_scaling = np.empty(shape=(self.number_components,len(self.inj_list)+1))
        self.conc_scaling[:,0] = ~self.in_syringe # Initial scaling (1 for cell, 0 for syringe)

        dcum = 1 # Cumulative dilution factor

        for i, inj in enumerate(self.inj_list):

            d  = 1 - (inj/self.V0) # Dilution factor for the current injection
            dcum *= d # Cumulative dilution factor

            # Concentration scaling for each component after injection
            self.conc_scaling[:,i+1] = (1 - dcum) * (self.in_syringe) + dcum * (~self.in_syringe)

        # Reduce the binding structure to include only components relevant to this isotherm
        self.binding, self.filter=binding.reduce(self.names_components)

        # Check if reactions occur in the syringe
        self.binding_syr, self.filter_syr = binding.reduce(self.names_components[self.in_syringe])
        if len(self.binding_syr.binding_states)==0:
            self.reaction_in_syringe=False # No reaction in syringe
        else:
            self.reaction_in_syringe=True # Reaction in syringe
            # Reduce the binding structure for differential calculations (components in syringe)
            self.binding_diff, self.filter_diff = self.binding.reduce(self.names_components[self.in_syringe])

        if filepath is None:
            # Generate synthetic data if no filepath is provided
            truths_syn=np.array(truths_syn)
            # Select true parameters relevant to this isotherm
            self.truths = np.array([truths_syn[i] for i in range(len(truths_syn)) if (i < len(self.filter) and self.filter[i]) or (i < 2*len(self.filter) and i >= len(self.filter) and self.filter[i-len(self.filter)]) or i>=int(2*len(self.filter))])
            # Calculate true differential heat values
            true_dq = self.get_dq_list(dg=truths_syn[:len(self.filter)],dh=truths_syn[len(self.filter):2*len(self.filter)], total_concentrations=truths_syn[2*len(self.filter):2*len(self.filter)+len(self.names_components)], dh_0=truths_syn[-2])
            # Add noise to true dq values to get observed dq values
            dq_obs = true_dq + np.random.normal(loc=0,scale=truths_syn[-1],size=len(true_dq))

            self.dq_list=np.array(dq_obs) # Observed differential heat list



    def save(self, path, name):
        """
        Saves the isotherm data to a file.

        Args:
            path (str): The directory path to save the file.
            name (str): The name of the file.
        """

        if not os.path.exists(path):
            os.makedirs(path) # Create directory if it doesn't exist

        with open(path+name, 'w') as synth_file:
            # Write isotherm data to the file
            synth_file.write('components,' + ','.join(self.names_components))
            synth_file.write('\n' + 'concentration_estimate,' + ','.join(map(str,self.concentration_estimate)))
            synth_file.write('\n' + 'in_syringe,' + ','.join(map(str,self.in_syringe)))
            synth_file.write('\n' + 'T,' + str(self.Temp-273.15))
            if self.filter_prior is not None:
                synth_file.write('\n' + 'filter_prior,' + ','.join(map(str,self.filter_prior)))
            if self.truths is not None:
                synth_file.write('\n' + 'truths,' + ','.join(map(str,self.truths)))

            for i, dq in enumerate(self.dq_list):

                synth_file.write('\n' + str(dq) + ',' + str(self.inj_list[i]*1e6)) # Write dq and injection volume (in uL)

            synth_file.write('\n' + 'skipped_inj,' + ','.join(map(str,self.skipped_injections))) # Write skipped injection indices

    def get_dq_list(self, dg, dh, total_concentrations, dh_0):
        """
        Calculates the predicted differential heat values per injection.

        Args:
            dg (ndarray): Standard Gibbs free energies of binding.
            dh (ndarray): Standard enthalpies of binding.
            total_concentrations (ndarray): Total concentrations of components.
            dh_0 (float): Differential heat of injection before binding.

        Returns:
            ndarray: Array of predicted differential heat values.
        """

        k = np.exp(dg/(R*self.Temp))[self.filter] # Calculate association binding constants for relevant stages

        concentrations_for_solver = total_concentrations*self.conc_scaling.T # Concentrations after each injection
        q_list = np.empty(len(concentrations_for_solver)) # Initialize array for total heat

        # Calculate heat change in the syringe if reaction occurs
        if self.reaction_in_syringe:
            if self.binding_syr.number_stages==1:
                # For a single binding stage in the syringe
                q_syr = dh[self.filter_syr]/2*(total_concentrations[self.in_syringe][0]+total_concentrations[self.in_syringe][1]+k[self.filter_diff]-np.sqrt((total_concentrations[self.in_syringe][0]+total_concentrations[self.in_syringe][1]+k[self.filter_diff])**2-4*total_concentrations[self.in_syringe][0]*total_concentrations[self.in_syringe][1]))

            else:
                # For multiple binding stages in the syringe, solve for free concentrations
                sol = root(self.binding_syr.solve_for_free_concentrations,x0=0.5*total_concentrations[self.in_syringe],args=(total_concentrations[self.in_syringe],k[self.filter_diff]),method='hybr',options={'xtol':1e-4})

                # check for convergence and try different methods if needed
                if np.sum(np.abs(self.binding_syr.solve_for_free_concentrations(np.abs(sol.x),total_concentrations[self.in_syringe],k[self.filter_diff])))/np.sum(np.abs(sol.x))>1e-3:
                    #print('1')
                    sol = root(self.binding_syr.solve_for_free_concentrations,x0=0.5*total_concentrations[self.in_syringe],args=(total_concentrations[self.in_syringe],k[self.filter_diff]),jac=self.binding_syr.jacobian_solve_for_free_concentrations,method='hybr',options={'xtol':1e-10})
                    if np.sum(np.abs(self.binding_syr.solve_for_free_concentrations(np.abs(sol.x),total_concentrations[self.in_syringe],k[self.filter_diff])))/np.sum(np.abs(sol.x))>1e-3:
                        #print('2')
                        sol = root(self.binding_syr.solve_for_free_concentrations,x0=0.5*total_concentrations[self.in_syringe],args=(total_concentrations[self.in_syringe],k[self.filter_diff]),method='lm',options={'xtol':1e-10})
                        if np.sum(np.abs(self.binding_syr.solve_for_free_concentrations(np.abs(sol.x),total_concentrations[self.in_syringe],k[self.filter_diff])))/np.sum(np.abs(sol.x))>1e-3:
                            #print('3')
                            sol = root(self.binding_syr.solve_for_free_concentrations,x0=0.5*total_concentrations[self.in_syringe],args=(total_concentrations[self.in_syringe],k[self.filter_diff]),jac=self.binding_syr.jacobian_solve_for_free_concentrations,method='lm',options={'xtol':1e-10})
                            if np.sum(np.abs(self.binding_syr.solve_for_free_concentrations(np.abs(sol.x),total_concentrations[self.in_syringe],k[self.filter_diff])))/np.sum(np.abs(sol.x))>1e-3:
                                #print('4')
                                sol = root(self.binding_syr.solve_for_free_concentrations,x0=0.5*total_concentrations[self.in_syringe],args=(total_concentrations[self.in_syringe],k[self.filter_diff]),method='df-sane',options={'ftol':1e-15})


                # Calculate heat in syringe from bound species concentrations
                q_syr=np.sum(np.prod(np.abs(sol.x)**self.binding_syr.components,axis=1)*self.binding_syr.degeneracy/k[self.filter_diff]*dh[self.filter_syr])
        else:
            q_syr=0 # No reaction in syringe, so no heat change

        # Calculate total heat in the cell after each injection
        if self.binding.number_stages==1:
            # For a single binding stage in the cell
            q_list = self.V0*dh[self.filter]/2*(concentrations_for_solver.T[0]+concentrations_for_solver.T[1]+k-np.sqrt((concentrations_for_solver.T[0]+concentrations_for_solver.T[1]+k)**2-4*concentrations_for_solver.T[0]*concentrations_for_solver.T[1]))

        else:
            # For multiple binding stages in the cell, solve for free concentrations
            for i in range(len(q_list)):

                if i==0:
                    sol = root(self.binding.solve_for_free_concentrations,x0=0.5*concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})
                    if np.all(self.in_syringe):
                        q_list[i]=0. # If all components are in the syringe, initial heat is 0
                        continue



                else:
                    sol = root(self.binding.solve_for_free_concentrations,x0=np.abs(sol.x),args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})

                # check for convergence and try different methods if needed
                if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                    sol = root(self.binding.solve_for_free_concentrations,x0=0.5*concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='hybr',options={'xtol':1e-10})
                    if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                        sol = root(self.binding.solve_for_free_concentrations,x0=0.5*concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='lm',options={'xtol':1e-10})
                        if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                            sol = root(self.binding.solve_for_free_concentrations,x0=0.5*concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='lm',options={'xtol':1e-10})
                            if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                                sol = root(self.binding.solve_for_free_concentrations,x0=0.5*concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='df-sane',options={'ftol':1e-15})


                # Calculate total heat from bound species concentrations
                q_list[i]=self.V0*np.sum(np.prod(np.abs(sol.x)**self.binding.components,axis=1)*self.binding.degeneracy/k*dh[self.filter])

        # Calculate differential heat per injection
        delta_q = q_list[1:] - (1 - self.inj_list / self.V0)*q_list[:-1]

        # Calculate final dq values including heat of injection and unit conversion
        dq_list = dh_0+ 1e9*(delta_q-q_syr*self.inj_list) ##unit conversion from kcal to ucal (dh_0 in ucal already)

        return dq_list

    def get_conc(self, dg, dh, total_concentrations, dh_0):
        """
        Calculates the concentrations of free and bound species after each injection.

        Args:
            dg (ndarray): Standard Gibbs free energies of binding.
            dh (ndarray): Standard enthalpies of binding.
            total_concentrations (ndarray): Total concentrations of components.
            dh_0 (float): Differential heat of injection before binding.

        Returns:
            ndarray: Array of concentrations of all species after each injection.
        """

        k = np.exp(dg/(R*self.Temp))[self.filter] # Calculate association binding constants for relevant stages

        concentrations_for_solver = total_concentrations*self.conc_scaling.T # Concentrations after each injection
        fc_list = np.empty(shape=concentrations_for_solver.shape) # Initialize array for free concentrations

        # Solve for free concentrations after each injection
        for i in range(len(fc_list)):

            if i==0:
                sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})

            else:
                sol = root(self.binding.solve_for_free_concentrations,x0=np.abs(sol.x),args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})

            # check for convergence and try different methods if needed
            if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:

                sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='hybr',options={'xtol':1e-10})
                if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                    sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='lm',options={'xtol':1e-10})
                    if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                        sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='lm',options={'xtol':1e-10})
                        if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                            sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='df-sane',options={'ftol':1e-15})

            fc_list[i]=np.abs(sol.x) # Store free concentrations (ensure non-negative)

        # Calculate concentrations of bound species
        conc_binding = np.prod(fc_list[:, None, :] ** self.binding.components[None, :, :],axis=2)*self.binding.degeneracy/k

        # Return free and bound concentrations
        return np.concatenate((fc_list,conc_binding),axis=1)


    def get_prior(self, prior_shape, filtering,  prior_width=1, nuisance_bounds = np.array([[-50.0, 50.0],[0.001, 10.0]]), jeffreys_sigma=False, h0_auto=True):
        """
        Creates a prior distribution object for the isotherm parameters.

        Args:
            prior_shape (str): Shape of the prior distribution.
            filtering (bool): Enable filtering based on concentration ratio.
            prior_width (float, optional): Prior width. Defaults to 1.
            nuisance_bounds (ndarray, optional): Bounds for nuisance parameters. Defaults to np.array([[-50.0, 50.0],[0.001, 10.0]]).
            jeffreys_sigma (bool, optional): Use Jeffreys prior for sigma. Defaults to False.
            h0_auto (bool, optional): Automatically determine h0 bound. Defaults to True.

        Returns:
            prior_isotherm: A prior_isotherm distribution object.
        """
        if (self.filter_prior is not None) and filtering:
            # Use filtering if filter_prior is available and filtering is enabled
            return prior_isotherm(iso=self, prior_shape=prior_shape, filtering=filtering, prior_width=prior_width, nuisance_bounds=nuisance_bounds, jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto)

        else:
            # Otherwise, create a prior without filtering
            return prior_isotherm(iso=self, prior_shape=prior_shape, filtering=False, prior_width=prior_width, nuisance_bounds=nuisance_bounds, jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto)

    def plot(self, dg, dh, total_concentrations, dh_0):
        """
        Plots the experimental and predicted isotherm data.

        Args:
            dg (ndarray): Standard Gibbs free energies of binding.
            dh (ndarray): Standard enthalpies of binding.
            total_concentrations (ndarray): Total concentrations of components.
            dh_0 (float): Differential heat of injection before binding.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """

        fig, ax = plt.subplots()
        # Calculate true dq values based on provided parameters
        dq_true=self.get_dq_list(dg, dh, total_concentrations, dh_0)
        # Plot experimental dq values
        ax.plot(self.dq_list, ls='None', color='black', marker='o')
        # Highlight skipped injections in red
        ax.plot([x - 1 for x in self.skipped_injections],self.dq_list[[x - 1 for x in self.skipped_injections]], ls='None', color='r', marker='o')
        # Plot predicted dq values
        ax.plot(dq_true, color='black')
        # Set y-axis limits
        ax.set_ylim(min(self.dq_list)-5,max(self.dq_list)+5)

        return fig # Return the figure object