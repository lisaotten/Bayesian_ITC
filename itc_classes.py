import numpy as np
from scipy.optimize import root
from scipy import stats
import pocomc as pc
import corner
import matplotlib.pyplot as plt
import yaml
import os
import torch
import shutil
import corner
import re

R = 0.001987 

class inverse_x_distribution:
    def __init__(self, a, b):
        if a <= 0 or b <= 0 or a >= b:
            raise ValueError("Require 0 < a < b.")
        self.a = a
        self.b = b
        self.log_norm = np.log(np.log(b / a))  # Normalization constant in log-space

    def logpdf(self, x):
        """Log-probability density function."""
        if np.isscalar(x):  # Handle scalar input
            if x < self.a or x > self.b:
                return -np.inf
            return -np.log(x) - self.log_norm
        else:  # Handle vectorized input
            x = np.asarray(x)
            log_pdf = -np.log(x) - self.log_norm
            log_pdf[(x < self.a) | (x > self.b)] = -np.inf
            return log_pdf
    def rvs(self, size=1, random_state=None):
        """Generate random samples."""
        rng = np.random.default_rng(random_state)
        u = rng.uniform(0, 1, size)
        return self.a * (self.b / self.a) ** u


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
        
    def __init__(self, iso, prior_shape, filtering, prior_width=1, nuisance_bounds=np.array([[-50.0, 50.0],[0.001, 10.0]]), jeffreys_sigma=False, h0_auto=True):
        
        self.iso=iso
        self.dim = int(self.iso.number_components + 2)
        self.prior_shape = prior_shape
        self.prior_width = prior_width
      
        if h0_auto:
            self.bounds = np.array([[0 , np.inf]]*self.iso.number_components + [[self.iso.dq_list[self.iso.skipped_mask][0],2*self.iso.dq_list[self.iso.skipped_mask][-1]-self.iso.dq_list[self.iso.skipped_mask][0]]]+[list(nuisance_bounds[1])])
        else:
            self.bounds = np.array([[0 , np.inf]]*self.iso.number_components + list(nuisance_bounds))
        
        self.prior_nuisance_list=[]
        self.prior_nuisance_list.append(stats.uniform(loc=self.bounds[self.iso.number_components][0], scale=self.bounds[self.iso.number_components][1]-self.bounds[self.iso.number_components][0]))

        if jeffreys_sigma:
            self.prior_nuisance_list.append(inverse_x_distribution(a=nuisance_bounds[1,0], b=nuisance_bounds[1,1]))

        else:
            self.prior_nuisance_list.append(stats.uniform(loc=nuisance_bounds[1,0], scale=nuisance_bounds[1,1]-nuisance_bounds[1,0]))
        
        if filtering:
            self.filtering = True
            self.filter_ind_c1= next(i for i, x in enumerate(self.iso.filter_prior) if x>0 and self.iso.in_syringe[i])
            self.filter_ind_c2= next(i for i, x in enumerate(self.iso.filter_prior) if x>0 and not self.iso.in_syringe[i])
            self.factor= self.iso.filter_prior[self.filter_ind_c1]/self.iso.filter_prior[self.filter_ind_c2]
            
        else:
            self.filtering = False
                   
    def bounds_plot(self, prior_width=None):
        if prior_width is None:
            prior_width=self.prior_width
            
        prior_width=list(np.atleast_1d(prior_width))
        
        if len(prior_width)==1:
            prior_width=prior_width*self.iso.number_components
            
        return np.array([[est*max(1-2*prior_width[i],0),est*(1+2*prior_width[i])] for i,est in enumerate(self.iso.concentration_estimate)] + list(self.bounds[-2:]))

    def return_simple_priors(self, prior_width):
        
        prior_width = np.atleast_2d(prior_width)
        if prior_width.shape[0] == 1:
            prior_width = np.repeat(prior_width, self.iso.number_components, axis=0)
            
        concentration = np.array(self.iso.concentration_estimate)[:, None] * np.ones_like(prior_width)

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
            
            concentration = np.array(self.iso.concentration_estimate)[:, None] * np.ones_like(prior_width)
            
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
                
        return_value+=sum(dist.logpdf(x.T[i+self.iso.number_components]) for i, dist in enumerate(self.prior_nuisance_list))

        if self.filtering:
            h0 = np.atleast_1d(x.T[-2])
            ind_inflection=np.argmin(np.abs(self.iso.dq_list[self.iso.skipped_mask][:,np.newaxis]-(h0+self.iso.dq_list[self.iso.skipped_mask][0])/2),axis=0)
            vol_inflection=np.array([sum(entry) for entry in [self.iso.inj_list[:i] for i in np.argmin(np.abs(self.iso.dq_list-self.iso.dq_list[self.iso.skipped_mask][ind_inflection][:, np.newaxis]), axis=1)]])
            ratio=x.T[self.filter_ind_c1]*vol_inflection/x.T[self.filter_ind_c2]/self.iso.V0
            return_value+=np.array(list(map(self.ratio_func, ratio))) 
            
        return np.squeeze(return_value) if shape_return else return_value
    
    def rvs(self, size=1, prior_width=None):
        
        if prior_width is None:
            prior_width = np.array([self.prior_width]*size).T    

        prior_list=np.array(self.return_simple_priors(prior_width))

        return_vector = np.array([list(map(lambda d: d.rvs(size=1)[0], dist)) for dist in prior_list]+[dist.rvs(size=size) for dist in self.prior_nuisance_list])
        
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

class prior_combined:
        
    def __init__(self, prior_binding_structure, prior_isotherms, cumulative_indices, prior_width_bounds=[]):

        self.bounds = np.array(list(prior_binding_structure.bounds) + [bound for prior in prior_isotherms for bound in prior.bounds] + [bound for bound in prior_width_bounds])
        self.dim = prior_binding_structure.dim + sum(prior.dim for prior in prior_isotherms) + len(prior_width_bounds)
        self.cumulative_indices = cumulative_indices

        self.prior_prior_width=[stats.uniform(loc=bound[0], scale=bound[1]-bound[0]) for bound in prior_width_bounds]
        self.dim_prior_prior_width=len(prior_width_bounds)
        
        self.prior_binding_structure = prior_binding_structure
        self.prior_isotherms = prior_isotherms
        
    def bounds_plot(self,prior_width=None):
     
        return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for prior in self.prior_isotherms for bound in prior.bounds_plot(prior_width)]
        
        if self.dim_prior_prior_width>0:
            if self.dim_prior_prior_width>1:
                name_to_width = dict(zip(self.prior_binding_structure.binding.names_components, prior_width))
                return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for prior in self.prior_isotherms for bound in prior.bounds_plot(np.array([name_to_width[name] for name in prior.iso.names_components]))]
            else:
                return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for prior in self.prior_isotherms for bound in prior.bounds_plot(prior_width)]
            return_bounds+=list(self.bounds[-self.dim_prior_prior_width:])
        else:
            return_bounds=list(self.prior_binding_structure.bounds_plot) + [bound for prior in self.prior_isotherms for bound in prior.bounds_plot(prior_width)]
  
        return np.array(return_bounds) 
        
    def logpdf(self, x):

        return_vector=self.prior_binding_structure.logpdf(x.T[:self.cumulative_indices[1]].T)

        if self.dim_prior_prior_width>0:
            if self.dim_prior_prior_width>1:
  
                name_to_width = dict(zip(self.prior_binding_structure.binding.names_components, x.T[self.cumulative_indices[-1]:]))
                
                return_vector+=sum(prior.logpdf(x.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T,np.array([name_to_width[name] for name in prior.iso.names_components])[:,None] if x.ndim==1 else np.array([name_to_width[name] for name in prior.iso.names_components])) for i, prior in enumerate(self.prior_isotherms))
            else:
                return_vector+=sum(prior.logpdf(x.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T,x.T[self.cumulative_indices[-1]]) for i, prior in enumerate(self.prior_isotherms))
                
            return_vector+=sum(prior.logpdf(x.T[self.cumulative_indices[-1]+i].T) for i, prior in enumerate(self.prior_prior_width))
 
        else:
            return_vector+=sum(prior.logpdf(x.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T) for i, prior in enumerate(self.prior_isotherms))
    
        return return_vector
    
    def rvs(self, size=1):
        
        return_vector_prior_width=[prior.rvs(size) for prior in self.prior_prior_width]
        
        return_vector=list(self.prior_binding_structure.rvs(size).T)
        
        if self.dim_prior_prior_width>0:
            if self.dim_prior_prior_width>1:
                name_to_width = dict(zip(self.prior_binding_structure.binding.names_components, return_vector_prior_width))
                for prior in self.prior_isotherms:
                    return_vector+= list(prior.rvs(size, np.array([name_to_width[name] for name in prior.iso.names_components])).T) 
            else:
                for prior in self.prior_isotherms:
                    return_vector+= list(prior.rvs(size, return_vector_prior_width[0]).T) 
        else:       
            for prior in self.prior_isotherms:
                    return_vector+= list(prior.rvs(size).T) 
                
        return np.array(return_vector+return_vector_prior_width).T

class prior_thermo:
        
    def __init__(self, binding, bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior, posterior_indices=None, posterior_data=None):
        
        self.binding=binding
        self.bounds = np.copy(bounds)
        self.bounds_plot = np.copy(bounds)
        self.dim = len(self.bounds)
        self.dimv= int(len(self.bounds)/2)
        self.dd_combinations = np.array(dd_combinations,dtype=int)
        
        if len(ddg_bounds) == 1:
            self.ddg_bounds = np.array(list(ddg_bounds)*len(self.dd_combinations))
        else:
            self.ddg_bounds = np.array(ddg_bounds)
            

        if len(ddh_bounds) == 1:
            self.ddh_bounds = np.array(list(ddh_bounds)*len(self.dd_combinations))
        else:
            self.ddh_bounds = np.array(ddh_bounds)

        self.last_non_zero_indices_dd = [max([i for i, x in enumerate(sublist) if x != 0]) for sublist in self.dd_combinations]

        if posterior:
            self.posterior=True
            self.posterior_dist=stats.gaussian_kde(posterior_data.T)
            self.posterior_indices=posterior_indices

        else:
            self.posterior=False
            self.posterior_indices=[]
        
    def logpdf(self, x):

        return_vector=sum(stats.uniform.logpdf(x.T[i],loc=self.bounds[i,0],scale=self.bounds[i,1]-self.bounds[i,0]) for i in range(self.dim) if i not in self.posterior_indices)

        if self.posterior:
            temp=self.posterior_dist.logpdf(x.T[self.posterior_indices])
            if len(temp)==1:
                temp=temp[0]
            return_vector+=temp
            
        for i, combination in enumerate(self.dd_combinations):
            sum_total=sum(combination[k]*x.T[k] for k in range(len(combination)))
            return_vector+=stats.uniform.logpdf(sum_total,loc=self.ddg_bounds[i,0],scale=self.ddg_bounds[i,1]-self.ddg_bounds[i,0])
            sum_total=sum(combination[k]*x.T[k+self.dimv] for k in range(len(combination)))
            return_vector+=stats.uniform.logpdf(sum_total,loc=self.ddh_bounds[i,0],scale=self.ddh_bounds[i,1]-self.ddh_bounds[i,0])
            
        return return_vector
    
    def rvs(self, size=1):
        
        return_vector=[]
        
        if self.posterior:
            ind_random=np.random.choice(range(len(self.posterior_dist.dataset.T)),size)
            
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
    
    def __init__(self, binding, list_isotherms, path, prior_prior_width=0):
        
        self.binding=binding
        self.list_isotherms=list_isotherms
        self.number_isotherms=len(list_isotherms)
        self.cumulative_indices = np.cumsum([self.binding.number_stages]*2+[iso.number_components+2 for iso in self.list_isotherms])
        
        self.n_dim = self.cumulative_indices[-1]+prior_prior_width
        
        self.labels = np.concatenate(('g'+self.binding.binding_states,'h'+self.binding.binding_states))
        for i, isotherm in enumerate(self.list_isotherms):
            self.labels=np.concatenate((self.labels,'iso'+str(i+1)+'_'+isotherm.names_components+np.where(isotherm.in_syringe, '_s', '_c'),['iso'+str(i+1)+'_dh0'],['iso'+str(i+1)+'_sigma']))
        if prior_prior_width==1:
            self.labels=np.concatenate((self.labels,['prior_width']))
        elif prior_prior_width>1:
            for i in range(prior_prior_width):
                self.labels=np.concatenate((self.labels,['prior_width_'+self.binding.names_components[i]]))
            
        if not os.path.exists(path):
            os.makedirs(path)
        self.path=path

        if not os.path.exists(self.path + 'figures/'):
            os.makedirs(self.path + 'figures/')

        if not os.path.exists(self.path + 'data/'):
            os.makedirs(self.path + 'data/')
    
    def log_like(self, parameters):
        
        logl=0
        for i, iso in enumerate(self.list_isotherms):
            dq_list_calc=iso.get_dq_list(dg = parameters[:self.cumulative_indices[0]], 
                                         dh = parameters[self.cumulative_indices[0]:self.cumulative_indices[1]], 
                                         total_concentrations = parameters[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][:iso.number_components],
                                         dh_0 = parameters[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components])
            
            sigma = parameters[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components+1]
            
            # Calculate the difference
            diff = dq_list_calc[self.list_isotherms[i].skipped_mask] - self.list_isotherms[i].dq_list[self.list_isotherms[i].skipped_mask]
            # Update log-likelihood
            logl += -len(diff) * np.log(np.sqrt(2*np.pi)*sigma)
            logl += -0.5 * np.dot(diff, diff)/ sigma**2
            
        return logl

    def get_prior(self, bounds, dd_combinations, ddg_bounds, ddh_bounds,  posterior, prior_shape, filtering, prior_width=1, nuisance_bounds = np.array([[-50.0, 50.0],[0.001, 10.0]]), posterior_indices=None, posterior_data=None, jeffreys_sigma=False,  h0_auto=True, prior_width_bounds=[]):

        prior_binding=self.binding.get_prior(bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior, posterior_indices, posterior_data)
        prior_isotherms=[iso.get_prior(prior_shape=prior_shape, filtering=filtering,  prior_width=prior_width, nuisance_bounds = nuisance_bounds, jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto) for iso in self.list_isotherms]
        
        return prior_combined(prior_binding, prior_isotherms, self.cumulative_indices, prior_width_bounds)


    def run(self, prior, n_effective, n_total, n_cpus=1, continue_run=False):

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
            
        if continue_run:
        
            if os.path.exists(os.path.join(self.path, "pmc_final.state")):

                print('Run already complete')
                sampler.run(n_total=n_total,n_evidence=n_total,save_every=10, resume_state_path = os.path.join(self.path, "pmc_final.state"))
                
            else:
                pmc_files = []
                
                # Iterate through files in the folder
                for file in os.listdir(self.path):
                    match = re.match(r"pmc_(\d+)\.state$", file)
                    if match:
                        pmc_files.append((int(match.group(1)), file))
                        
                if not pmc_files:
                    sampler.run(n_total=n_total,n_evidence=n_total,save_every=10)
                    
                else:
                    # Find the file with the highest number
                    path_continue=max(pmc_files, key=lambda x: x[0])[1]
 
                    sampler.run(n_total=n_total,n_evidence=n_total,save_every=10, resume_state_path = os.path.join(self.path, path_continue))
            
        else:
            sampler.run(n_total=n_total,n_evidence=n_total,save_every=10)
            
        results = sampler.results
        samples, weights, logl, logp = sampler.posterior()
        samples_reweighted, logl_reweighted, logp_reweighted = sampler.posterior(resample=True)

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

        medoid=geometric_medoid(samples)
        
        dg = medoid[:self.cumulative_indices[0]], 
        dh = medoid[self.cumulative_indices[0]:self.cumulative_indices[1]], 
        
        for i,iso in enumerate(self.list_isotherms):
            
            total_concentrations = medoid[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][:iso.number_components],
            dh_0 = medoid[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components]
            sigma = medoid[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]][iso.number_components+1]
            
            iso_syn=self.binding.synthetic_isotherm(dg, dh, total_concentrations, dh_0, sigma, iso.names_components, iso.concentration_estimate, iso.in_syringe, iso.filter_prior, iso.inj_list[0]*1e6, iso.inj_list[-1]*1e6, len(iso.inj_list)+1, iso.Temp , iso.V0)
            iso_syn.save(path, 'iso'+str(i+1))
    
    def compute_dd_samples(self, samples, dd_combinations, labels=None):
   
        N, total_dim = samples.shape
        k, d = dd_combinations.shape
    
        dd_samples = np.zeros((N, 2 * k))
        
        dd_labels = np.zeros(k, dtype='U100')  
        
        for i in range(k):
           
            dd_samples[:, i] = np.dot(samples[:, :d], dd_combinations[i])
    
            dd_samples[:, k + i] = np.dot(samples[:, d:2*d], dd_combinations[i])

            dd_labels[i] = f"comb({str(dd_combinations[i])})"
            
        dd_labels=np.concatenate([('g' + dd_labels), ('h' + dd_labels)]) if labels is None else np.concatenate([('g' + labels), ('h' + labels)])
        
        return dd_samples, dd_labels
    

    def evaluate_prior_dependence(self, prior_shape, filtering, samples, prior_width_original, prior_width_min, bounds_plot, jeffreys_sigma=False, h0_auto=True, prior_width_max=None, truths=None, logz=0., name='prior_dependence', nuisance_bounds = np.array([[-50.0, 50.0],[0.001, 10.0]])):

        if prior_width_max is None:
            prior_width_max = prior_width_original
        
        prior_isotherms_original=[iso.get_prior(prior_width=prior_width_original, prior_shape=prior_shape, filtering=filtering, nuisance_bounds=nuisance_bounds,jeffreys_sigma=jeffreys_sigma,h0_auto=h0_auto) for iso in self.list_isotherms]
            
        widtharray=np.linspace(prior_width_min,prior_width_max,50)
        logzarray=[None]*len(widtharray)
        percentiles=[None]*len(widtharray)
        weights=[None]*len(widtharray)

        diff0=-sum(prior.logpdf(samples.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T) for i, prior in enumerate(prior_isotherms_original))
       
        for k,w in enumerate(widtharray):
            prior_isotherms_new = [iso.get_prior(prior_width=w, prior_shape=prior_shape, filtering=filtering, nuisance_bounds=nuisance_bounds,jeffreys_sigma=jeffreys_sigma,h0_auto=h0_auto) for iso in self.list_isotherms]
            diff=diff0+sum(prior.logpdf(samples.T[self.cumulative_indices[1+i]:self.cumulative_indices[2+i]].T) for i, prior in enumerate(prior_isotherms_new))
            logzarray[k]=logz+np.logaddexp.reduce(diff)-np.log(len(samples))
            weights[k]=np.exp(diff)
            print(logzarray[k],weights[k])
            percentiles[k]=np.percentile(samples,(2.5,25,50,75,97.5),axis=0,weights=weights[k], method='inverted_cdf')
          
        percentiles=np.array(percentiles)
        
        np.savetxt(os.path.join(self.path + 'data/', 'widtharray.csv'), widtharray, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', 'logzarray.csv'), logzarray, delimiter=',')
        np.savetxt(os.path.join(self.path + 'data/', 'weightsarray.csv'), weights, delimiter=',')
        
        figures=[]
        
     
        fig, ax = plt.subplots()
        ax.plot(widtharray,logzarray,color='k')
        ax.set_ylim(logzarray[-1]*1.1-np.max(logzarray)*0.1,np.max(logzarray)*1.1-0.1*logzarray[-1])
        ax.plot(prior_width_original, logz, 'o', color='k')
        plt.savefig(os.path.join(self.path +'figures/',name+'_logz.png'))
        plt.close(fig)

        figures.append(fig)

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

        if labels is None:
            labels=self.labels
            
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

        fig=corner.corner(samples[0][:,indices],color=colors[0],weights=weights[0],range=bounds_plot[indices],truth_color='k',truths=truths[indices], bins=80,labels=labels[indices],plot_datapoints=False,hist_kwargs=dict(density=True),plot_density=True)
        for i in range(1,len(samples)):
            corner.corner(samples[i][:,indices],fig=fig,color=colors[i],weights=weights[i],range=bounds_plot[indices],truth_color='k',truths=truths[indices],bins=80,plot_datapoints=False,hist_kwargs=dict(density=True),plot_density=True)
        
        plt.savefig(os.path.join(self.path +'figures/',name+'.png'))
        plt.close(fig)
        return fig

    def plot_histograms(self, samples, bounds_plot, name='hist', weights=None,truths=None, labels=None,colors=['k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']):

        if labels is None:
            labels=self.labels
   
        if not isinstance(samples, list):
            samples = [samples]
        if weights is None:
            weights = [None] * len(samples)
        elif not isinstance(weights, list):
            weights = [weights] * len(samples)
            
        figures = []
    
        for i in range(len(samples[0].T)):

            fig, ax = plt.subplots()
            for j in range(len(samples)):
                ax.hist(samples[j][:,i], bins=80, range=bounds_plot[i],weights=weights[j], histtype='step', color=colors[j])
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

    def plot_isotherms(self, samples, name='iso', colors=['grey','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']):
      
        if not isinstance(samples, list):
            samples = [samples]
            
        figures = []    
        inds = np.random.randint(list(map(len,samples)), size=(1000,len(samples)))
        
        for j, iso in enumerate(self.list_isotherms):
            fig, ax = plt.subplots()
            for k in range(len(samples)):
                for i, ind in enumerate(inds[:,k]):
                    parameters = samples[k][ind]
            
                    y_pred_i = iso.get_dq_list(dg = parameters[:self.cumulative_indices[0]], 
                                                 dh = parameters[self.cumulative_indices[0]:self.cumulative_indices[1]], 
                                                 total_concentrations = parameters[self.cumulative_indices[1+j]:self.cumulative_indices[2+j]][:iso.number_components],
                                                 dh_0 = parameters[self.cumulative_indices[1+j]:self.cumulative_indices[2+j]][iso.number_components])
                                    
                    ax.plot(y_pred_i, alpha=0.01, color=colors[k])
                    
            ax.plot(iso.dq_list, ls='None', color='black', marker='o')
            ax.plot([x - 1 for x in iso.skipped_injections],iso.dq_list[[x - 1 for x in iso.skipped_injections]], ls='None', color='r', marker='o')
            ax.set_ylim(min(iso.dq_list)-5,max(iso.dq_list)+5)
            ax.set_title('iso'+str(j+1)+' '+' '.join(iso.names_components+np.where(iso.in_syringe, '_s', '_c')))
                
            plt.savefig(os.path.join(self.path +'figures/',name+'_'+str(j+1)+'.png'))
            plt.close(fig)  
    
            figures.append(fig)

        return figures
        
class binding_structure:
    
    def __init__(self, components, names_components, degeneracy, binding_states):
        
        self.components=np.copy(components) 
        self.names_components=names_components
        self.degeneracy=np.copy(degeneracy)
        self.binding_states=np.copy(binding_states)
        
        self.number_stages=len(components)
        self.number_components=len(components.T)
   
    def solve_for_free_concentrations(self,free_concentrations,total_concentrations,k):
        
        return total_concentrations - np.abs(free_concentrations) - np.sum(np.prod(np.abs(free_concentrations)**self.components,axis=1)*self.degeneracy/k*self.components.T,axis=1)

    def jacobian_solve_for_free_concentrations(self, free_concentrations, total_concentrations, k):

        free_concentrations=np.abs(free_concentrations)+1e-15
        common = np.prod((free_concentrations)**self.components,axis=1)*self.degeneracy/k*self.components.T
        jacobian = np.eye(len(free_concentrations))-np.sum(common[:, None, :] * self.components.T[None, :, :], axis=2) / (free_concentrations[None, :])
    
        return jacobian
        
    def min_solve_for_free_concentrations(self, free_concentrations, total_concentrations, k):
        return np.sum(self.solve_for_free_concentrations(free_concentrations,total_concentrations,k)**2)

    def jacobian_min_solve_for_free_concentrations(self, free_concentrations, total_concentrations, k):
        return np.sum(2*self.solve_for_free_concentrations(free_concentrations,total_concentrations,k)*self.jacobian_solve_for_free_concentrations(free_concentrations, total_concentrations, k),axis=0)
        
    def get_prior(self, bounds, dd_combinations, ddg_bounds, ddh_bounds, posterior, posterior_indices=None, posterior_data=None):

        return prior_thermo(self, bounds, dd_combinations, ddg_bounds, ddh_bounds,  posterior, posterior_indices, posterior_data)

    def reduce(self, names_components):
        
        iso_indexes_for_filter = [self.names_components.index(name) for name in names_components]
        non_iso_indexes = [i for i, name in enumerate(self.names_components) if name not in names_components]
        filter=(self.components[:, non_iso_indexes] == 0).all(axis=1)

        return binding_structure(self.components[filter][:,iso_indexes_for_filter], names_components, self.degeneracy[filter], binding_states=self.binding_states[filter]), filter

    def synthetic_isotherm(self, dg, dh, total_concentrations, dh_0, sigma, names_components_syn, concentration_estimate_syn, in_syringe_syn, filter_prior_syn=None, first_inj_vol= 2, inj_vol = 10, inj_count = 35, Temp=298.15 , V0 = 1.42e-3, show_concs=False, colors= [
                                                                                                                                                                                                                                                                    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                                                                                                                                                                                                                                                                    'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan', 
                                                                                                                                                                                                                                                                    'darkblue', 'gold', 'lime', 'crimson', 'indigo', 
                                                                                                                                                                                                                                                                    'maroon', 'teal', 'turquoise', 'lightgreen', 'darkorange', 
                                                                                                                                                                                                                                                                    'navy', 'magenta', 'salmon', 'orchid', 'darkgrey'
                                                                                                                                                                                                                                                                    ]):

        inj_list_syn = [first_inj_vol*1e-6] + [inj_vol*1e-6]*inj_count
        
        skipped_injections_syn = [1] 
        
        iso_indexes = [i for i, name in enumerate(self.names_components) if name in names_components_syn]

        truths_syn=list(dg)+list(dh)+list(total_concentrations[iso_indexes])+[dh_0]+[sigma]
        
        if filter_prior_syn is None:
            filter_prior=None
        else:
            filter_prior=filter_prior_syn[iso_indexes]
            
        isotherm_syn=isotherm(self, inj_list_syn=inj_list_syn, skipped_injections_syn=skipped_injections_syn, concentration_estimate_syn=concentration_estimate_syn[iso_indexes], in_syringe_syn=in_syringe_syn[iso_indexes], filter_prior_syn=filter_prior, names_components_syn=names_components_syn,  truths_syn= truths_syn, Temp=Temp , V0=V0)
        fig=isotherm_syn.plot(dg, dh, total_concentrations[iso_indexes], dh_0)
        fig.show()
        
        if show_concs:
            concs = isotherm_syn.get_conc(dg, dh, total_concentrations[iso_indexes], dh_0)
            labels = np.concatenate((isotherm_syn.binding.names_components, isotherm_syn.binding.binding_states))
            
            # Create a single figure with two subplots
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
            
            # Adjust layout
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the bottom for the legend      
            plt.show()
       
        return isotherm_syn

class isotherm:
    
    def __init__(self, binding, filepath=None, inj_list_syn=None, skipped_injections_syn=None, concentration_estimate_syn=None, in_syringe_syn=None, filter_prior_syn=None, names_components_syn=None, truths_syn=None, Temp=298.15 , V0 = 1.42e-3):

        self.V0 = V0
        self.Temp = Temp
        
        if filepath is not None:
            
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
                        self.Temp = 273.15 + float(values[1])
                            
                    elif len(values) == 2:
                        dq_list.append(float(values[0]))
                        inj_list.append(float(values[1])*1e-6)
    
            
            self.dq_list = np.array(dq_list) 
            self.inj_list = np.array(inj_list)
            self.skipped_injections = np.array(skipped_injections)
            self.concentration_estimate = np.array(concentration_estimate)
            self.in_syringe = np.array(in_syringe)
            self.names_components = np.array(names_components)
    
            if len(filter_prior):
                self.filter_prior = np.array(filter_prior)
    
            else:
                self.filter_prior = None

            if len(truths):
                self.truths = np.array(truths) 
    
            else:
                self.truths = None

        else:
            self.dq_list = np.array([]) 
            self.inj_list = np.array(inj_list_syn)
            self.skipped_injections = np.array(skipped_injections_syn)
            self.concentration_estimate = np.array(concentration_estimate_syn)
            self.in_syringe = np.array(in_syringe_syn)
            self.names_components = np.array(names_components_syn)
            self.truths = np.array(truths_syn) 
    
            if filter_prior_syn is None:
                self.filter_prior = None
    
            else:
                self.filter_prior = np.array(filter_prior_syn)
            

        self.number_components = len(self.concentration_estimate)

        self.skipped_mask = np.isin(np.arange(len(self.inj_list)), self.skipped_injections-1, invert=True)

        self.conc_scaling = np.empty(shape=(self.number_components,len(self.inj_list)+1))
        self.conc_scaling[:,0] = ~self.in_syringe
        
        dcum = 1

        for i, inj in enumerate(self.inj_list):
            
            d  = 1 - (inj/self.V0)
            dcum *= d
            
            self.conc_scaling[:,i+1] = (1 - dcum) * (self.in_syringe) + dcum * (~self.in_syringe)

        self.binding, self.filter=binding.reduce(self.names_components)

        if filepath is None:

            truths_syn=np.array(truths_syn)
            self.truths = np.array([truths_syn[i] for i in range(len(truths_syn)) if (i < len(self.filter) and self.filter[i]) or (i < 2*len(self.filter) and i >= len(self.filter) and self.filter[i-len(self.filter)]) or i>=int(2*len(self.filter))])
            true_dq = self.get_dq_list(dg=truths_syn[:len(self.filter)],dh=truths_syn[len(self.filter):2*len(self.filter)], total_concentrations=truths_syn[2*len(self.filter):2*len(self.filter)+len(self.names_components)], dh_0=truths_syn[-2])
            dq_obs = true_dq + np.random.normal(loc=0,scale=truths_syn[-1],size=len(true_dq))

            self.dq_list=np.array(dq_obs)
        
    def save(self, path, name):
        
        if not os.path.exists(path):
            os.makedirs(path)
             
        with open(path+name, 'w') as synth_file:
            
            synth_file.write('components,' + ','.join(self.names_components))
            synth_file.write('\n' + 'concentration_estimate,' + ','.join(map(str,self.concentration_estimate)))
            synth_file.write('\n' + 'in_syringe,' + ','.join(map(str,self.in_syringe)))
            synth_file.write('\n' + 'T,' + str(self.Temp-273.15))
            if self.filter_prior is not None:
                synth_file.write('\n' + 'filter_prior,' + ','.join(map(str,self.filter_prior)))
            if self.truths is not None:
                synth_file.write('\n' + 'truths,' + ','.join(map(str,self.truths)))
                
            for i, dq in enumerate(self.dq_list):
                
                synth_file.write('\n' + str(dq) + ',' + str(self.inj_list[i]*1e6))
                
            synth_file.write('\n' + 'skipped_inj,' + ','.join(map(str,self.skipped_injections)))
        
    def get_dq_list(self, dg, dh, total_concentrations, dh_0):

        k = np.exp(dg/(R*self.Temp))[self.filter]
    
        concentrations_for_solver = total_concentrations*self.conc_scaling.T
        q_list = np.empty(len(concentrations_for_solver))
         
        if self.binding.number_stages==1:
            q_list = self.V0*dh[self.filter]/2*(concentrations_for_solver.T[0]+concentrations_for_solver.T[1]+k-np.sqrt((concentrations_for_solver.T[0]+concentrations_for_solver.T[1]+k)**2-4*concentrations_for_solver.T[0]*concentrations_for_solver.T[1]))
        
        else:
            for i in range(len(q_list)):
                        
                if i==0:
                    sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})
                
                else:
                    sol = root(self.binding.solve_for_free_concentrations,x0=np.abs(sol.x),args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})
                
                # check for convergence
                if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                    sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='hybr',options={'xtol':1e-10})
                    if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                        sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='lm',options={'xtol':1e-10})
                        if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                            sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='lm',options={'xtol':1e-10})
                            if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                                sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='df-sane',options={'ftol':1e-15})
                                
                                
                q_list[i]=self.V0*np.sum(np.prod(np.abs(sol.x)**self.binding.components,axis=1)*self.binding.degeneracy/k*dh[self.filter])
        
        delta_q = q_list[1:] - q_list[:-1]  # q_list[i+1] - q_list[i]
        avg_q = (q_list[1:] + q_list[:-1]) / 2  # (q_list[i+1] + q_list[i]) / 2
        dq_list = (delta_q + self.inj_list / self.V0 * avg_q) * 1e9 + dh_0 ##unit conversion from kcal to ucal (dh_0 in ucal already)
   
        return dq_list  
        
    def get_conc(self, dg, dh, total_concentrations, dh_0):

        k = np.exp(dg/(R*self.Temp))[self.filter]
    
        concentrations_for_solver = total_concentrations*self.conc_scaling.T
        fc_list = np.empty(shape=concentrations_for_solver.shape)
         
        for i in range(len(fc_list)):
                    
            if i==0:
                sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})
            
            else:
                sol = root(self.binding.solve_for_free_concentrations,x0=np.abs(sol.x),args=(concentrations_for_solver[i],k),method='hybr',options={'xtol':1e-4})
            
            # check for convergence
            if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                
                sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='hybr',options={'xtol':1e-10})
                if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                    sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='lm',options={'xtol':1e-10})
                    if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                        sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),jac=self.binding.jacobian_solve_for_free_concentrations,method='lm',options={'xtol':1e-10})
                        if np.sum(np.abs(self.binding.solve_for_free_concentrations(np.abs(sol.x),concentrations_for_solver[i],k)))/np.sum(np.abs(sol.x))>1e-3:
                            sol = root(self.binding.solve_for_free_concentrations,x0=concentrations_for_solver[i],args=(concentrations_for_solver[i],k),method='df-sane',options={'ftol':1e-15})
                            
                            
            fc_list[i]=np.abs(sol.x)
        
        conc_binding = np.prod(fc_list[:, None, :] ** self.binding.components[None, :, :],axis=2)*self.binding.degeneracy/k
        
        return np.concatenate((fc_list,conc_binding),axis=1) 
        
    def get_prior(self, prior_shape, filtering,  prior_width=1, nuisance_bounds = np.array([[-50.0, 50.0],[0.001, 10.0]]), jeffreys_sigma=False, h0_auto=True):
        if (self.filter_prior is not None) and filtering:
            return prior_isotherm(iso=self, prior_shape=prior_shape, filtering=filtering, prior_width=prior_width, nuisance_bounds=nuisance_bounds, jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto)
        
        else:
            return prior_isotherm(iso=self, prior_shape=prior_shape, filtering=False, prior_width=prior_width, nuisance_bounds=nuisance_bounds, jeffreys_sigma=jeffreys_sigma, h0_auto=h0_auto)
        
    def plot(self, dg, dh, total_concentrations, dh_0):
        
        fig, ax = plt.subplots()
        dq_true=self.get_dq_list(dg, dh, total_concentrations, dh_0)     
        ax.plot(self.dq_list, ls='None', color='black', marker='o')
        ax.plot([x - 1 for x in self.skipped_injections],self.dq_list[[x - 1 for x in self.skipped_injections]], ls='None', color='r', marker='o')
        ax.plot(dq_true, color='black')
        ax.set_ylim(min(self.dq_list)-5,max(self.dq_list)+5)

        return fig