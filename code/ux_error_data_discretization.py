#!/usr/bin/env python
"""
Sample code for sampling from the posterior distribution with data from the nominal mesh only.
"""
from __future__ import division
from __future__ import print_function
from numpy import *
from scipy import special
import scipy.optimize as opt
import emcee
from matplotlib import pyplot as plt
import corner
from scipy.interpolate import UnivariateSpline
try:
    xrange
except NameError:
    xrange = range



""""""""""""""""""""""""""""""""""" Parameters """""""""""""""""""""""""""""""""""""""
# Import data from Matlab : post - process of the simulation :  for our data

y_plus_dat = loadtxt("y_plus.dat")
q_AJ_prior = loadtxt('prior_ux_our_data.dat')
mean_ux_nom = loadtxt('mean_ux_nom.dat', delimiter = ',')
mean_ux_coarsest = loadtxt('mean_ux_coarsest.dat', delimiter = ',')
mean_ux_finest = loadtxt('mean_ux_finest.dat', delimiter = ',')
sigma_eff_nom = loadtxt('sigma_eff_nom.dat', delimiter = ',')
sigma_eff_coarsest = loadtxt('sigma_eff_coarsest.dat', delimiter = ',')
sigma_eff_finest = loadtxt('sigma_eff_finest.dat', delimiter = ',')

# Import data from oliver simulation

#q_AJ_prior = loadtxt('prior_ux_oliver.dat', delimiter = ',')
#ux_nom = loadtxt('u_nom_oliver.dat', delimiter = ' ')
#ux_coarse = loadtxt('u_coarse_oliver.dat', delimiter = ' ')
#ux_coarsest = loadtxt('u_coarsest_oliver.dat', delimiter = ' ')

#print(q_AJ_prior, ux_nom, ux_coarse, ux_coarsest)

#mean_ux_nom = []
#mean_ux_coarse = []
#mean_ux_coarsest = []
#sigma_eff_nom = []
#sigma_eff_coarse = []
#sigma_eff_coarsest = []


#for i in range(31):
	#mean_ux_nom.append(ux_nom[i,2])
	#mean_ux_coarse.append(ux_coarse[i,2])
	#mean_ux_coarsest.append(ux_coarsest[i,2])
	#sigma_eff_nom.append(ux_nom[i,3])
	#sigma_eff_coarse.append(ux_coarse[i,3])
	#sigma_eff_coarsest.append(ux_coarsest[i,3])



# Fixed parameters

# Mesh size
		
h_i = 0.5
h_coarsest = 1.0 
h_coarse = sqrt(0.5)
h_finest = 0.5 * sqrt(0.5)

print(mean_ux_nom, mean_ux_coarsest, sigma_eff_nom, sigma_eff_coarsest, q_AJ_prior, sigma_eff_finest, mean_ux_finest)


"""""""""""""""""""""  Probability Function for epsilon_h/q_h  """""""""""""""""""""""""""

def lnprior_error_norm(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar, z, p = theta
	u = z / (h_i ** p)
	
	if p <= 0:
		return -inf

	return alpha*log(beta) + (alpha - 1)*log(p) - log(2*pi) - beta*p - 0.5*(((q_bar - q_0) / sigma_q ) ** 2) - 0.5*((u / sigma_C )**2) - special.gammaln(alpha) - log(sigma_C * sigma_q)


def lnlike_error_norm(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar, z, p = theta
	#u = (z * mean_q_data_i) / (h_i ** p)

	if p <= 0:
		return -inf
	
	return -log(2*pi * sigma_n_i * sigma_n_coarsest) - 0.5 * ((q_bar - mean_q_data_i - z ) ** 2 / (sigma_n_i ** 2)) \
		 - 0.5 * ((q_bar - mean_q_data_coarsest - (z) * ((h_coarsest/h_i) ** p) ) ** 2 / (sigma_n_coarsest ** 2)) 

def lnlike_error_norm_with_finest(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar, z, p = theta
	#u = (z * mean_q_data_i) / (h_i ** p)

	if p <= 0:
		return -inf
	
	return -log(sqrt(2*pi)*2*pi * sigma_n_i * sigma_n_coarsest * sigma_n_finest) - 0.5 * ((q_bar - mean_q_data_i - z ) ** 2 / (sigma_n_i ** 2)) \
		- 0.5 * ((q_bar - mean_q_data_coarsest - (z) * ((h_coarsest/h_i) ** p) ) ** 2 / (sigma_n_coarsest ** 2)) \
		- 0.5 * ((q_bar - mean_q_data_finest - (z) * ((h_finest/h_i) ** p) ) ** 2 / (sigma_n_finest ** 2)) 

def lnprob_error_norm(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar, z ,p = theta

	lp = lnprior_error_norm(theta)
	ll = lnlike_error_norm_with_finest(theta)
	if not isfinite(lp) or not isfinite(ll):
		return -inf

	return lp + ll - p * log(h_i)

with open("mean_discr_error.dat", "w") as out_file:	
	
	# Writing the header in the file
	out_string = "mean_discretization_error, mean_acceptance_fraction, discr_error_5, discr_error_95, autocorrelation_time_q, autocorrelation_time_z, autocorrelation_time_p"


	for i in range(0,len(sigma_eff_nom)):
	
		sigma_n_i = sigma_eff_nom[i]
		sigma_n_coarsest = sigma_eff_coarsest[i]
		#sigma_n_coarse = sigma_eff_coarse[i]
		sigma_n_finest = sigma_eff_finest[i]
		mean_q_data_i = mean_ux_nom[i]
		#mean_q_data_coarse = mean_ux_coarse[i]
		mean_q_data_coarsest = mean_ux_coarsest[i]
		mean_q_data_finest = mean_ux_finest[i]
		

		# Information on the prior from Del Alamo and Jimenez

		q_0 = q_AJ_prior[i] 
		beta = 1 / 2
		alpha = 3
		sigma_q = 2.5 / 100 * q_0
		sigma_C = 4 * sigma_q


		""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

		"""""""""""""""""""""""""""""""""""  Emcee Procedure  """""""""""""""""""""""""""""""""""""""

		Ndim, Nwalkers = 3, 100

		# Choice 1 : Maximum likelihood for starting points
		fun_error = lambda *args: -lnlike_error_norm(*args)
		result = opt.minimize(fun_error, [mean_q_data_i, -0.01, 4])
		p0_error_norm = [result['x']+1.e-4*random.randn(Ndim) for i in range(Nwalkers)]

				  
		# Initialize the sampler with the chosen specs.
		sampler_error_norm = emcee.EnsembleSampler(Nwalkers,Ndim,lnprob_error_norm, a = 3)

		# Run 1000 steps as a burn-in.
		pos_error_norm, prob_error_norm, state_error_norm = sampler_error_norm.run_mcmc(p0_error_norm, 10000)

		# Reset the chain to remove the burn-in samples.
		sampler_error_norm.reset()

		# Starting from the final position in the burn-in chain, sample for 10000 steps.
		sampler_error_norm.run_mcmc(pos_error_norm, 20000, rstate0=state_error_norm)

		# Compute the quantities of interest : mean of the discretization error, mean acceptance fraction
		# 5th and 95th percentile of the error and autocorrelation time

		mean_discr_error = median(sampler_error_norm.flatchain[:,1], axis=0)
		mean_discr_error = mean_discr_error / mean_q_data_i
		mean_acceptance_fraction = mean(sampler_error_norm.acceptance_fraction)
		discr_error_5, discr_error_95 = percentile(sampler_error_norm.chain[:,:,1], [5,95])
		discr_error_5, discr_error_95 = discr_error_5 / mean_q_data_i, discr_error_95 / mean_q_data_i 
		[autocor_time_q, autocor_time_z, autocor_time_p] = sampler_error_norm.get_autocorr_time(c = 1)

		# Loop on the results to be written in the output file
		out_string += "\n"
		out_string += str(mean_discr_error) + ',' + str(mean_acceptance_fraction) + ',' + str(discr_error_5) + ',' + str(discr_error_95) + ',' + str(autocor_time_q) + ',' + str(autocor_time_z) + ',' + str(autocor_time_p)
		print("lol")

	# Write the complete results in the output file
	out_file.write(out_string)

		

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


