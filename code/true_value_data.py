#!/usr/bin/env python
"""
Sample code for sampling from the posterior distribution with data from the nominal mesh only.
"""
from __future__ import division
from __future__ import print_function
from numpy import *
from scipy import special
import math as m
import scipy.optimize as opt
import emcee
from matplotlib import pyplot as plt
import matplotlib
import corner
from scipy.interpolate import UnivariateSpline, interp1d
from ar_test import sigma_eff
import scipy.stats as stats
import pandas as pd 
from matplotlib import rcParams


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
sigma_eff_nom = loadtxt('sigma_eff_nom.dat', delimiter = ',')
sigma_eff_coarsest = loadtxt('sigma_eff_coarsest.dat', delimiter = ',')


# Fixed parameters

# Mesh size
		
h_i = 0.5
h_coarsest = 1.0 
h_coarse = sqrt(0.5)

print(mean_ux_nom, mean_ux_coarsest, sigma_eff_nom, sigma_eff_coarsest, q_AJ_prior)



"""""""""""""""""""""""  Probability Function for q_bar, C , p  """""""""""""""""""""""""""


# Defining the prior

# Informative prior

def lnprior(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar,C,p = theta

	if p <= 0.0:
		return -inf

	return alpha * log(beta) + (alpha - 1) * log(p) - log(2 * pi) - beta * p - 0.5 * (((q_bar - q_0) / sigma_q ) ** 2)\
	- 0.5 * ((C / sigma_C ) **2) - log(m.gamma(alpha)) - log(sigma_C * sigma_q)


	# Defining the likelihood 

def lnlike(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar,C,p = theta

	if p <= 0.0:
		return - inf

	#return - log(sqrt(2*pi) * 2 * pi * sigma_n_i * sigma_n_coarse * sigma_n_coarsest) - 0.5 * (((q_bar - mean_q_data_i - C * (h_i ** p)) * (1.0/sigma_n_i)) ** 2) - 0.5 * (((q_bar - mean_q_data_coarse - C * (h_coarse ** p)) * (1.0/sigma_n_coarse)) ** 2) - 0.5 * (((q_bar - mean_q_data_coarsest - C * (h_coarsest ** p)) * (1.0/sigma_n_coarsest)) ** 2)
	return - log( 2 * pi * sigma_n_i * sigma_n_coarsest) \
	- 0.5 * (((q_bar - mean_q_data_i - C * (h_i ** p)) / (sigma_n_i)) ** 2) \
	- 0.5 * (((q_bar - mean_q_data_coarsest - C * (h_coarsest ** p)) /(sigma_n_coarsest)) ** 2) \


# Defining the posterior = prior * likelihood

def lnprob(theta):

	lp = lnprior(theta)
	ll = lnlike(theta)
	if not isfinite(lp) or not isfinite(ll):
		return -inf
	return lp + ll


with open("true_value_percentile_oliver.dat", "w") as out_file:	
	
	# Writing the header in the file
	out_string = "mean_true_value, mean_acceptance_fraction, true_value_5, true_value_25, true_value_75, true_value_95, autocorrelation_time_q, autocorrelation_time_z, autocorrelation_time_p"

	#for i in range(0,1):
	for i in range(0,len(sigma_eff_nom)):
	
		sigma_n_i = sigma_eff_nom[i]
		sigma_n_coarsest = sigma_eff_coarsest[i]
		#sigma_n_coarse = sigma_eff_coarse[i]
		mean_q_data_i = mean_ux_nom[i]
		#mean_q_data_coarse = mean_ux_coarse[i]
		mean_q_data_coarsest = mean_ux_coarsest[i]
		
		#sigma_n_i = 0.007636637330394
		#sigma_n_coarsest = 0.006476647806633
		#sigma_n_coarse = 0.00022051760857
		#mean_q_data_i = 0.204450528379045
		#mean_q_data_coarse = 0.37008279258
		#mean_q_data_coarsest = 0.195927154500829

		# Information on the prior from Del Alamo and Jimenez

		#q_0 = 0.202763919782201
		q_0 = q_AJ_prior[i] 
		beta = 1 / 2
		alpha = 3
		sigma_q = 2.5 / 100 * q_0
		sigma_C = 4 * sigma_q


		""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

		"""""""""""""""""""""""""""""""""""  Emcee Procedure  """""""""""""""""""""""""""""""""""""""

		Ndim, Nwalkers = 3, 100

		# Choice 1 : Maximum likelihood for starting points
		fun = lambda *args: -lnlike(*args)
		result = opt.minimize(fun, [mean_q_data_i, -0.01, 4])
		p0 = [result['x']+1.e-4*random.randn(Ndim) for i in range(Nwalkers)]

		# Initialize the sampler with the chosen specs.
		sampler = emcee.EnsembleSampler(Nwalkers,Ndim,lnprob, a = 3)

		# Run 1000 steps as a burn-in.
		pos, prob, state = sampler.run_mcmc(p0, 10000)

		# Reset the chain to remove the burn-in samples.
		sampler.reset()

		# Starting from the final position in the burn-in chain, sample for 10000 steps.
		sampler.run_mcmc(pos, 40000, rstate0=state)

		# Compute the quantities of interest : mean of the discretization error, mean acceptance fraction
		# 5th and 95th percentile of the error and autocorrelation time

		mean_true_value = median(sampler.flatchain[:,0], axis=0)
		mean_acceptance_fraction = mean(sampler.acceptance_fraction)
		true_value_5, true_value_25, true_value_75, true_value_95 = percentile(sampler.chain[:,:,0], [5,25,75,95])
		[autocor_time_q, autocor_time_z, autocor_time_p] = sampler.get_autocorr_time(c = 1)

		# Loop on the results to be written in the output file
		out_string += "\n"
		out_string += str(mean_true_value) + ',' + str(mean_acceptance_fraction) + ',' + str(true_value_5) + ',' + str(true_value_25) + ',' + str(true_value_75) + ',' + str(true_value_95) + ',' + str(autocor_time_q) + ',' + str(autocor_time_z) + ',' + str(autocor_time_p)
		print("lol")
		print(true_value_5, true_value_25, true_value_75, true_value_95)

	# Write the complete results in the output file
	out_file.write(out_string)




