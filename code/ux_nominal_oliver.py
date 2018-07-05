
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
from ux_nominal_oliver_data import sigma_eff, mean_ux, y_plus_dat, q_AJ_prior

try:
    xrange
except NameError:
    xrange = range


""""""""""""""""""""""""""""""""""" Parameters """""""""""""""""""""""""""""""""""""""
print('ok')
print(mean_ux)
print(q_AJ_prior)
print(sigma_eff
	)
# Import data from Matlab : post - process of the simulation



# Data imported from the Matlab file :  post-process of the simulation

"""""""""""""""""""""  Probability Function for epsilon_h/q_h  """""""""""""""""""""""""""

def lnprior_error_norm(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar, z, p = theta
	#u = (z * mean_q_data_i) / (h_i ** p)
	u = z / (h_i ** p)

	if p <= 0:
		return -inf

	return alpha*log(beta) + (alpha - 1)*log(p) - log(2*pi) - beta*p - 0.5*(((q_bar - q_0) / sigma_q ) ** 2) - 0.5*((u / sigma_C )**2) - log(special.gamma(alpha)) - log(sigma_C * sigma_q)

def lnlike_error_norm(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar, z, p = theta
	#u = (z * mean_q_data_i) / (h_i ** p)
	u = z / (h_i ** p)
	
	return -log(sqrt(2*pi) * sigma_n_i ) - 0.5*(((q_bar - mean_q_data_i - u * (h_i ** p)) / sigma_n_i) ** 2)


def lnprob_error_norm(theta):
	# The parameters are stored as a vector of values, so one shall unpack them
	q_bar, z ,p = theta

	lp = lnprior_error_norm(theta)
	if not isfinite(lp):
		return -inf

	return lp + lnlike_error_norm(theta) - log(((h_i) ** p))

with open("mean_discr_error_oliver_al.dat", "w") as out_file:	
	
	# Writing the header in the file
	out_string = "mean_discretization_error, mean_acceptance_fraction, discr_error_5, discr_error_95, autocorrelation_time_q, autocorrelation_time_z, autocorrelation_time_p"


	for i in range(0,3):
		
		sigma_n_i = sigma_eff[i]
		mean_q_data_i = mean_ux[i]
		

		# h_i on the nominal mesh
		
		h_i = 1.25e-01


		# Information on the prior from Del Alamo and Jimenez

		q_0 = q_AJ_prior[i + 1] 
		beta = 1 / 2
		alpha = 3
		sigma_q = 2.5 / 100 * q_0
		sigma_C = 4 * sigma_q


		""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

		"""""""""""""""""""""""""""""""""""  Emcee Procedure  """""""""""""""""""""""""""""""""""""""

		Ndim, Nwalkers = 3, 200

		# Choice 1 : Maximum likelihood for starting points
		fun_error = lambda *args: -lnlike_error_norm(*args)
		result = opt.minimize(fun_error, [mean_q_data_i, - 2, 4])
		p0_error_norm = [result['x']+1.e-4*random.randn(Ndim) for i in range(Nwalkers)]

				  
		# Initialize the sampler with the chosen specs.
		sampler_error_norm = emcee.EnsembleSampler(Nwalkers,Ndim,lnprob_error_norm)

		# Run 1000 steps as a burn-in.
		pos_error_norm, prob_error_norm, state_error_norm = sampler_error_norm.run_mcmc(p0_error_norm, 1000)

		# Reset the chain to remove the burn-in samples.
		sampler_error_norm.reset()

		# Starting from the final position in the burn-in chain, sample for 10000 steps.
		sampler_error_norm.run_mcmc(pos_error_norm, 50000, rstate0=state_error_norm)

		#figure2 = corner.corner(sampler_error_norm.flatchain[:,1], bins = 1000, smooth1d = 11, range = [[-1.0, 1.0]])
		#figure2.savefig("corner_test_oliver.png")
		# Compute the quantities of interest : mean of the discretization error, mean acceptance fraction
		# 5th and 95th percentile of the error and autocorrelation time

		mean_discr_error = median(sampler_error_norm.flatchain[:,1], axis=0)
		mean_acceptance_fraction = mean(sampler_error_norm.acceptance_fraction)
		discr_error_5, discr_error_95 = percentile(sampler_error_norm.chain[:,:,1], [5,95])
		[autocor_time_q, autocor_time_z, autocor_time_p] = sampler_error_norm.get_autocorr_time(c = 1)

		# Loop on the results to be written in the output file
		out_string += "\n"
		out_string += str(mean_discr_error) + ',' + str(mean_acceptance_fraction) + ',' + str(discr_error_5) + ',' + str(discr_error_95) + ',' + str(autocor_time_q) + ',' + str(autocor_time_z) + ',' + str(autocor_time_p)
		print("lol")
		print(discr_error_5, discr_error_95)
		print(discr_error_5 / mean_q_data_i , discr_error_95 / mean_q_data_i)
	# Write the complete results in the output file
	out_file.write(out_string)

		

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



