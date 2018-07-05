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

mean_ux_nom = loadtxt('mean_ux_nom.dat', delimiter = ',')
sigma_eff_nom = loadtxt('sigma_eff_nom.dat', delimiter = ',')
print(mean_ux_nom, sigma_eff_nom)

#ux_nom = loadtxt('u_nom_oliver.dat', delimiter = ' ')
#mean_ux_nom = []
#sigma_eff_nom = []

#for i in range(31):
	#mean_ux_nom.append(ux_nom[i,2])
	#sigma_eff_nom.append(ux_nom[i,3])

"""""""""""""""""""""  Probability Function for e_sampling_h/q_h  """""""""""""""""""""""""""

with open("mean_sampling_error.dat", "w") as out_file:	

	out_string =  "mean_sampling_error, sampling_error_5, sampling_error_95"

	for i in range(len(sigma_eff_nom)):

		# Sample from a normal distribution with mean 0 and standard deviation sigma_n_i
		mu, sigma_n_i = 0, sigma_eff_nom[i] 
		sampler_sampling_error = random.normal(mu, sigma_n_i, 10000000)

		
		# Compute the mean and the 5th and 95th percentile
		mean_sampling_error = median(sampler_sampling_error, axis = 0)
		sampling_error_5, sampling_error_95 = percentile(sampler_sampling_error, [5,95])
		sampling_error_5, sampling_error_95 = sampling_error_5 / abs(mean_ux_nom[i]), sampling_error_95 / abs(mean_ux_nom[i])

		# Loop on the results to be written in the output file
		out_string += "\n"
		out_string += str(mean_sampling_error) + ',' + str(sampling_error_5) + ',' + str(sampling_error_95) 
		print("lol")


	# Write the complete results in the output file
	out_file.write(out_string)


