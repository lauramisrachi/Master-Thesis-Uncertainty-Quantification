from ar import arsel
from numpy import *
from scipy.signal import lfilter, lfiltic
import scipy.io
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rcParams
from matplotlib.patches import Rectangle

matplotlib.rc('xtick', labelsize = 13) 
matplotlib.rc('ytick', labelsize = 13) 
rcParams["font.size"] = 12
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica"]
rcParams["text.usetex"] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rc('savefig', dpi = 500)

def trapezes(f,a,b,n):
	h = (b - a)/float(n)
	z = 0.5 * (f[a] + f[b])
	for i in range(1,n):
		z = z + f[a + i * h]

	return h * z


ux_data = loadtxt("integral_time_scale.dat", delimiter = ',')
M = len(ux_data)
print(M)



# Computing the arsel for each y+ value set of data

a = arsel(ux_data)
autocor_function = a.autocor
p = [reciprocal(roots(model[::-1])) for model in a.AR]
x = a.mu[0] + lfilter([1], a.AR[0], sqrt(a.sigma2eps[0])*random.randn(M))

zi = lfiltic([1], a.AR[0], a.autocor[0])
rho = ones(M+1)
rho[1:] = lfilter([1], a.AR[0], zeros(M), zi=zi)[0]
sigma_eff = a.sigma2x
mean = a.mu
Tu = sigma_eff / mean

d = arange(M + 1)
I = trapezes(rho,0,M,M)
epsilon_T = 0.005
T = (2 * I) / ((epsilon_T) ** 2) * (Tu ** 2) 

print(I,Tu,T)

#plt.plot(d, rho)
#plt.show()






