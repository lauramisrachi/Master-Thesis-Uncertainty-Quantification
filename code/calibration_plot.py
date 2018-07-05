from numpy import *
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rcParams

matplotlib.rc('xtick', labelsize = 13) 
matplotlib.rc('ytick', labelsize = 13) 
rcParams["font.size"] = 12
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica"]
rcParams["text.usetex"] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rc('savefig', dpi = 500)

# Import the calibration file

calibration_finest = loadtxt("calibration_pdf_oliver.dat", skiprows = 1)
y_plus_data = loadtxt("y_plus_data_oliver.dat")
print(y_plus_data)
print(calibration_finest)
y_plus_good = y_plus_data[0:len(calibration_finest) + 1]
N = len(calibration_finest)
print(N)


plt.plot(y_plus_good, calibration_finest, lw = 2.5, color = 'darkblue')
plt.ylim([0,1.0])
plt.xlim([0, y_plus_good[N-1] + 1])
plt.axhspan(0.05, 0.95, color = 'lightgray')
plt.xlabel("$y_+$", fontsize = 17)
plt.ylabel("$CDF(q_{finest})$", fontsize = 17)
plt.title("$  < u'u' >  $", fontsize = 17)
plt.savefig('test_calib_xd.png')