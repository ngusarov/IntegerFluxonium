import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


filename = os.path.join(filepath_parent, 'data.ddh5')
h5file = h5py.File(filename,"r")

qu_freq = h5file["data"]["qu_freq"][0][:]*1e-9   
mag_dB = np.array(h5file["data"]["mag_dB"][:][:])
peak_fit = np.empty((0,3), float)


from fit_code.spectroscopy_1D_plot import spectroscopy_1D_plot
for run in range(repetitions):

    qu_freq = h5file["data"]["qu_freq"][0]*1e-9   
    mag_dB = np.array(h5file["data"]["mag_dB"][run][:]).T

    fig_fit, res_freq, FWHM = spectroscopy_1D_plot(qu_freq, mag_dB, xlabel='qu freq [GHz]', ylabel='Mag[]', fit_enabled=True)
    #plt.close()
    peak_fit = np.append(peak_fit, [[run, res_freq, FWHM]], axis=0)
fig_stats = plt.figure()
ax = fig_stats.add_subplot(111)

ax.plot(peak_fit[:,0], peak_fit[:,1], marker = 'o', color='navy')
ax2 = ax.twinx()
ax2.plot(peak_fit[:,0], peak_fit[:,2], marker = 'o', color='orange')
ax.set_xlabel('run', fontsize=14)
ax.set_ylim([peak_fit[:,1].mean()-0.015, peak_fit[:,1].mean()+0.015])
ax.spines['right'].set_color('navy')
ax.tick_params(axis='y', colors='navy')
ax2.set_ylabel('fwhm', fontsize=14)
ax2.spines['right'].set_color('orange')
ax2.tick_params(axis='y', colors='orange')
ax2.set_ylim([peak_fit[:,2].mean()-0.003, peak_fit[:,2].mean()+0.003])
ax.set_ylabel('qu 0-1 freq fit [GHz]', fontsize=14)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
fig_stats.savefig(f"{filepath_parent}/qubit_freq_stats.png", bbox_inches='tight')
       

