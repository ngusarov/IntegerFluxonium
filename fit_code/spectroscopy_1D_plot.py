# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:24:53 2024

@author: taketo
"""

import numpy as np 
import h5py 
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import fits as fits

import seaborn as sns
colors = sns.color_palette("tab10", 6)
figsize=(10,8)
def spectroscopy_1D_plot(ro_freq_sweep, data, xlabel='freq', ylabel='Mag[]', fit_enabled=False):    
    
    fig_fit = plt.figure(figsize=figsize)
    ax_fit = fig_fit.add_subplot(111)
    transform = fits.Transform(ro_freq_sweep, data, 'ph')
    tranformed_data = data#transform#.data()
    ax_fit.plot(ro_freq_sweep, tranformed_data, color='tab:blue')
    if fit_enabled == True:
        # print(ro_freq_sweep.shape, tranformed_data.shape)
        
        try:
            fit = fits.Lorentzian(ro_freq_sweep, tranformed_data)
            (FWHM, FWHM_err) = (fit.fwhm(), fit.fwhm_error())
            (res_freq, res_freq_error) = (fit.resonance_frequency(), fit.resonance_frequency_error())
            # print("Error: ", "FWHM", FWHM_err, "res_freq", res_freq_error)
            freq_fit = np.linspace(np.min(ro_freq_sweep), np.max(ro_freq_sweep), 1001)
            data_fit = fit.fit(freq_fit)
            FWHM_abs = np.abs(FWHM)

            # if (FWHM_err <= 0.003 and res_freq_error <= 0.0009) and \
            #     (res_freq < max(ro_freq_sweep) and res_freq > min(ro_freq_sweep)) and \
            #         data_fit[1] < data_fit[0]:
            #     ax_fit.plot(freq_fit, data_fit, color='tab:red', ls='-', lw=1)
            # else:
            #     idx = np.argmin(tranformed_data)
            #     res_freq = ro_freq_sweep[idx]
            #     ax_fit.axvline(res_freq, color='tab:red', ls='--', lw=1)

            ax_fit.set_title(f"f0 = {res_freq:.4e} GHz | FWHM = {FWHM_abs:.4e} GHz")
            ax_fit.vlines(res_freq-FWHM_abs/2, tranformed_data.min(), tranformed_data.max(), linestyles='dashed')
            ax_fit.vlines(res_freq+FWHM_abs/2, tranformed_data.min(), tranformed_data.max(), linestyles='dashed')

        except:
            print("Fit failed")
            idx = np.argmin(tranformed_data)
            res_freq = ro_freq_sweep[idx]
            ax_fit.axvline(res_freq, color='tab:red', ls='--', lw=1)
            FWHM_abs = 0

    else:
        res_freq = None
        FWHM_abs = None
    ax_fit.set_xlabel(xlabel)
    ax_fit.set_ylabel(ylabel)
    
    return fig_fit, res_freq, FWHM_abs

def spectroscopy_1D_plot_dispersive_shift(ro_freq_sweep, data, tags, xlabel, ylabel='Mag[]', colors=colors):    
    
    fig_fit = plt.figure(figsize=figsize)
    ax_fit = fig_fit.add_subplot(111)
    res_freq_multi = []
    FWHM_multi = []
    center_point_idx = []
    title = 'f0'
    for idx, trace_tag in enumerate(tags):
        trace_data = data[idx]
        transform = fits.Transform(ro_freq_sweep, trace_data, 'ampl')
        tranformed_data = transform.data()
        fit = fits.Lorentzian(ro_freq_sweep, tranformed_data)
        (FWHM, FWHM_err) = (fit.fwhm(), fit.fwhm_error())
        res_freq = fit.resonance_frequency()
        ax_fit.plot(ro_freq_sweep, transform.data(), label = trace_tag, color = colors[idx])
        freq_fit = np.linspace(np.min(ro_freq_sweep), np.max(ro_freq_sweep), 1001)
        data_fit = fit.fit(freq_fit)
        ax_fit.plot(freq_fit, data_fit, color='black', ls='-', lw=2)
    
        res_freq_multi = np.append(res_freq_multi, int(res_freq))
        FWHM_multi = np.append(FWHM_multi, int(FWHM))
        center_point_idx = np.append(center_point_idx, int(np.abs(transform.data() - data_fit.max()).argmin()))
        title = title + " | " + str(int(res_freq)) + " Hz" 

    ax_fit.set_xlabel(xlabel)
    ax_fit.set_ylabel(ylabel)
    ax_fit.legend()
    ax_fit.set_title(title)

    #phase plot with unwrap angle data, unwrapping starts from center frequency
    fig_phase = plt.figure()
    ax_phase = fig_phase.add_subplot(111)
    ax_phase_delta = ax_phase.twinx()
    for idx, trace_tag in enumerate(tags):
        trace_data = data[idx]
        print(center_point_idx[idx])
        transform_left = fits.Transform(np.flip(ro_freq_sweep[0:int(center_point_idx[idx])]), np.flip(trace_data[0:int(center_point_idx[idx])]), 'angle')
        tranformed_data_left = transform_left.data()
        transform_right = fits.Transform(ro_freq_sweep[int(center_point_idx[idx])::], trace_data[int(center_point_idx[idx])::], 'angle')
        tranformed_data_right = transform_right.data()
        tranformed_data = np.concatenate((np.flip(tranformed_data_left), tranformed_data_right))
        ax_phase.plot(ro_freq_sweep, tranformed_data, label = trace_tag, color = colors[idx])
    
        #plot phase delta, park at frequency between both resonance frequencies
        phase_delta_idx = int(np.abs(center_point_idx[idx] + center_point_idx[idx-1])/2)
        print(phase_delta_idx)
        if idx>0:
            trace_data = data[idx-1] - data[idx]
            transform_left = fits.Transform(np.flip(ro_freq_sweep[0:phase_delta_idx]), np.flip(trace_data[0:phase_delta_idx]), 'angle')
            tranformed_data_left = transform_left.data()
            transform_right = fits.Transform(ro_freq_sweep[phase_delta_idx::], trace_data[phase_delta_idx::], 'angle')
            tranformed_data_right = transform_right.data()
            tranformed_data = np.concatenate((np.flip(tranformed_data_left), tranformed_data_right))
            ax_phase_delta.plot(ro_freq_sweep, tranformed_data, label = 'delta()', color = 'green')
    
    y_max = np.abs(ax_phase.get_ylim()).max()
    ax_phase.set_ylim(ymin=-y_max, ymax=y_max)
    y_max = np.abs(ax_phase_delta.get_ylim()).max()
    ax_phase_delta.set_ylim(ymin=-y_max, ymax=y_max)
    ax_phase.set_xlabel(xlabel)
    ax_phase.set_ylabel('phase []')
    ax_phase_delta.set_ylabel('phase delta []')
    ax_phase.legend()
    ax_phase.set_title('phase unwrapped around center between each resonance frequency pair')

    return fig_fit, res_freq_multi, FWHM_multi



if __name__ == '__main__':
    #provide path to database element

    '''    
    # example for single trace
    path = r"\\sb1files.epfl.ch\SQIL\Projects\Transmon\Data\Database\data\AQUA_leftline_double_transmon_240417\2024-04-19\2024-04-19T101556_0c5f94ec-onetone_vs_ro_power"
    filename = path+r"\data.ddh5"
    h5file = h5py.File(filename,"r")
    
    ro_freq = h5file["data"]["ro_freq"][0]        
    data = h5file["data"]["data"][0]
    spectroscopy_1D_plot(ro_freq, data)
    plt.show()
    '''
    
    
    #example for dispersive shift
    path = r"Z:\Projects\Transmon\Data\Database\data\AQUA_leftline_double_transmon_240501\2024-05-10\2024-05-10T163456_788ee0f3-dispersive_shift"
    filename = path+r"\data.ddh5"
    h5file = h5py.File(filename,"r")
    
    ro_freq = h5file["data"]["ro_freq"][:][:]    
    ground_data = h5file["data"]["ground_data"][:][:]       
    excited_data = h5file["data"]["excited_data"][:][:]   
    fig_1D_plot, res_freq_multi, FWHM_multi = spectroscopy_1D_plot_dispersive_shift(ro_freq, [ground_data, excited_data], ['g-state', 'e-state'], 'RO Frequency [Hz]', 'Mag []', ['blue', 'red'], )
    plt.show()
    

    '''
    #example for single trace of VNA
    path = r""
    filename = path+r"\data.ddh5"
    h5file = h5py.File(filename,"r")
    
    ro_freq = h5file["data"]["ro_freq"][0]        
    data = h5file["data"]["mag_dB"][0]
    spectroscopy_1D_plot(ro_freq, 10**(np.array(data)/20)) # rescale from dB to linear
    plt.show()
    '''

    '''    
    #example for extracted single trace from VNA
    filepath_parent = r'Z:\Projects\SpinChain\samples\20241010_single_fluxonium\database\2024-10-18\00263-CW_onetone_fixed_freq_vs_flux_vs_flux_through_power_2024-10-18T143416'
    filename = os.path.join(filepath_parent, 'data.ddh5')
    h5file = h5py.File(filename,"r")

    power = h5file["data"]["power"][:] 
    current = h5file["data"]["current"][0]
    mag_dB = np.array(h5file["data"]["mag_dB"][0][:])
    phase =  np.array(h5file["data"]["phase"][0][:])

    spectroscopy_1D_plot(current, mag_dB, xlabel='current', ylabel='Mag[]')

    spectroscopy_1D_plot(current, phase, xlabel='current', ylabel='Phase')
    '''

    