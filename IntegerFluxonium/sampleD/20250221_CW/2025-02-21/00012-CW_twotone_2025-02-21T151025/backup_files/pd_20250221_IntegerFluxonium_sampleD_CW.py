import numpy as np
pd_file=__file__


ro_freq = 7.460e9 #[Hz]
qu_freq = 3.795e9 #[Hz]
qu_freq02_by2 = 0
freq_03 = 0
# freq_05 = 0

# two_chi = 10e6 #[Hz]

param_dict = {
    "ro_freq": ro_freq, 
    "ro_power":-35, # [dBm]
    
    "qu_freq": qu_freq,
    "qu_power": -15,  # [dBm]
    
    "current":-505.8e-6, # [A]
     
    #### CW onetone ####
    "CW_onetone":{
        "ro_freq_step":1e6, #[Hz]
        "ro_freq_start":ro_freq - 350e6, # [Hz]
        "ro_freq_stop":ro_freq + 350e6, # [Hz] must be larger than start
        "ro_freq_npts":False, # if "False", npts is calculated using ro_freqstep
        
        "sweep":False,#"ro_power", # str. name of the sweep parameter
    },
    
    #### CW twotone ####
    "CW_twotone":{
        "qu_freq_step":0.25e6,#0.05e6, # [Hz]
        "qu_freq_start": qu_freq - 30e6 , # [Hz]
        "qu_freq_stop":qu_freq + 60e6, #qu_freq+300e6, # [Hz]
        "qu_freq_npts":False, # if "False", npts is calculated using ro_freqstep

        "sweep":False, # str. name of the sweep parameter
    },
    
    "CW_twotone_sweep_ro_freq":{
        "ro_freq_step":0.25e6, #[Hz]
        "ro_freq_start":ro_freq-200e6, # [Hz]
        "ro_freq_stop":ro_freq + 200e6, # [Hz] must be larger than start
        "ro_freq_npts":False, # if "False", npts is calculated using ro_freqstep
        
        "sweep":False, # str. name of the sweep parameter
    },
    
    #### VNA setting ####
    "vna_bw":0.5e2, # bandwidth [Hz]
    "vna_avg":1, #10# average [#]
   
    #### yokogawa setting ####
    "gs_rampstep":1e-6,
    "gs_delay":500e-6,
    "gs_voltage_lim": 1, # [V]
    "gs_current_range": 1e-3,
    
    "index":False,
    
    #### sweep list ####
    
    #### sweep list ####
    # "sweep_start":20.35e9-6*two_chi,
    # "sweep_stop":20.35e9+6*two_chi,
    # "sweep_npts":41,
    # "sweep_list":False, #s[10*np.log10(p_mW) for p_mW in np.linspace(0.016, 0.027, 5)]
    
    
    
    "sweep_start":-40,
    "sweep_stop":15,#-40
    "sweep_npts":111,#10
    "sweep_list":False, #[10*np.log10(p_mW) for p_mW in np.linspace(0.001, 0.002, 10)], 
    
}
""" Note
"qu_freq_01": 4.7317e9,
"""