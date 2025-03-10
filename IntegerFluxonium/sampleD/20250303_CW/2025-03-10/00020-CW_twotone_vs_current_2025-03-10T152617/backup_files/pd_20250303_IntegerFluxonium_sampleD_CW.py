import numpy as np
pd_file=__file__


ro_freq = 7.4600e9 #[Hz]
# qu_freq = 3.790e9 #[Hz]
qu_freq = 3.645e9 #[Hz] # for "current":-495.826e-6, # [A]
qu_freq02_by2 = 0
freq_03 = 0
# freq_05 = 0

# two_chi = 10e6 #[Hz]

param_dict = {
    "ro_freq": ro_freq, 
    "ro_power":-10,#-30, # [dBm]
    
    "qu_freq": qu_freq,
    "qu_power": -10,#-29,  # [dBm]
    
    
    #"current":-505.8e-6, # [A]
    #"current":-493.129e-6, # [A]
    "current":5.0e-6, # [A]
     
    #### CW onetone ####
    "CW_onetone":{
        "ro_freq_step":2e6, #[Hz]
        "ro_freq_start":ro_freq,# - 50e6, # [Hz]
        "ro_freq_stop":ro_freq,# + 50e6, # [Hz] must be larger than start
        "ro_freq_npts":False, # if "False", npts is calculated using ro_freqstep
        
        "sweep":False,#"ro_power", # str. name of the sweep parameter
    },
    
    #### CW twotone ####
    "CW_twotone":{
        "qu_freq_step":0.5e6,#5e6, # [Hz]
        "qu_freq_start": 3.50e9,#qu_freq,# - 800e6 , # [Hz]
        "qu_freq_stop": 3.65e9, #qu_freq + 20e6, #qu_freq+300e6, # [Hz]
        "qu_freq_npts":False, # if "False", npts is calculated using ro_freqstep

        "sweep":"current",#"index", # str. name of the sweep parameter
    },
    
    "CW_twotone_sweep_ro_freq":{
        "ro_freq_step":0.25e6, #[Hz]
        "ro_freq_start":ro_freq-200e6, # [Hz]
        "ro_freq_stop":ro_freq + 200e6, # [Hz] must be larger than start
        "ro_freq_npts":False, # if "False", npts is calculated using ro_freqstep
        
        "sweep":False, # str. name of the sweep parameter
    },
    
    #### VNA setting ####
    "vna_bw":100, # bandwidth [Hz]
    "vna_avg":1, #10# average [#]
   
    #### yokogawa setting ####
    "gs_rampstep":1e-6,
    "gs_delay":500e-6,
    "gs_voltage_lim": 1, # [V]
    "gs_current_range": 1e-3,
    
    "index": False,
    #"sweep_start":0,
    #"sweep_stop":1000000,#-40
    #"sweep_npts":1000001,#10
    #"sweep_list":False, #[10*np.log10(p_mW) for p_mW in np.linspace(0.001, 0.002, 10)], 
    
    
    # current
    "sweep_start":5.0e-6,#-506e-6,
    "sweep_stop": 7.0e-6,#-456e-6,
    "sweep_npts": 3,#51,
    "sweep_list":False, #[10*np.log10(p_mW) for p_mW in np.linspace(0.001, 0.002, 10)], 
    
    
    # ro_freq
    #"sweep_start":7.4640e9 - 50e6,
    #"sweep_stop": 7.4640e9 + 50e6,#-40
    #"sweep_npts":51,#10
    #"sweep_list":False, #[10*np.log10(p_mW) for p_mW in np.linspace(0.001, 0.002, 10)], 
    
    # ro_power
    #"sweep_start":-50,
    #"sweep_stop": -20,#-40
    #"sweep_npts":31,#10
    #"sweep_list":False, #[10*np.log10(p_mW) for p_mW in np.linspace(0.001, 0.002, 10)], 
    
    # qu_power
    #"sweep_start":-20,
    #"sweep_stop": -0,#-40
    #"sweep_npts":21,#10
    #"sweep_list":False, #[10*np.log10(p_mW) for p_mW in np.linspace(0.001, 0.002, 10)], 
    
    
    
    #### sweep list ####
    
    #### sweep list ####
    # "sweep_start":20.35e9-6*two_chi,
    # "sweep_stop":20.35e9+6*two_chi,
    # "sweep_npts":41,
    # "sweep_list":False, #s[10*np.log10(p_mW) for p_mW in np.linspace(0.016, 0.027, 5)]
    
    
    

}
""" Note
"qu_freq_01": 4.7317e9,
"""