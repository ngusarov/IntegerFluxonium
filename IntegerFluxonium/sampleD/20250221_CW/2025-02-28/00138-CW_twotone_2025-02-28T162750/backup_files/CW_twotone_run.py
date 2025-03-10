"""
for fluxonium measurement.
CW twotone with VNA and Signal Core.
@Taketo
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import tqdm
import sys
import os
import json
import h5py
from distutils.dir_util import copy_tree
from helpers.customized_drivers.Rohde_Schwarz_ZNA26 import (
    RohdeSchwarzZNA26,
)
from helpers.customized_drivers.ZNB_taketo import (
    RohdeSchwarzZNBChannel,
)
from qcodes.instrument_drivers.yokogawa import YokogawaGS200
from qcodes_contrib_drivers.drivers.SignalCore.SignalCore import SC5521A
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzSGS100A
from helpers.taketo_datadict_storage import DataDict, DDH5Writer
from helpers.setup.CW_Setup import (
    setup_file,
    db_path_local,
    db_path,
    pd_file,
    param_dict,
    vna_IP,
    yoko_IP,
    sgs_IP,
    wiring,
)
from helpers.utilities import(
    current_range_check
)
sys.path.append("../analysis_code")
from spectroscopy_1D_plot import spectroscopy_1D_plot
from spectroscopy_2D_plot import spectroscopy_2D_plot
from CW_twotone_plot import *

####################################################################################
qudrive_source="SGS" # "SGS" or "SignalCore"
Yokogawa=True # True or False
####################################################################################

exp_name = "CW_twotone"
tags = ["0_CW twotone"]

# define qubit drive frequency list
freq_start=param_dict["CW_twotone"]["qu_freq_start"]
freq_stop=param_dict["CW_twotone"]["qu_freq_stop"]
if param_dict["CW_twotone"]["qu_freq_npts"]==False:
    freq_npts = int(abs(freq_stop-freq_start)/param_dict["CW_twotone"]["qu_freq_step"])+1
else:
    freq_npts = param_dict["CW_twotone"]["qu_freq_npts"]
param_dict["CW_twotone"]["qu_freq_npts"] = freq_npts # update freq_npts to param_dict
freq_list = np.linspace(freq_start,freq_stop,freq_npts)

# define datadict
if param_dict["CW_twotone"]["sweep"]==False:
    sweep_list = [0]
    # define DataDict for saving in DDH5 format
    datadict = DataDict(
            qu_freq=dict(unit="sec"),
            mag_dB=dict(axes=["qu_freq"],
                        unit="dB"),
            phase=dict(axes=["qu_freq"],
                       unit="rad")
            )
    datadict.validate()
else:
    exp_name = exp_name+"_vs_"+param_dict["CW_twotone"]["sweep"]
    tags.append(f"0_{exp_name}")
    param_dict[param_dict["CW_twotone"]["sweep"]]="sweeping"
    if param_dict["sweep_list"]==False:
        sweep_list = np.linspace(param_dict["sweep_start"],
                                param_dict["sweep_stop"],
                                param_dict["sweep_npts"]
                                )
        param_dict["sweep_list"]=sweep_list
    else:
        sweep_list = param_dict["sweep_list"]
        param_dict["sweep_start"]=False
        param_dict["sweep_stop"]=False
        param_dict["sweep_npts"]=False
    # define DataDict for saving in DDH5 format
    datadict = DataDict(
            qu_freq=dict(unit="sec"),
            sweep_param=dict(unit=""),
            mag_dB=dict(axes=["qu_freq","sweep_param"],
                        unit="dB"),
            phase=dict(axes=["qu_freq","sweep_param"],
                       unit="rad")
            )
    datadict.validate()

with DDH5Writer(datadict, db_path_local, name=exp_name) as writer:
    filepath_parent = writer.filepath.parent
    writer.add_tag(tags)
    writer.save_dict('param_dict.json',param_dict)
    writer.backup_file([__file__, setup_file, pd_file])
    writer.save_text("wiring.md", wiring)
    
    # take the last two stages of the filepath_parent
    path = str(filepath_parent)  
    last_two_parts = path.split(os.sep)[-2:]
    new_path = os.path.join(db_path, *last_two_parts)
    writer.save_text("directry_path.md",new_path)
    
    # connect to the equipment
    vna = RohdeSchwarzZNA26('VNA',
                            vna_IP,
                            init_s_params=False,
                            reset_channels=False,
                            )
    
    
    # setting of VNA
    # keeping the current S21 settings (calibration and etc...)
    #chan = RohdeSchwarzZNBChannel(
    #        vna,
    #        name="S21",
    #        channel=1,
    #        vna_parameter="S21",
    #        
    #        existing_trace_to_bind_to="Trc1",
    #    )
    #vna.channels.append(chan)
    
    # vna.add_channel('S21')
    # vna.cont_meas_on()
    # vna.display_single_window()
    # vna.channels.S21.sweep_type('CW_Point')
    # vna.channels.S21.power.set(-50) # for safety
    
    vna.add_channel('S21')
    vna.cont_meas_on()
    vna.display_single_window()
    vna.channels.S21.sweep_type('CW_Point')
    vna.channels.S21.power.set(-50) # for safety
    
    
    print(vna.channels)
    print()
    print(vna.channels.S21)
    print()
    #print(vna.ask("*LRN?"))  # Lists all available channels and traces
    print()
    print("VNA is ON:", vna.rf_on())
    print("Current sweep type:", vna.channels.S21.sweep_type.get())

    
    
    
    
    
    
    ## update parameters
    # setting of VNA        
    vna.channels.S21.npts(1)
    #vna.channels.S21.start(param_dict["ro_freq"])
    #vna.channels.S21.stop(param_dict["ro_freq"])
    vna.channels.S21.cw_frequency(param_dict["ro_freq"])
    vna.channels.S21.power.set(param_dict["ro_power"])
    vna.channels.S21.bandwidth(param_dict["vna_bw"]) 
    vna.channels.S21.avg(param_dict["vna_avg"])
    
    
    
    vna.rf_on()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    if qudrive_source=="SGS":
        sgsa = RohdeSchwarzSGS100A("SGSA100", sgs_IP)
        # setting of R&S SGS100A
        sgsa.status(False)
        sgsa.power(-60) # for safety
        sgsa.status(True)
    if qudrive_source=="SignalCore":
        sc = SC5521A('mw1')
        # setting of signal core
        sc.power(-10) # for safety
        sc.status("off")
        sc.clock_frequency(10)
        sc.status('on')   
    if Yokogawa==True:
        gs = YokogawaGS200("gs200", yoko_IP)
        
        
    if qudrive_source=="SGS":
        sgsa.power(param_dict["qu_power"])
    if qudrive_source=="SignalCore":
        sc.power(param_dict["qu_power"])
    if Yokogawa==True:
        # setting of Yokogawa
        gs.voltage_limit(param_dict["gs_voltage_lim"])
        gs.current_range(param_dict["gs_current_range"])
        current_range_check(param_dict)
        gs.ramp_current(param_dict["current"],
                        param_dict["gs_rampstep"],
                        param_dict["gs_delay"])
        # time.sleep(1.0) # wait for the current reaching at the target
    
    
    
        
    for sweep_param in tqdm.tqdm(sweep_list):
        # update param_dict
        if not param_dict["CW_twotone"]["sweep"]==False:
            param_dict[param_dict["CW_twotone"]["sweep"]]=sweep_param
        
        ## update parameters
        # setting of VNA        
        #vna.channels.S21.npts(1)
        #vna.channels.S21.start(param_dict["ro_freq"])
        #vna.channels.S21.stop(param_dict["ro_freq"])
        #vna.channels.S21.power.set(param_dict["ro_power"])
        #vna.channels.S21.bandwidth(param_dict["vna_bw"]) 
        #vna.channels.S21.avg(param_dict["vna_avg"])
        
        #if qudrive_source=="SGS":
        #    sgsa.power(param_dict["qu_power"])
        #if qudrive_source=="SignalCore":
        #    sc.power(param_dict["qu_power"])
        #if Yokogawa==True:
        #    # setting of Yokogawa
        #    gs.voltage_limit(param_dict["gs_voltage_lim"])
        #    gs.current_range(param_dict["gs_current_range"])
        #    current_range_check(param_dict)
        #    gs.ramp_current(param_dict["current"],
        #                    param_dict["gs_rampstep"],
        #                    param_dict["gs_delay"])
        #    # time.sleep(1.0) # wait for the current reaching at the target
        
        mag_dB_array = np.zeros(freq_npts)
        phase_array = np.zeros(freq_npts)
        # measurement
        for idx, qu_freq in enumerate(freq_list):
            if qudrive_source=="SGS":
               sgsa.frequency(qu_freq)
            if qudrive_source=="SignalCore":
                sc.frequency(qu_freq)
             
           
            #mag_dB, phase =vna.channels.S21.trace_db_phase.get()
            vna.channels.S21.autoscale()
            mag_dB, phase =vna.channels.S21.point_fixed_frequency_db_phase.get()
            mag_dB_array[idx] = mag_dB#[0]
            phase_array[idx] = phase#[0]
        
            # # record the measurement time (in YYYYMMDDhhmmss format)
            # now = datetime.datetime.now()
            # time = int(now.strftime('%Y%m%d%H%M%S'))
        
        # save the data
        if param_dict["CW_twotone"]["sweep"]==False:
            writer.add_data(
                qu_freq=freq_list,
                mag_dB=mag_dB_array,
                phase=phase_array
                )
        else:
            writer.add_data(
                qu_freq=freq_list,
                sweep_param=sweep_param,
                mag_dB=mag_dB_array,
                phase=phase_array
                )
        
        # plot the single trace
        if param_dict["CW_twotone"]["sweep"]==False:
            fig_fit, f0, fwhm = spectroscopy_1D_plot(freq_list, 10**(mag_dB_array/20))
            fig_fit.suptitle(f"Qubit drive power: {param_dict["qu_power"]}dBm")
            fig_fit.savefig(f"{writer.filepath.parent}/00_1D_vs_qu_freq_fit.png", bbox_inches='tight')
            writer.save_text(f"fitted_resonator_freq_at_qu_power_{param_dict["qu_power"]}dBm.md", f'fitted_qubit_freq:{f0} Hz')
            
    vna.rf_off()
    vna.clear_channels()
    vna.close()
    if qudrive_source=="SGS":
        sgsa.status(False)
        sgsa.close()
    if qudrive_source=="SignalCore":
        sc.status('off')
        sc.close()
    if Yokogawa==True:
        gs.close()
    
### plotting
path = str(filepath_parent)  
print(path) 
# plot 1D vs qu frequency
if param_dict["CW_twotone"]["sweep"]==False:
    plotting_1D_vs_qu_freq(path)
    
# plot 1D vs ro power
if freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="qu_power":
        plotting_1D_vs_qu_power(path)
# plot 1D vs current
if freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="current":
        plotting_1D_vs_current(path)
# plot 1D vs ro power
if freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="qu_freq":
        plotting_1D_vs_qu_freq(path)
# plot 1D vs ro power
if freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="qu_power":
        plotting_1D_vs_qu_power(path)

# plot 1p5D vs qu_power
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="qu_power":
        plotting_1p5D_vs_qu_power(path)
# plot 1p5D vs current
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="current":
        plotting_1p5D_vs_current(path)
# plot 1p5D vs ro_freq
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="ro_freq":
        plotting_1p5D_vs_ro_freq(path)
# plot 1p5D vs ro_power
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="ro_power":
        plotting_1p5D_vs_ro_power(path)
        
# plot 2D vs ro_freq vs qu_power
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="qu_power":
        plotting_2D_vs_qu_power(path)
# plot 2D vs current
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="current":
        plotting_2D_vs_current(path)
# plot 2D vs ro_freq vs ro_freq
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="ro_freq":
        plotting_2D_vs_ro_freq(path)
# plot 2D vs ro_freq vs ro_power
if not freq_npts==1:
    if param_dict["CW_twotone"]["sweep"]=="ro_power":
        plotting_2D_vs_ro_power(path)


# copy the directory to the server
copy_tree(filepath_parent,new_path)

plt.show()