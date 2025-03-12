"""@author Taketo
Set up file for fluxonium CW measurement.
"""
import sys

setup_file = __file__
# database path
db_path_local=r"C:\Users\sqil\Desktop\data_local\IntegerFluxonium\sampleD\20250303_CW"
db_path=r"Z:\Projects\IntegerFluxonium\Measurements\Second_iteration\SampleD_UMD_NoiseSpec\CW_meas_data\sampleD_20250303_CW"

# import param_dict
sys.path.append(r"\\sb1files\SQIL\Projects\IntegerFluxonium\Measurements\Second_iteration\SampleD_UMD_NoiseSpec\CW_measurement\measurement_codes\helpers\param_dict")
from pd_20250303_IntegerFluxonium_sampleD_CW import pd_file, param_dict

# wiring/RT attenuation setup(RIGHT Line)
wiring = "\n".join([
    # "ZNA26_port1 - 3feet(T40-3FT-KMKM+) - PowerSplitter(ZN4PD-K44+)_port1",
    # "Signal Core - 3feet(T40-3FT-KMKM+) - PowerSplitter(ZN4PD-K44+)_port2",
    # "SGS - 3feet(T40-3FT-KMKM+) - PowerSplitter(ZN4PD-K44+)_port2",
    # "PowerSplitter(ZN4PD-K44+)_portS - 6feet(E40-6FT-KMKM+) - In11",
    
    # "Out Rightline - RTamp - 6feet(E40-6FT-KMKM+) - SMAfemale-female - 2feet(T40-2FT-KMKM+) - ZNA26_port2",
 #   - PowerSplitter(ZN4PD-K44+)_port1",
    # "R&S_SGS100A - 2feet(T40-2FT-KMKM+) - 10dB - 10dB - PowerSplitter(ZN4PD-K44+)_port2",
])

# Instrument IP address
vna_IP = r'TCPIP0::192.168.1.60::inst0::INSTR' # AQUA ZNA26.5GHz
# vna_IP = r'TCPIP0::192.168.1.131::inst0::INSTR' # SQIL ZNB20.0GHz
yoko_IP = r'TCPIP0::192.168.1.51::inst0::INSTR'
# sgs_IP = r"USB::0x0AAD::0x0088::114906::INSTR" # SGS connected via USB port
sgs_IP = r'TCPIP0::192.168.1.58::inst0::INSTR'

# ZI descriptor 