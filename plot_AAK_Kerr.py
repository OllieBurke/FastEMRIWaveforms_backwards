import cupy as cp
import numpy as np
import os 
import sys
sys.path.append("../")


# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform, AAKWaveformBase, KerrEquatorialEccentric,KerrEquatorialEccentricWaveformBase
from few.trajectory.inspiral import EMRIInspiral
from few.summation.directmodesum import DirectModeSum 
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.aakwave import AAKSummation, KerrAAKSummation 

from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc
from few.utils.modeselector import ModeSelector, NeuralModeSelector
from few.utils.utility import get_separatrix, Y_to_xI, get_p_at_t

import matplotlib.pyplot as plt

# Import features from eryn

YRSID_SI = 31558149.763545603

np.random.seed(1234)

def sensitivity_LWA(f):
    """
    LISA sensitivity function in the long-wavelength approximation (https://arxiv.org/pdf/1803.01944.pdf).
    
    args:
        f (float): LISA-band frequency of the signal
    
    Returns:
        The output sensitivity strain Sn(f)
    """
    
    #Defining supporting functions
    L = 2.5e9 #m
    fstar = 19.09e-3 #Hz
    
    P_OMS = (1.5e-11**2)*(1+(2e-3/f)**4) #Hz-1
    P_acc = (3e-15**2)*(1+(0.4e-3/f)**2)*(1+(f/8e-3)**4) #Hz-1
    
    #S_c changes depending on signal duration (Equation 14 in 1803.01944)
    #for 1 year
    alpha = 0.171
    beta = 292
    kappa = 1020
    gamma = 1680
    fk = 0.00215
    #log10_Sc = (np.log10(9)-45) -7/3*np.log10(f) -(f*alpha + beta*f*np.sin(kappa*f))*np.log10(np.e) + np.log10(1 + np.tanh(gamma*(fk-f))) #Hz-1 
    
    A=9e-45
    Sc = A*f**(-7/3)*np.exp(-f**alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(fk-f)))
    sensitivity_LWA = (10/(3*L**2))*(P_OMS+4*(P_acc)/((2*np.pi*f)**4))*(1 + 6*f**2/(10*fstar**2))+Sc
    return sensitivity_LWA
def zero_pad(data):
    """
    Inputs: data stream of length N
    Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
    """
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    """
    Compute stationary noise-weighted inner product
    Inputs: sig1_f and sig2_f are signals in frequency domain 
            N_t length of padded signal in time domain
            delta_t sampling interval
            PSD Power spectral density

    Returns: Noise weighted inner product 
    """
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))
def SNR_function(sig1_t, dt, N_channels = 2):
    N_t = len(sig1_t[0])

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))
    
    freq_np = xp.asnumpy(xp.fft.rfftfreq(N_t, dt))

    freq_np[0] = freq_np[1] 

    PSD = 2 * [xp.asarray(sensitivity_LWA(freq_np))]

    SNR2 = xp.asarray([inner_prod(sig1_f[i], sig1_f[i], N_t, dt,PSD[i]) for i in range(N_channels)])

    SNR = xp.sum(SNR2)**(1/2)

    return SNR

def mismatch_func(sig1_t,sig2_t,dt,N_channels = 2):
    N_t = len(sig1_t[0])

    sig1_f = [xp.fft.rfft(zero_pad(sig1_t[i])) for i in range(N_channels)]
    sig2_f = [xp.fft.rfft(zero_pad(sig2_t[i])) for i in range(N_channels)]
    N_t = len(zero_pad(sig1_t[0]))
    
    freq_np = xp.asnumpy(xp.fft.rfftfreq(N_t, dt))

    freq_np[0] = freq_np[1] 

    PSD = 2 * [xp.asarray(sensitivity_LWA(freq_np))]

    aa = xp.asarray([inner_prod(sig1_f[i],sig1_f[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
    bb = xp.asarray([inner_prod(sig2_f[i],sig2_f[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
    ab = xp.asarray([inner_prod(sig1_f[i],sig2_f[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])

    overlap = 0.5*xp.sum(ab/(np.sqrt(aa*bb)))
    mismatch = 1 - overlap
    return mismatch
##======================Likelihood and Posterior (change this)=====================

M = 1e6; mu = 10; a = 0.0; p0 = 8.58; e0 = 0.1; x_I0 = 1.0;
dist = 1.0; 

qS = np.pi/3 ; phiS = np.pi/8; qK = np.pi/5; phiK = np.pi/6; 

Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 2.0

delta_t = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

use_gpu = True
xp = cp

# define trajectory
func = "KerrEccentricEquatorial"
insp_kwargs = {
    "err": 1e-10,
    "DENSE_STEPPING": 0,
    "use_rk4": False,
    "func": func,
    }
# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}

## ===================== CHECK TRAJECTORY ====================
# 
traj = EMRIInspiral(func=func, inspiral_kwargs = insp_kwargs)  # Set up trajectory module, pn5 AAK

# Compute trajectory 
if a < 0:
    a *= -1.0 
    x_I0 *= -1.0


t_traj, *out = traj(M, mu, a, p0, e0, x_I0, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

print("Final value in semi-latus rectum", out[0][-1])
print("Separatrix is located at",get_separatrix(a,out[1][-1],1.0))

traj_args = [M, mu, a, out[1][0], x_I0]
index_of_p = 3

# Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 
p_new = get_p_at_t(
    traj,
    T,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=[5, 13],
)

print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", out[0][-1])
print("Separatrix : ", get_separatrix(a, out[1][-1], 1.0))

p0 = p_new

inspiral_kwargs_Kerr = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e4),
        "err": 1e-10,  # To be set within the class
        "use_rk4": False,
        "integrate_phases":True,
        'func': 'KerrEccentricEquatorial'
    }
# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": True,  # GPU is availabel for this type of summation
    "pad_output": True,
}
    

amplitude_kwargs = {
    "specific_spins":[a],
    "use_gpu": True
    }
Waveform_model_AAK = GenerateEMRIWaveform(
AAKWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
KerrAAKSummation, # Define the summation for the amplitudes
inspiral_kwargs=inspiral_kwargs_Kerr,
sum_kwargs=sum_kwargs,
use_gpu=use_gpu,
return_list=True,
frame="detector"
)
    

print("Building waveform model now")
Waveform_model_Kerr = GenerateEMRIWaveform(
KerrEquatorialEccentricWaveformBase, # Define the base waveform
EMRIInspiral, # Define the trajectory
AmpInterpKerrEqEcc, # Define the interpolation for the amplitudes
InterpolatedModeSum, # Define the type of summation
ModeSelector, # Define the type of mode selection
inspiral_kwargs=inspiral_kwargs_Kerr,
sum_kwargs=sum_kwargs,
amplitude_kwargs=amplitude_kwargs,
use_gpu=use_gpu,
return_list=True,
frame='detector'
)

## ============= USE THE LONG WAVELENGTH APPROXIMATION, VOMIT ================ ##
params_Kerr = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 
params_pn5 = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0] 

nmodes_AAK = 50
specific_modes = [(2,2,n) for n in range(-(nmodes_AAK -2),-1 + nmodes_AAK)]
# specific_modes = "all"
# specific_modes = [(2,2,-1)]

waveform_AAK = Waveform_model_AAK(*params_pn5, T = T, dt = delta_t, mich = False, nmodes = nmodes_AAK)
waveform_Kerr = Waveform_model_Kerr(*params_Kerr, T = T, dt = delta_t, mich = False, mode_selection=specific_modes) 

N_t = len(zero_pad(waveform_AAK[0]))

freq_bin_np = np.fft.rfftfreq(N_t, delta_t)
freq_bin_np[0] = freq_bin_np[1]

PSD = 2*[cp.asarray(sensitivity_LWA(freq_bin_np))]

N_channels = 2

waveform_AAK_fft = [xp.fft.rfft(zero_pad(waveform_AAK[i])) for i in range(N_channels)]
waveform_Kerr_fft = [xp.fft.rfft(zero_pad(waveform_Kerr[i])) for i in range(N_channels)]

SNR_Kerr = SNR_function(waveform_Kerr, delta_t, N_channels = 2)
SNR_AAK = SNR_function(waveform_AAK, delta_t, N_channels = 2)

print("Truth waveform, final SNR for Kerr = ",SNR_Kerr)
print("Truth waveform, final SNR for AAK = ",SNR_AAK)

aa = xp.asarray([inner_prod(waveform_AAK_fft[i],waveform_AAK_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
bb = xp.asarray([inner_prod(waveform_Kerr_fft[i],waveform_Kerr_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])
ab = xp.asarray([inner_prod(waveform_AAK_fft[i],waveform_Kerr_fft[i], N_t, delta_t, PSD[i]) for i in range(N_channels)])

overlap = 0.5*xp.sum(ab/(np.sqrt(aa*bb)))
mismatch = 1 - overlap

print('mismatch between waveforms', mismatch)

t = np.arange(0,N_t*delta_t, delta_t)
waveform_AAK_I_np = xp.asnumpy(waveform_AAK[0])
waveform_Kerr_I_np = xp.asnumpy(waveform_Kerr[0])

waveform_AAK_II_np = xp.asnumpy(waveform_AAK[1])
waveform_Kerr_II_np = xp.asnumpy(waveform_Kerr[1])

import matplotlib.pyplot as plt

plt.plot(t[0:800], waveform_AAK_I_np[0:800], label = "AAK")
plt.plot(t[0:800]+delta_t,waveform_Kerr_I_np[0:800], label = "Kerr")
plt.xlabel(r'Time [seconds]')
plt.ylabel(r'Waveform strain (channel I)')
plt.title("Start: (M, mu, a, e0, T) = (1e6, 10, {}, {}, {})".format(a, e0, T))
plt.legend(fontsize = 16)
plt.savefig("waveform_plots/Frames/waveform_plot_start_plus.pdf",bbox_inches='tight')
plt.clf()

plt.plot(t[-100:], waveform_AAK_I_np[-100:], label = "AAK")
plt.plot(t[-100:]+delta_t,waveform_Kerr_I_np[-100:], label = "Kerr")
plt.xlabel(r'Time [seconds]')
plt.ylabel(r'Waveform strain (channel II)')
plt.title("End: (M, mu, a, p0, e0) = (1e6, 10, 0.3,  {}, {}, {})".format(a, e0, T))
plt.legend(fontsize = 16)
plt.savefig("waveform_plots/Frames/waveform_plot_end_plus.pdf",bbox_inches='tight')
plt.clf()


## ================== Try to average over angles ================ ##

print("Now averaging over angles")
qS_vec = np.random.uniform(0,np.pi,10) ; 
phiS_vec = np.random.uniform(0,2*np.pi,10) 
qK_vec = np.random.uniform(0, np.pi, 10)
phiK_vec = np.random.uniform(0, 2*np.pi,10)

delta_t = 10.0
from tqdm import tqdm as tqdm
mm_vec = []

for qS, phiS, qK, phiK in zip(qS_vec, phiS_vec, qK_vec, phiK_vec):
    params = [M,mu,a,p0,e0,1.0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]  

    waveform_AAK = Waveform_model_AAK(*params, T = T, dt = delta_t, mich = False, nmodes = nmodes_AAK)
    waveform_Kerr = Waveform_model_Kerr(*params, T = T, dt = delta_t, mich = False, mode_selection=specific_modes) 

    mm_vec.append(xp.asnumpy(mismatch_func(waveform_Kerr,waveform_AAK,delta_t,N_channels = 2)))

print("Average mismatch is",np.mean(mm_vec))
breakpoint()
