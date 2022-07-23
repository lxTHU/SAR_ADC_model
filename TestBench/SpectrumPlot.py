#BSD 3-Clause License
#Copyright (c) 2018, lxTHU (LiXuan) 
#All rights reserved.

#%% 
#import 
import numpy as np
import matplotlib.pyplot as plt
# import mkl_fft as fft
import scipy.fftpack as fft


#%%
#ADC Spectrum by LiXuan, 2018 
def spectrum_singletone(Vout, Plot=True, RL=50,WINDOWWIDE=0,HDmax=13):
    Vout = np.array(Vout) #To nparray format
    # Vout = VinSHs-VresLSB
    VDfft = fft.fft(Vout)/(Vout.size) #FFT of Digital-domain output
    VDfftpow = (VDfft[1:]*VDfft[-1:0:-1]).real/RL #Calculate bins' power, remove DC
    MAXF = (Vout.size-1)//2 #Maxinum (exclude +-: nyquist) frequency
    VDfftpow = np.hstack(((VDfft[0]**2).real/RL, VDfftpow[0:MAXF]*2, VDfftpow[MAXF:-MAXF])) #[DC; p+n freq power; +-1 nyquist power] 
    FIN = np.argmax(VDfftpow[1:])+1 #Find main tone(exclude DC)

    FHDs = (Vout.size/2-abs((FIN*np.arange(HDmax).reshape(-1,1))%(Vout.size)-Vout.size/2)).astype(int)
    FHDsWD = FHDs+np.arange(-WINDOWWIDE,WINDOWWIDE+1,1)
    # print(FHDs,"\n",FHDsWD)
    FHDsWDremoveDCFIN = np.setdiff1d(FHDsWD, [[0], [FIN]]+np.arange(-WINDOWWIDE,WINDOWWIDE+1,1)) #HDs Mask w/o DC & FIN
    BinsDCFIN = np.setdiff1d(FHDsWD, FHDsWDremoveDCFIN) #Masks of DC & FIN
    BinsWoFHDsWD = np.setdiff1d(np.arange(VDfftpow.size), FHDsWD) #Noises Mask (w/o HDs & DC & FIN)
    # print(FHDsWD.remove(33))
    # print(FHDsWDremoveDCFIN)
    # print(BinsDCFIN)

    Pdc = VDfftpow[:WINDOWWIDE+1].sum() #Total DC power
    Pfin = VDfftpow[FIN-WINDOWWIDE:FIN+WINDOWWIDE+1].sum() #Total main tone power
    PHd = VDfftpow[FHDsWDremoveDCFIN].sum() #Total Harmonic Distortion power
    Pnoise = VDfftpow[BinsWoFHDsWD].sum() #Total Others: noise power
    # print(Pfin)

    VDfftpowWoDCFIN = np.copy(VDfftpow)
    VDfftpowWoDCFIN[BinsDCFIN] = 0
    FMAXHD = np.argmax(VDfftpowWoDCFIN) #Find max harmonics (exclude DC & FIN)
    PmaxHd = VDfftpowWoDCFIN[FMAXHD-WINDOWWIDE:FMAXHD+WINDOWWIDE+1].sum() #Max harmonics power
    # Phds = #Every harmonic distortion
    PHds = np.fromiter(map(np.sum, VDfftpow[FHDsWD]), float)

    SNDR = 10*np.log10(Pfin/(Pnoise+PHd))
    ENOB = (SNDR-1.76)/6.02
    SNR = 10*np.log10(Pfin/Pnoise)
    THD = 10*np.log10(PHd/Pfin)
    HDS = 10*np.log10(PHds/Pfin)
    SFDR = 10*np.log10(Pfin/PmaxHd)
    # print(ENOB, SNDR,SNR,THD,SFDR,FIN,HDs,VDfftpow)

    if Plot is True:
        with plt.style.context([]):#'science','no-latex']):
            # from matplotlib import mlab
            # plt.magnitude_spectrum(VinSHs-VresLSB, Fs=NumOfSARs, scale='dB', window=np.ones(VinSHs.size))
            # plt.psd(VinSHs-VresLSB, Fs=NumOfSARs, window=mlab.window_none)
            plt.figure(figsize=(6,4))
            plt.plot(VDfftpow, label='Vout') # +0.1e-13 To increase the spectrum MIN
        #     plt.plot(VDfftpow, label='Vout')
            plt.yscale('log')
            TicksY = (np.logspace(-6,-1,6).reshape(-1,1)*np.linspace(.2,1,9)).reshape(-1)**2
            plt.yticks(TicksY)
            plt.xlim(-1)
            plt.ylim(1e-14)
            plt.ylabel("Power (W)")
            plt.xlabel("Frequency (1/TotalSampleTime)")
            plt.legend()
            plt.grid()
            plt.text(MAXF*.35,Pfin*.5e-3,"ENOB={0:.2f}bits\nSNDR={1:.2f}dB\nSNR ={2:.2f}dB\nTHD ={3:.2f}dB\nSFDR={4:.2f}dB".format(ENOB,SNDR,SNR,THD,SFDR))
            plt.text(FIN-MAXF*.04,Pfin*.5e-10, " HDs:\n"+"".join(list(map("{0:2}:{1:7.2f}dBc\n".format, np.arange(HDmax), HDS))))
            plt.plot(FIN,Pfin,'r.')
            plt.annotate("{0:.2f}dB(@{1})".format(10*np.log10(Pfin),FIN),xy=(FIN,Pfin))
            plt.plot(FMAXHD,PmaxHd,'r.')
            plt.annotate("{0:.2f}dB(@{1})".format(10*np.log10(PmaxHd),FMAXHD),xy=(FMAXHD,PmaxHd))
            plt.show()
    return {'ENOB':ENOB, 'SNDR':SNDR, 'SNR':SNR, 'THD':THD, 'SFDR':SFDR, 'Pdc':Pdc, 'Pfin':Pfin, 'psd':VDfftpow, 'fft':VDfft, 'FIN':FIN}


print("Func. Ready")


#%%
#Generate Sine Wave 
NumOfSARs = 199
Freq=92/199
t = np.arange(NumOfSARs)
VinSHs = 1.000*np.cos(2*np.pi*t*Freq)
plt.plot(VinSHs)
VinSHs[3] += 7e-3
VinSHs[37] += -2e-3

#PlotSpectrum Test
spectrum = spectrum_singletone(VinSHs) #with Plot
