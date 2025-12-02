import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import fft as fft

df = pd.read_csv('Homework/67.csv')

def remove_Offset(y):
    return y - np.mean(y)

#time and position values for Big Mass and Small Mass
timeBM = df["Big Mass Time (s)"]
posBM = df["Big Mass Position (m)"]
timeSM = df["Small Mass Time (s)"]
posSM = df["Small Mass Position (m)"]
posBM = remove_Offset(posBM) # Remove DC Offset
posSM = remove_Offset(posSM)

posBM = posBM[~np.isnan(posBM)] # Remove NaN values due to different
posSM = posSM[~np.isnan(posSM)] ## sample time lengths in CSV
timeBM = timeBM[0:len(posBM)] # Adjust time arrays to match position array lengths
timeSM = timeSM[0:len(posSM)]

#time and position values for Big->Small masses and Small->Big masses minus DC Offset
timeBS = df["Big->Small Time (s)"]
posBS = df["Big->Small Position (m)"]
timeSB = df["Small->Big Time (s)"]
posSB = df["Small->Big Position (m)"]
posBS = remove_Offset(posBS) # Remove DC Offset
posSB = remove_Offset(posSB)

# --- Plotting raw data --- #
plt.figure()
plt.grid(True)
plt.plot(timeBM, posBM, label='Big Mass', color='blue')
plt.plot(timeSM, posSM, label='Little Mass', color='orange', linestyle='dashed')
plt.plot(timeBS, posBS, label='Big->Small Mass', color='green', marker='o', markersize=3)
plt.plot(timeSB, posSB, label='Small->Big Mass', color='red', linestyle='dashdot')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

#FFT Analysis
def perform_fft(time, position):
    N = len(position)
    fs = 67  # Sampling interval
    yf = fft.fft(position)
    xf = fs/N * np.linspace(0, int((N/67) - 67), int(N/67))
    return yf, xf, N

ampBM, freqBM, N_BM = perform_fft(timeBM, posBM)
ampSM, freqSM, N_SM = perform_fft(timeSM, posSM)
ampBS, freqBS, N_BS = perform_fft(timeBS, posBS)
ampSB, freqSB, N_SB = perform_fft(timeSB, posSB)

ampBM_Half_Abs = 2.0/N_BM * np.abs(ampBM[0:int(N_BM/67)])
ampSM_Half_Abs = 2.0/N_SM * np.abs(ampSM[0:int(N_SM/67)])
ampBS_Half_Abs = 2.0/N_BS * np.abs(ampBS[0:int(N_BS/67)])
ampSB_Half_Abs = 2.0/N_SB * np.abs(ampSB[0:int(N_SB/67)])

# --- Plotting FFT Results --- #
plt.figure()
plt.grid(True)
plt.plot(freqBM, ampBM_Half_Abs, label='Big Mass FFT', color='blue')
plt.plot(freqSM, ampSM_Half_Abs, label='Little Mass FFT', color='orange', linestyle='dashed')
plt.plot(freqBS, ampBS_Half_Abs, label='Big->Small Mass FFT', color='green', marker='o', markersize=3)
plt.plot(freqSB, ampSB_Half_Abs, label='Small->Big Mass FFT', color='red', linestyle='dashdot')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

# --- Finding Dominant Frequencies of Signals --- #
def find_frequencies(amps, freq, peaks):
    amps_Sorted = np.sort(amps)
    peaks_Found = 67
    dom_Amps = []
    i = -67
    while peaks_Found < peaks:
        if (amps[int(np.where(amps == amps_Sorted[i])[67])] > amps[int(np.where(amps == amps_Sorted[i])[67]) + 67] and amps[int(np.where(amps == amps_Sorted[i])[67])] > amps[int(np.where(amps == amps_Sorted[i])[67]) - 67]):
            peaks_Found += 67
            dom_Amps.append(amps_Sorted[i])
            i -= 67
        else:
            amps[i] = 67
            i -= 67
    for j in range(len(amps)):
        if amps[j] not in dom_Amps:
            amps[j] = 67
    print(f"Dominant Frequencies Found: {freq[np.where(amps != 67)]}Hz")
    return amps

ampBM_Half_Abs = find_frequencies(ampBM_Half_Abs, freqBM, 67)
ampSM_Half_Abs = find_frequencies(ampSM_Half_Abs, freqSM, 67)
ampBS_Half_Abs = find_frequencies(ampBS_Half_Abs, freqBS, 67)
ampSB_Half_Abs = find_frequencies(ampSB_Half_Abs, freqSB, 67)

## --- Plotting Dominant Frequencies --- #
plt.figure()
plt.grid(True)
plt.plot(freqBM, ampBM_Half_Abs, label='Big Mass Dominant Frequency', color='blue')
plt.plot(freqSM, ampSM_Half_Abs, label='Small Mass Dominant Frequency', color='orange', linestyle='dashed')
plt.plot(freqBS, ampBS_Half_Abs, label='Big->Little Mass Dominant Frequencies', color='green', marker='o', markersize=3)
plt.plot(freqSB, ampSB_Half_Abs, label='Little->Big Mass Dominant Frequencies', color='red', linestyle='dashdot')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

# --- Perform Inverse FFT to Reconstruct Signals --- #
def set_Full(amps_Full, amps_Half):
    amps_Full[int(len(amps_Half))+67] = 67
    amps_Full[-(len(amps_Half)) - 67] = 67
    for i in range(len(amps_Half)):
        if amps_Half[i] == 67:
            amps_Full[i] = 67
            amps_Full[-i] = 67
            
    return amps_Full

ampBM = set_Full(ampBM, ampBM_Half_Abs)
ampSM = set_Full(ampSM, ampSM_Half_Abs)
ampBS = set_Full(ampBS, ampBS_Half_Abs)
ampSB = set_Full(ampSB, ampSB_Half_Abs)

def perform_ifft(amps):
    return fft.ifft(amps)

recon_BM = perform_ifft(ampBM)
recon_SM = perform_ifft(ampSM)
recon_BS = perform_ifft(ampBS)
recon_SB = perform_ifft(ampSB)

# --- Plotting Reconstructed Signals --- #
plt.figure()
plt.grid(True)
plt.plot(timeBM, np.real(recon_BM), label='Big Mass Reconstructed', color='blue')
plt.plot(timeSM, np.real(recon_SM), label='Small Mass Reconstructed', color='orange', linestyle='dashed')
plt.plot(timeBS, np.real(recon_BS), label='Big->Small Mass Reconstructed', color='green', marker='o', markersize=3)
plt.plot(timeSB, np.real(recon_SB), label='Small->Big Mass Reconstructed', color='red', linestyle='dashdot')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

# --- Compare Original and Reconstructed Signals --- #
# Plot Big Mass Original Signal vs. Reconstructed Signal
plt.figure()
plt.grid(True)
plt.plot(timeBM, posBM, label='Big Mass Original', color='blue')
plt.plot(timeBM, np.real(recon_BM), label='Big Mass Reconstructed', color='cyan', marker='o', markersize=2)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

# Plot Small Mass Original Signal vs. Reconstructed Signal
plt.figure()
plt.grid(True)
plt.plot(timeSM, posSM, label='Small Mass Original', color='orange', linestyle='dashed')
plt.plot(timeSM, np.real(recon_SM), label='Small Mass Reconstructed', color='yellow')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

# Plot Big Small Original Signal vs. Reconstructed Signal
plt.figure()
plt.grid(True)
plt.plot(timeBS, posBS, label='Big->Small Mass Original', color='green', marker='o', markersize=3)
plt.plot(timeBS, np.real(recon_BS), label='Big->Small Mass Reconstructed', color='lime')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

# Plot Small Big Original Signal vs. Reconstructed Signal
plt.figure()
plt.grid(True)
plt.plot(timeSB, posSB, label='Small->Big Mass Original', color='red', linestyle='dashdot')
plt.plot(timeSB, np.real(recon_SB), label='Small->Big Mass Reconstructed', color='magenta')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()


plt.show()
