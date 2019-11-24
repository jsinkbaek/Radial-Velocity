import radial_velocity_functions as rf
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc

plt.rcParams.update({'font.size': 22})  # Changes font size on all plots

wlength, data, bjd, bar_rvc = rf.project1_txtload('filenames')

# # Interpolate datasets

# Wavelength interpolation steps build with equal velocity spacing
c = scc.c / 1000  # speed of light in km/s
wl_start, wl_end = wlength[-1, 0], wlength[-1, -1]
deltav = 0.25  # km/s
step_amnt = np.log10(wl_end/wl_start) / np.log10(1.0 + deltav/c) + 1
wl_intp = wl_start * (1.0 + deltav/c)**(np.linspace(1, step_amnt, step_amnt))

# Interpolate with new wavelength steps
data_intp = rf.interpol(wlength, data, wl_intp)


# # Cross correlate
corr = np.zeros(data_intp.shape)
wl_shift = wl_intp - wl_intp[len(wl_intp)//2]
for i in range(0, len(corr[:, 0])):
    corr[i, :] = rf.cross_correlate(data_intp[i, :], data_intp[-1, :], plot=False)[0]

plt.plot(wl_shift, corr[0, :])
plt.xlabel('Wavelength shift [Å]')
plt.ylabel('Normalized cross correlation')
plt.xlim([-10, 10])
plt.show()

# Velocity shift
v_shift = np.linspace(-step_amnt//2, step_amnt//2, step_amnt) * deltav

plt.plot(v_shift, corr[0, :])
plt.xlabel('Radial velocity shift [km/s]')
plt.ylabel('Normalized cross correlation')
plt.xlim([-250, 250])
plt.show()

