import radial_velocity_functions as rf
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc
from detect_peaks import detect_peaks

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

plt.figure()
plt.plot(wl_shift, corr[0, :])
plt.xlabel('Wavelength shift [Å]')
plt.ylabel('Normalized cross correlation')
plt.xlim([-10, 10])
plt.show(block=False)

# Velocity shift
v_shift = np.linspace(-step_amnt//2, step_amnt//2, step_amnt) * deltav

peak_indx = []
for i in range(0, len(corr[:, 0])-1):
    peak_indx.append(detect_peaks(corr[i, :], mph=0.2))


indx_max = np.empty((len(corr[:, 0])-1, ))
indx_min = np.empty((len(corr[:, 0])-1, ))
v_pmax = np.empty((len(corr[:, 0])-1, ))
v_pmin = np.empty((len(corr[:, 0])-1, ))
pmax = np.empty((len(corr[:, 0])-1, ))
pmin = np.empty((len(corr[:, 0])-1, ))
indx_max[:] = np.nan
indx_min[:] = np.nan
v_pmax[:] = np.nan
v_pmin[:] = np.nan
pmax[:] = np.nan
pmin[:] = np.nan


for i in range(0, len(indx_max)):
    indx_max[i] = np.argmax(corr[i, peak_indx[i]])
    v_pmax[i] = v_shift[peak_indx[i]][int(indx_max[i])]
    pmax[i] = corr[i, peak_indx[i][int(indx_max[i])]]
    if len(corr[i, peak_indx[i]]) == 1:
        pass
    else:
        indx_min[i] = np.argmin(corr[i, peak_indx[i]])
        v_pmin[i] = v_shift[peak_indx[i]][int(indx_min[i])]
        pmin[i] = corr[i, peak_indx[i][int(indx_min[i])]]

plt.figure()
for i in range(0, len(indx_max)):
    plt.plot(v_shift, corr[i, :])
plt.xlabel('Radial velocity shift [km/s]')
plt.ylabel('Normalized cross correlation')
plt.plot(v_pmax, pmax, 'b*')
plt.plot(v_pmin, pmin, 'r*')
plt.xlim([-200, 100])
plt.show(block=False)

sort_indx = np.argsort(bjd)
bjd_sort = bjd[sort_indx]
v_pmax_sort = (v_pmax+bar_rvc)[sort_indx]
v_pmin_sort = (v_pmin+bar_rvc)[sort_indx]

plt.figure()
plt.plot(bjd_sort, v_pmax_sort, 'r*', bjd_sort, v_pmax_sort, 'r',
         bjd_sort, v_pmin_sort, 'b*', bjd_sort, v_pmin_sort, 'b')
plt.show(block=False)

v_ratio = v_pmax_sort / v_pmin_sort

plt.figure()
plt.plot(bjd_sort, v_ratio, 'b', bjd_sort, v_ratio, 'r*')
plt.show()
