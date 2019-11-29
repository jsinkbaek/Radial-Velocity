import radial_velocity_functions as rf
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc
from detect_peaks import detect_peaks
from scipy.optimize import curve_fit
import lmfit


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


indx_pmax = np.empty((len(corr[:, 0]) - 1,))
indx_pmin = np.empty((len(corr[:, 0]) - 1,))
v_pmax = np.empty((len(corr[:, 0])-1, ))
v_pmin = np.empty((len(corr[:, 0])-1, ))
pmax = np.empty((len(corr[:, 0])-1, ))
pmin = np.empty((len(corr[:, 0])-1, ))
indx_pmax[:] = np.nan
indx_pmin[:] = np.nan
v_pmax[:] = np.nan
v_pmin[:] = np.nan
pmax[:] = np.nan
pmin[:] = np.nan


for i in range(0, len(indx_pmax)):
    indx_pmax[i] = np.argmax(corr[i, peak_indx[i]])
    v_pmax[i] = v_shift[peak_indx[i]][int(indx_pmax[i])]
    pmax[i] = corr[i, peak_indx[i][int(indx_pmax[i])]]
    if len(corr[i, peak_indx[i]]) == 1:
        indx_pmin[i] = indx_pmax[i]
    else:
        indx_pmin[i] = np.argmin(corr[i, peak_indx[i]])
    v_pmin[i] = v_shift[peak_indx[i]][int(indx_pmin[i])]
    pmin[i] = corr[i, peak_indx[i][int(indx_pmin[i])]]

plt.figure()
for i in range(0, len(indx_pmax)):
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


# # Phaseplot
pguess = np.mean([56524.9-56513.8, 56524.6-56513.7])
freq_guess = 1/pguess
# rf.phaseplot(v_pmax_sort, bjd_sort, v_pmin_sort, bjd_sort, pguess=pguess)
# plt.close()
t_pmax = bjd_sort-bjd_sort[0]
t_pmin = bjd_sort-bjd_sort[0]
# t_pmin = np.delete(bjd_sort, np.argwhere(np.isnan(v_pmin_sort))) - bjd_sort[0]
# v_pmin_sort = np.delete(v_pmin_sort, np.argwhere(np.isnan(v_pmin_sort)))
system_peaks = np.append(v_pmax_sort, v_pmin_sort)
system_mean = np.mean(system_peaks)


smodel = lmfit.Model(rf.sine_dstar)
print('parameter names: {}'.format(smodel.param_names))
print('independent variables: {}'.format(smodel.independent_vars))
params = smodel.make_params(a=50, b=50, freq=freq_guess, phase=1, const=system_mean)
params['a'].set(min=0, max=100)
params['b'].set(min=0, max=100)
params['freq'].set(min=0.5*freq_guess, max=2*freq_guess)
params['phase'].set(min=-np.pi, max=np.pi)
params['const'].set(min=-100, max=0)
# print(params)
# print(params['freq'])

# fitres_max = smodel.fit(v_pmax_sort, x=t_pmax, a=50, freq=freq_guess, phase=1, const=system_mean)
# popt_pmax = fitres_max.best_values
# fitres_min = smodel.fit(v_pmin_sort, x=t_pmin, a=50, freq=freq_guess, phase=1, const=system_mean)
# popt_pmin = fitres_min.best_values
fitres = smodel.fit(v_pmax_sort+v_pmin_sort, params, x=t_pmax)
fit_vals = fitres.best_values
fit_a, fit_b, fit_freq, fit_phase, fit_const = fit_vals['a'], fit_vals['b'], fit_vals['freq'], fit_vals['phase'], \
                                               fit_vals['const']
print(fit_vals)
print(fit_a, fit_b)

plt.figure()
t_fit = np.linspace(bjd_sort[0], bjd_sort[-1], 500000) - bjd_sort[0]
fit_pmax = rf.sine_const(t_fit, fit_a, fit_freq, fit_phase, fit_const)
fit_pmin = rf.sine_const(t_fit, fit_b, fit_freq, fit_phase+np.pi, fit_const)
plt.plot(t_pmax, v_pmax_sort, 'r*', t_fit, fit_pmax, 'r--', linewidth=0.5)
plt.plot(t_pmin, v_pmin_sort, 'b*', t_fit, fit_pmin, 'b--', linewidth=0.5)
fit_tot = rf.sine_dstar(t_fit, fit_a, fit_b, fit_freq, fit_phase, fit_const)

crossings = detect_peaks((fit_pmax-fit_pmin)**2, valley=True)
# plt.plot(t_fit[crossings], fit_pmax[crossings], 'k*')
# plt.plot(t_fit[crossings], fit_pmin[crossings], 'k*')
# print(((fit_pmax-fit_pmin)**2)[crossings])
# print(np.std(((fit_pmax-fit_pmin)**2)[crossings]))
plt.show(block=False)

plt.figure()
plt.plot(t_pmax, v_pmax_sort+v_pmin_sort, '*')
plt.plot(t_fit, fit_tot)
plt.show(block=True)
