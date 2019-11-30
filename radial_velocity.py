import radial_velocity_functions as rf
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc
from detect_peaks import detect_peaks
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

# rf.phaseplot(v_pmax_sort, bjd_sort, v_pmin_sort, bjd_sort, pguess=pguess)
# plt.close()

# # Sine fitting to data

# Prepare fit x axis data and initial guesses
freq_guess = 1/pguess
t_pmax = bjd_sort-bjd_sort[0]
t_pmin = bjd_sort-bjd_sort[0]
system_peaks = np.append(v_pmax_sort, v_pmin_sort)
system_mean_guess = np.mean(system_peaks)

# Make fit model and set initial values
print('')
smodel = lmfit.Model(rf.sine_const)
print('parameter names: {}'.format(smodel.param_names))
params = smodel.make_params(a=50, freq=freq_guess, phase=1, const=system_mean_guess)
params['a'].set(min=0, max=100)
params['freq'].set(min=0.1*freq_guess, max=3*freq_guess)
params['phase'].set(min=0, max=2*np.pi)
params['const'].set(min=-200, max=0)
print('')

# Fit sine model to one star
params['freq'].set(value=freq_guess, vary=True)
params['phase'].set(value=1, vary=True)
fit1 = smodel.fit(v_pmin_sort, params, x=t_pmin)
fit1_vals = fit1.best_values
fit1_a, fit1_freq, fit1_phase, fit1_const = fit1_vals['a'], fit1_vals['freq'], fit1_vals['phase'], fit1_vals['const']
print('fit1 report')
print(fit1.fit_report(show_correl=False))
print('')
print('')

# Fit sine model to other star using restrictions previous fit
params['freq'].set(value=fit1_freq, vary=False)  # Make cyclic frequency the same as first fit
params['phase'].set(value=fit1_phase + np.pi, vary=False)  # Make phase exactly opposite the first fit
fit2 = smodel.fit(v_pmax_sort, params, x=t_pmax)
fit2_vals = fit2.best_values
fit2_a, fit2_freq, fit2_phase, fit2_const = fit2_vals['a'], fit2_vals['freq'], fit2_vals['phase'], fit2_vals['const']
print('fit2 report')
print(fit2.fit_report(show_correl=False))


# Construct fit curves
t_fit = np.linspace(bjd_sort[0], bjd_sort[-1], 500000) - bjd_sort[0]
fit1_line = rf.sine_const(t_fit, fit1_a, fit1_freq, fit1_phase, fit1_const)
fit2_line = rf.sine_const(t_fit, fit2_a, fit2_freq, fit2_phase, fit2_const)

# Plot fit with data
plt.figure()
plt.plot(t_pmax, v_pmax_sort, 'r*', t_fit, fit2_line, 'r--', linewidth=0.5)
plt.plot(t_pmin, v_pmin_sort, 'b*', t_fit, fit1_line, 'b--', linewidth=0.5)

crossings = detect_peaks((fit1_line - fit2_line) ** 2, valley=True)
plt.plot(t_fit[crossings], fit1_line[crossings], 'k*')
plt.plot(t_fit[crossings], fit2_line[crossings], 'k*')
print(' ')
print('Standard deviation from 0 at closest value crossings', np.std(((fit1_line-fit2_line)**2)[crossings]))
plt.show(block=True)

y_crossings = np.append(fit1_line[crossings], fit2_line[crossings])
system_mean = np.mean(y_crossings)
print('y_crossings')
print(y_crossings)
print('system mean velocity [km/s]', system_mean)
print('')

