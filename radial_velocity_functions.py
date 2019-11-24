import numpy as np
import os
import matplotlib.pyplot as plt


def project1_txtload(fileref):
    # Expects each file to have the same number of measurements
    with open(os.path.join('Datafiles', fileref+'.txt'), 'r') as f:
        references = []
        for line in f:
            references.append(line.split()[0])

    for i in range(0, len(references)):
        dat_temp = np.loadtxt(os.path.join('Datafiles', references[i]))
        if i == 0:
            data = np.zeros((len(references)+1, len(dat_temp[:, 1])))
            wlength = np.zeros((len(references)+1, len(dat_temp[:, 0])))

        data[i, :] = dat_temp[:, 1]
        wlength[i, :] = dat_temp[:, 0]

    # Solar reference
    data_temp = np.loadtxt(os.path.join('Datafiles', 'solar_template.txt'))
    data[-1, :] = data_temp[:, 1]
    wlength[-1, :] = data_temp[:, 0]

    data_temp = np.loadtxt(os.path.join('Datafiles', 'fileinfo.txt'))
    bjd = data_temp[:, 1]
    bar_rvc = data_temp[:, 2]
    return wlength, data, bjd, bar_rvc


def cross_correlate(a, b, spacing=None, plot=True):
    # Expects uniform and same spacing for a and b, or that spacing is calculated outside the function
    a = np.mean(a)-a
    b = np.mean(b)-b
    steps = None
    result = np.correlate(a, b, mode='same')
    result = result/np.max(result)

    if spacing is not None:
        steps = np.linspace(-len(result)//2 * spacing, len(result)//2 * spacing, len(result))
    if plot is True and spacing is not None:
        plt.plot(steps, result)
        plt.show()
    elif plot is True:
        plt.plot(result)
        plt.show()

    return result, steps


def interpol(xp, yp, x):
    y = np.zeros((len(yp[:, 0]), len(x)))
    for i in range(0, len(yp[:, 0])):
        y[i, :] = np.interp(x, xp[i, :], yp[i, :])
    return y

