import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

import numpy as np


inches_per_dot = 1/72.27
cm2inch = 1/2.54 # inch per cm
columnwidth = 630.185 * inches_per_dot
columnwidth = 469.75502 * inches_per_dot
#columnwidth = 524.07272*inches_per_dot

columnwidth=columnwidth*0.9

left_margin = 3. * cm2inch # cm
right_margin = 1.*cm2inch  # cm
figure_width = columnwidth # cm
figure_height = columnwidth/1. # cm
top_margin = 2.*cm2inch    # cm
bottom_margin = 1.5*cm2inch # cm

box_width = left_margin + figure_width + right_margin   # cm
box_height = top_margin + figure_height + bottom_margin # cm


from matplotlib import ticker as mticker
def plot_hp_convergence(errors, hs, N_ths, kappa_e, N_modes, H, title = None, filename=None):

    fig, ax = plt.subplots(nrows=2,figsize=(box_width,box_height))

    f = mticker.ScalarFormatter(useMathText=True, useOffset=False)

    for err, h in zip(errors,hs):
        ax[0].semilogy(N_ths,err,'.-', label=f'$kh = { f.format_data(float(f'{kappa_e*h: .2e}'))}$')


    if title:
        fig.suptitle(title)

    ax[0].set_xlabel('Number of basis functions per element ($N_p$)')
    # ax[0].set_ylabel('$\\left\\Vert u - u_h\\right\\Vert_2^2 \\, / \\, \\left\\Vert u \\right\\Vert_2^2$')
    ax[0].set_ylabel('Relative $L_2$ error')
    ax[0].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax[0].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax[0].set_xticks(range(3,17,2))
    ax[0].grid(True,which="major",ls='--')
    ax[0].legend()


    for err, N_th in zip(errors.transpose(),N_ths):
        ax[1].loglog(kappa_e*hs,err,'.-', label=f'$N_P = {N_th}$')

    ax[1].set_xlabel('Scaled mesh parameter ($kh$)')
    # ax[1].set_ylabel('$\\left\\Vert u - u_h\\right\\Vert_2^2 \\, / \\, \\left\\Vert u \\right\\Vert_2^2$')
    ax[1].set_ylabel('Relative $L_2$ error')
    ax[1].legend(loc="lower right")
    ax[1].yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax[1].yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

    plt.grid(True,which="major",ls='--')


    fig.subplots_adjust(left   = left_margin / box_width,
                        bottom = bottom_margin / box_height,
                        right  = 1. - right_margin / box_width,
                        top    = 1. - top_margin   / box_height)

    if filename:
        plt.savefig(filename)
    return ax
