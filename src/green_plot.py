import matplotlib.pyplot as plt
import numpy as np 
from paper_plots import plot_hp_convergence
from scipy.stats import linregress

def add_trend_line(IDs: list[int], xs, ys, errors, k, hs, ax):
    for ID, x, y in zip(IDs, xs, ys):
        result = linregress(np.log(k*hs), np.log(errors[:, ID]))
        m = result.slope
        n = result.intercept
        ax.plot(k*hs, 0.5*np.exp(n)*(k*hs)**m, '--k')
        ax.text(x,y,f'$\\propto (kh)^{{{m : .1f}}}$')


kappa_e = 8.
N_modes = 15
H = 1.

data = np.load('./green_original.npz')
errors = data['errors']
hs = data['hs']
N_ths = data['N_ths']



ax = plot_hp_convergence(errors=errors, hs=hs, N_ths=N_ths, kappa_e=kappa_e, N_modes=N_modes, H=H)
add_trend_line(IDs=[2,5],xs = [1,1.8], ys=[2E-3, 1E-7], errors=errors, k=kappa_e, hs=hs, ax = ax[1] )
plt.savefig('./with_slopes_correct.pdf')