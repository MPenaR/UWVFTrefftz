# %%
import numpy as np
from paper_plots import plot_hp_convergence
import matplotlib.pyplot as plt

# %%
saved_file = np.load('fundamental_outside8.npz')

# %%
errors = saved_file["errors"]
hs = saved_file["hs"]
N_ths = saved_file["N_ths"]

# %%
H = 1.
kappa_e = 8
N_modes = 15
title = f'Fundamental solution as incident field with no scatterer.\n $\\kappa={kappa_e}$, $H = {H:.0f}$, $R = \\lambda$, $M={N_modes}$ modes for de NtD map'


# %%
plot_hp_convergence(errors=errors,hs=hs, N_ths=N_ths, kappa_e=kappa_e,N_modes=N_modes,H=H, title=title, filename=f'fundamental_ouside_{int(kappa_e)}.pdf')
plt.show()
