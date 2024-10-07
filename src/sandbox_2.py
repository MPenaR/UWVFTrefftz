# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: UWVFTrefftz--hy3v2Qt
#     language: python
#     name: python3
# ---

# %%
from domains import Waveguide, ScattererShape, ScattererType

# %%
domain = Waveguide(H=1,R=2)

# %%
domain.add_scatterer(scatterer_shape=ScattererShape.DIAMOND, scatterer_type=ScattererType.SOUND_SOFT, params=([0,0.5],1,0.05))
domain.generate_mesh(h_max=0.2)
domain.plot_mesh()

# %%

# %%
