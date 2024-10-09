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

# %% [markdown]
# All the meshing functions can be found in the `meshes.py` package. Right now there are three meshes defined: 
# - `testMesh` returns a rectangular mesh without scatterer.
# - `waveguideMesh` returns a full mesh of a waveguide with a circular scatterer.
# - `toyMesh` returns a square mesh with only two triangular elements.

# %%
from meshes import testMesh


# %%
Omega = testMesh()

# %%
