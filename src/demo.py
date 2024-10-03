# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: UWVFTrefftz--hy3v2Qt
#     language: python
#     name: python3
# ---

# All the meshing functions can be found in the `meshes.py` package. Right now there are three meshes defined: 
# - `testMesh` returns a rectangular mesh without scatterer.
# - `waveguideMesh` returns a full mesh of a waveguide with a circular scatterer.
# - `toyMesh` returns a square mesh with only two triangular elements.

from meshes import testMesh


Omega = testMesh()


