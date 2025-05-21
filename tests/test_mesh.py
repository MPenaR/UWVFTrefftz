import numpy as np 
# from block_fluxes import SoundHard_block

# NP = 3
# d_1 = 0.5
# k = 0.8




element_dtype = [("face", "i4"), ("edges", "i4", (3)), ("points", "i4", (3))]

elements = np.array( [(0, [0, 1, 4], [0, 1, 2]),
                      (0, [4, 2, 3], [0, 2, 3])], dtype=element_dtype)

# A_wall = np.zeros([2*NP, 2*NP], dtype=np.complex128)


# theta_0 = 0
# theta = np.linspace(0,2*np.pi,NP) + theta_0
# d = np.column_stack([np.cos(theta), np.sin(theta)])

# d_d = np.zeros([NP,NP,2])
# d_d[:,:,0] = - np.subtract.outer(np.cos(theta), np.cos(theta))
# d_d[:,:,1] = - np.subtract.outer(np.sin(theta), np.sin(theta))
    
#     l = edge.l
#     N = edge.N
#     T = edge.T
#     M = edge.M


# A_wall[:NP,:NP] = SoundHard_block( k=k, edge=e0, d=d, d_d=d_d, d_1=d_1)
# A_wall[NP:,NP:] = SoundHard_block( k=k, edge=e2, d=d, d_d=d_d, d_1=d_1)
