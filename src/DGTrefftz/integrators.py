import numpy as np
from numpy.typing import NDArray
from itertools import permutations

int_array = NDArray[np.int32]
float_array = NDArray[np.float64]


def fekete3(f, r_A=(0,0), r_B=(1,0), r_C=(0,1)):
    ABx = r_B[0] - r_A[0]
    ABy = r_B[1] - r_A[1]
    ACx = r_C[0] - r_A[0]
    ACy = r_C[1] - r_A[1]
    S = 0.5 * np.abs(ABx*ACy - ABy*ACx)
    J = S/2.
    points = np.array([[1./3, 1./3, 1./3],
                    [0., 0., 1.],
                    [0., 1., 0.],
                    [1., 0., 0.],
                    [0., 0.2763932023, 0.7236067977],
                    [0.7236067977, 0.2763932023, 0.],
                    [0.7236067977, 0., 0.2763932023],
                    [0., 0.7236067977, 0.2763932023],
                    [0.2763932023, 0.7236067977, 0.],
                    [0.2763932023, 0., 0.7236067977]]) 
    
    x = r_A[0]*points[:,0] + r_B[0]*points[:,1] + r_C[0]*points[:,2]
    y = r_A[1]*points[:,0] + r_B[1]*points[:,1] + r_C[1]*points[:,2]
    
    w = np.array([ 0.9, 0.1/3, 0.1/3, 0.1/3, 1./6, 1./6, 1./6, 1./6, 1./6, 1./6 ])
    z = f(x,y)
    I = np.sum(z*w)
    return J*I 





def parse_one_line( line : str) -> tuple[int_array, float_array, float_array]:

    lines = line.split("orbit=")[1:]
    
    N = len(lines)

    w = np.zeros(N, dtype=np.float64)
    orbits = np.zeros(N, dtype=np.int32)
    xy = np.zeros((N,2), dtype=np.float64)
    
    for i, line in enumerate(lines):
        start, end = line.split("w=")
        orbits[i] = int(start.split(" ")[0])
        xy[i,:] = [ float(start.split(" ")[1]), float(start.split(" ")[2])]
        w[i] = float(end)

    return orbits, xy, w

def z_coordinate(x : float_array, y: float_array) -> float_array:
    z = 1 - x - y

    return np.column_stack((x,y,z))

def expand_orbit( x: float, y:float, z:float, orbit = None) -> float_array:
    if orbit == 1:
        return np.array([[x,y,z]])
    if orbit == 3:
        xyz = np.array([x,y,z])
        d = np.array([np.abs(z-y), np.abs(x-z),np.abs(y-x)])
        y = xyz[np.argmin(d)]
        x = xyz[(np.argmin(d) + 1) % 3]
        return np.array([[x,x,y], 
                         [x,y,x],
                         [y,x,x]])
    if orbit == 6:
        return np.array(list(permutations((x,y,z))))

def weights_and_points( xy : float_array, orbits: int_array, w: float_array) -> tuple[float_array, float_array]:
    N = orbits.sum()
    w = w.repeat(repeats=orbits)
    xyz = z_coordinate(x=xy[:,0], y=xy[:,1])
    points = np.vstack([expand_orbit(x=xyz[i,0], y=xyz[i,1], z=xyz[i,2], orbit=orbits[i]) for i in range(xyz.shape[0])])
    return points, w

d3 = 'orbit=1 0.3333333333 0.3333333333 w= 0.9000000000orbit=3 0.0000000000 0.0000000000 w= 0.0333333333orbit=6 0.0000000000 0.2763932023 w= 0.1666666667'
d6 = 'orbit=1 0.3333333333 0.3333333333 w= 0.2178563571orbit=3 0.1063354684 0.1063354684 w= 0.1104193374orbit=3 0.0000000000 0.5000000002 w= 0.0358939762orbit=3 0.0000000000 0.0000000000 w= 0.0004021278orbit=6 0.1171809171 0.3162697959 w= 0.1771348660orbit=6 0.0000000000 0.2655651402 w= 0.0272344079orbit=6 0.0000000000 0.0848854223 w= 0.0192969460'
d9 = 'orbit=1 0.3333333333 0.3333333333 w= 0.1096011288orbit=3 0.1704318201 0.1704318201 w= 0.0767491008orbit=3 0.0600824712 0.4699587644 w= 0.0646677819orbit=3 0.0489345696 0.0489345696 w= 0.0276211659orbit=3 0.0000000000 0.0000000000 w= 0.0013925011orbit=6 0.1784337588 0.3252434900 w= 0.0933486453orbit=6 0.0588564879 0.3010242110 w= 0.0619010169orbit=6 0.0551758079 0.1543901944 w= 0.0437466450orbit=6 0.0000000000 0.4173602935 w= 0.0114553907orbit=6 0.0000000000 0.2610371960 w= 0.0093115568orbit=6 0.0000000000 0.1306129092 w= 0.0078421987orbit=6 0.0000000000 0.0402330070 w= 0.0022457501'

fekete_points_weights = {}

orbits, xy, w = parse_one_line(d3)
fekete_points_weights["3"] = points, w = weights_and_points(xy=xy, orbits=orbits,w=w)

orbits, xy, w = parse_one_line(d6)
fekete_points_weights["6"] = points, w = weights_and_points(xy=xy, orbits=orbits,w=w)

orbits, xy, w = parse_one_line(d9)
fekete_points_weights["9"] = points, w = weights_and_points(xy=xy, orbits=orbits,w=w)


def fekete_integrator(f, r_A=(0,0), r_B=(1,0), r_C=(0,1), level = "3"):
    ABx = r_B[0] - r_A[0]
    ABy = r_B[1] - r_A[1]
    ACx = r_C[0] - r_A[0]
    ACy = r_C[1] - r_A[1]
    S = 0.5 * np.abs(ABx*ACy - ABy*ACx)
    J = S/2.
    points, w = fekete_points_weights[level]
    x = r_A[0]*points[:,0] + r_B[0]*points[:,1] + r_C[0]*points[:,2]
    y = r_A[1]*points[:,0] + r_B[1]*points[:,1] + r_C[1]*points[:,2]
    
    z = f(x,y)
    I = np.sum(z*w)
    return J*I 



def vec_fekete3(f, r_A=(0,0), r_B=(1,0), r_C=(0,1)):
    ABx = r_B[0] - r_A[0]
    ABy = r_B[1] - r_A[1]
    ACx = r_C[0] - r_A[0]
    ACy = r_C[1] - r_A[1]
    S = 0.5 * np.abs(ABx*ACy - ABy*ACx)
    J = S/2.
    points = np.array([[1./3, 1./3, 1./3],
                    [0., 0., 1.],
                    [0., 1., 0.],
                    [1., 0., 0.],
                    [0., 0.2763932023, 0.7236067977],
                    [0.7236067977, 0.2763932023, 0.],
                    [0.7236067977, 0., 0.2763932023],
                    [0., 0.7236067977, 0.2763932023],
                    [0.2763932023, 0.7236067977, 0.],
                    [0.2763932023, 0., 0.7236067977]]) 
    
    x = r_A[0]*points[:,0] + r_B[0]*points[:,1] + r_C[0]*points[:,2]
    y = r_A[1]*points[:,0] + r_B[1]*points[:,1] + r_C[1]*points[:,2]
    
    w = np.array([ 0.9, 0.1/3, 0.1/3, 0.1/3, 1./6, 1./6, 1./6, 1./6, 1./6, 1./6 ])
    z = f(x,y) # N x N x N_points
    I = np.sum(z*w, axis=-1)
    return J*I 


def vec_fekete_integrator(f, r_A=(0,0), r_B=(1,0), r_C=(0,1), level='3'):
    ABx = r_B[0] - r_A[0]
    ABy = r_B[1] - r_A[1]
    ACx = r_C[0] - r_A[0]
    ACy = r_C[1] - r_A[1]
    S = 0.5 * np.abs(ABx*ACy - ABy*ACx)
    J = S/2.

    points, w = fekete_points_weights[level]

    
    x = r_A[0]*points[:,0] + r_B[0]*points[:,1] + r_C[0]*points[:,2]
    y = r_A[1]*points[:,0] + r_B[1]*points[:,1] + r_C[1]*points[:,2]
    
    z = f(x,y) # N x N x N_points
    I = np.sum(z*w, axis=-1)
    return J*I 














if __name__ == "__main__":
    print(f"{fekete_integrator(lambda x, y : 1, level="3")=}")
    print(f"{fekete_integrator(lambda x, y : 1, level="6")=}")
    print(f"{fekete_integrator(lambda x, y : 1, level="9")=}")