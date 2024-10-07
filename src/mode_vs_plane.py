# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Arc


# %%
H = 1
R = 10
Ny = 20
Nx = (R//H)*Ny

x = np.linspace(-2*R,0,Nx)
y = np.linspace(-H,H,Ny)
y_w = np.linspace(-2*H,2*H,Ny)
X_w, Y_w = np.meshgrid(x,y_w) 
X, Y = np.meshgrid(x,y)

k = 5.
l = 2*np.pi/k


alfa = np.arcsin(l/2*H)

# Z = np.exp(1j*k*X)

d_up = np.array( [np.cos(alfa), np.sin(alfa)])
d_down = np.array( [np.cos(alfa), -np.sin(alfa)])

Z_up = np.exp(1j*k*(d_up[0]*X+d_up[1]*Y))
Z_up_w = np.exp(1j*k*(d_up[0]*X_w+d_up[1]*Y_w))

Z_down = np.exp(1j*k*(d_down[0]*X+d_down[1]*Y))
Z_down_w = np.exp(1j*k*(d_down[0]*X_w+d_down[1]*Y_w))

Z_exact = np.cos(np.pi*Y/H)*np.exp(1j*(np.sqrt(k**2 - (np.pi/H)**2))*X)


# %%
def plot_field(Z,Z_w, d = None, ax=None, title=None,):
    if ax is None:
        _, ax = plt.subplots()
    
    ax.imshow(np.real(Z_w), origin='lower', extent=[-2*R,0,-2*H,2*H], alpha=0.6, interpolation='bicubic')
    ax.imshow(np.real(Z), origin='lower', extent=[-2*R,0,-H,H], interpolation='bicubic' )
    ax.set_ylim([-2*H,2*H])
    ax.hlines([-H,H],-2*R,0,colors='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if title:
        ax.set_title(title)
    
    if d is not None:
        A = np.array([-R,0.])
        B = A + d*l
        edge_width=1
        arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(2*edge_width, 3*edge_width, edge_width)
        arrow = FancyArrowPatch(posA=A, posB=B, arrowstyle=arrowstyle, color='w')
        ax.add_artist(arrow)


def plot_restriction(f_w,t_w,f,y, ax):
    ax.plot(f_w,y_w,'b',alpha=0.6)
    ax.plot(f,y,'b')
    ax.hlines([-H,H],xmin=-1.5,xmax=1.5,colors='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')



def make_figure(n,t, w, Z_up, Z_up_w, Z_down, Z_down_w, folder=None):

    Z_up = np.exp(-1j*w*t)*Z_up
    Z_up_w = np.exp(-1j*w*t)*Z_up_w
    Z_down = np.exp(-1j*w*t)*Z_down
    Z_down_w = np.exp(-1j*w*t)*Z_down_w

    fig = plt.figure(figsize=(10,6))

    gs = fig.add_gridspec(nrows=3,ncols=2, width_ratios=(8,1))


    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[2,0])

    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,1])


    plot_field(Z_up,Z_up_w, d_up, ax=ax0)
    plot_field(Z_down,Z_down_w, d_down, ax=ax1)
    plot_field(0.5*(Z_up+Z_down),0.5*(Z_up_w+Z_down_w), ax=ax2)

    plot_restriction(np.real(Z_up_w[:,-1]),y_w, np.real(Z_up[:,-1]),y, ax3)
    plot_restriction(np.real(Z_down_w[:,-1]),y_w, np.real(Z_down[:,-1]),y, ax4)
    plot_restriction(0.5*np.real(Z_up_w[:,-1]+Z_down_w[:,-1]),y_w, 0.5*np.real(Z_up[:,-1]+Z_down[:,-1]),y, ax5)

    ax2.plot([-R,-R+l*d_up[0]],[-H,-H+l*d_up[1]],'w')
    ax2.vlines(-R,-H,H, colors='w')
    ax2.plot([-R,-R+l/np.sqrt(1-(l/(2*H)**2))],[H,-H],'w')
    ax2.plot([-R,-R+l/np.sqrt(1-(l/(2*H)**2))],[-H,-H],'w')
    ax2.text(-R-0.5,0,'$2H$', color='r', rotation=90)
    ax2.text(-R+0.5,-H-0.5,r"$\lambda'$", color='r')
    th = np.arcsin(l/(2*H))*180/np.pi
    ax2.text(-R+0.2,-0.5,r"$\lambda$", color='r',rotation=th)
    r = 1
    arc1 = Arc(xy=[-R,H],width=r,height=r,angle=-90.,theta1=0,theta2=th, edgecolor='w')
    ax2.add_patch(arc1)
    arc2 = Arc(xy=[-R,-H],width=r,height=r,angle=0.,theta1=0,theta2=th, edgecolor='w')
    ax2.add_patch(arc2)
    ax2.text(-R+0.1,0,r"$\alpha$", color='r')
    ax2.text(-R+0.6,-H+0.1,r"$\alpha$", color='r')
    if folder: 
        filename = f'frame_{n:03d}.png'
        print(f'saving file: {filename}')
        plt.savefig(f'{folder}{filename}')
        plt.close()

n = 0

T = 5
w = 4*2*np.pi/T


folder = './mode_vs_plane/'

for n,t in enumerate(np.linspace(0,T,100)):
    make_figure(n,t, w, Z_up, Z_up_w, Z_down, Z_down_w, folder)


# %%
Z_up.shape

# %%
