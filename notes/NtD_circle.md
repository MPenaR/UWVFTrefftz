# Neumann to Dirichlet operators and fluxes.

## Waveguide
### Dirichlet to Neumann
The right exterior problem in the waveguide:

$$
\begin{cases}
\Delta u + k^2u=0 & \text{in }(R,\infty)\times[0,H]\\
u = f & \text{on } x=R
\end{cases}
$$
admits a solution in terms of waveguide modes:

$$
u(\mathbf{x})=\frac{a_0}{\sqrt{2\pi}}e^{ikx}+\sum_{n=1}^\infty a_n\frac{e^{i\sqrt{k^2 - \left(\frac{n\pi}{H}\right)^2}x}}{\sqrt{\pi}} \cos\left(n\pi\frac{y}{H}\right)
$$

$$
u(\mathbf{x})=\frac{a_0}{\sqrt{2\pi}}e^{ikx}+\sum_{n=1}^\infty a_n\frac{e^{i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}kx}}{\sqrt{\pi}} \cos\left(n\pi\frac{y}{H}\right)
$$

If we impose the boundary condition:

$$
\frac{a_0}{\sqrt{2\pi}}e^{ikR}+\sum_{n=1}^\infty a_n\frac{e^{i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}kR}}{\sqrt{\pi}} \cos\left(n\pi\frac{y}{H}\right) = \frac{f_0}{\sqrt{2\pi}}+\sum_{n=1}^\infty \cos\left(n\pi\frac{y}{H}\right)\frac{f_n}{\sqrt{\pi}}
$$
where $f_n$, $n=0,1\dots$ are the cosine coeficcients of the function $f$. The solution is then: 
$$
u(\mathbf{x})=\frac{\int_0^Hf(\eta)\,\mathrm{d}\eta}{2\pi}e^{ik(x-R)}+\sum_{n=1}^\infty \frac{\int_0^H f(\eta)\cos(n\pi\frac{\eta}{H})\,\mathrm{d}\eta}{\pi} \cos\left(n\pi\frac{y}{H}\right)e^{i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}ki(x-R)}
$$

The right Dirichlet to Neumann map is then: 

$$
\mathrm{DtN}(f)=ik\frac{\int_0^Hf(\eta)\,\mathrm{d}\eta}{2\pi}+ik\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}\sum_{n=1}^\infty \frac{\int f(\eta)\cos(n\pi\frac{\eta}{H})\,\mathrm{d}\eta}{\pi} \cos\left(n\pi\frac{y}{H}\right)
$$

### Neumann to Dirichlet operator

In the same manner, for the problem

$$
\begin{cases}
\Delta u + k^2u=0 & \text{in }(R,\infty)\times[0,H]\\
\frac{\partial u}{\partial x} = f & \text{on } x=R
\end{cases}
$$

we also use the expansion:

$$
u(\mathbf{x})=\frac{a_0}{\sqrt{2\pi}}e^{ikx}+\sum_{n=1}^\infty a_n\frac{e^{i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}kx}}{\sqrt{\pi}} \cos\left(n\pi\frac{y}{H}\right)
$$

but now the Neumann boundary condition looks like:

$$
ik\frac{a_0}{\sqrt{2\pi}}e^{ikR}+\sum_{n=1}^\infty ik\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}a_n\frac{e^{i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}kR}}{\sqrt{\pi}} \cos\left(n\pi\frac{y}{H}\right) = \frac{f_0}{\sqrt{2\pi}}+ \sum_{n=1}^\infty f_n \frac{\cos\left(n\pi\frac{y}{H}\right)}{\sqrt{\pi}}
$$

so in this case the solution is:

$$
u(\mathbf{x})=\frac{1}{ik}\frac{\int_0^Hf(\eta)\,\mathrm{d}\eta}{2\pi}e^{ik(x-R)}+\frac{1}{ik}\sum_{n=1}^\infty \frac{1}{\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}} \frac{\int_0^H f(\eta)\cos(n\pi\frac{\eta}{H})\,\mathrm{d}\eta}{\pi} \cos\left(n\pi\frac{y}{H}\right)e^{i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}ki(x-R)}
$$

and the Neumann to Dirichlet operator is:

$$
\mathrm{NtD}(f)=\frac{1}{ik}\frac{\int_0^Hf(\eta)\,\mathrm{d}\eta}{2\pi}+\frac{1}{ik}\sum_{n=1}^\infty \frac{1}{\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}} \frac{\int_0^H f(\eta)\cos(n\pi\frac{\eta}{H})\,\mathrm{d}\eta}{\pi} \cos\left(n\pi\frac{y}{H}\right)
$$
### Left operators

The case for the left operatores is very similar, in this case the Neumann problem is

$$
\begin{cases}
\Delta u + k^2u=0 & \text{in }(-\infty,R)\times[0,H]\\
-\frac{\partial u}{\partial x} = f & \text{on } x=R
\end{cases}
$$

The expansion looks like

$$
u(\mathbf{x})=\frac{a_0}{\sqrt{2\pi}}e^{-ikx}+\sum_{n=1}^\infty a_n\frac{e^{-i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}kx}}{\sqrt{\pi}} \cos\left(n\pi\frac{y}{H}\right)
$$

the condition is then rewritten as:

$$
ik\frac{a_0}{\sqrt{2\pi}}e^{-ikR}+\sum_{n=1}^\infty ik\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}a_n\frac{e^{-i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}kR}}{\sqrt{\pi}} \cos\left(n\pi\frac{y}{H}\right) = \frac{f_0}{\sqrt{2\pi}}+ \sum_{n=1}^\infty f_n \frac{\cos\left(n\pi\frac{y}{H}\right)}{\sqrt{\pi}}
$$

So the solution is:

$$
u(\mathbf{x})=\frac{1}{ik}\frac{\int_0^Hf(\eta)\,\mathrm{d}\eta}{2\pi}e^{-ik(x-R)}+\frac{1}{ik}\sum_{n=1}^\infty \frac{1}{\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}} \frac{\int_0^H f(\eta)\cos(n\pi\frac{\eta}{H})\,\mathrm{d}\eta}{\pi} \cos\left(n\pi\frac{y}{H}\right)e^{-i\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}ki(x-R)}
$$
but the Neumann to Dirichlet operator is again: 

$$
\mathrm{NtD}(f)=\frac{1}{ik}\frac{\int_0^Hf(\eta)\,\mathrm{d}\eta}{2\pi}+\frac{1}{ik}\sum_{n=1}^\infty \frac{1}{\sqrt{1 -n^2 \left(\frac{\pi}{kH}\right)^2}} \frac{\int_0^H f(\eta)\cos(n\pi\frac{\eta}{H})\,\mathrm{d}\eta}{\pi} \cos\left(n\pi\frac{y}{H}\right)
$$
## Circular domain
### Dirichlet to Neumann map
The Dirichlet problem on the complement of a circle:

$$
\begin{cases}
\Delta u + k^2u = 0 & \text{in }\mathbb{R}^2 \setminus\overline{\mathrm{B}_R}\\
u = f & \text{on }\partial\mathrm{B}_R\\
\partial_r u - iku = o\left( \frac{1}{\sqrt{r}} \right)\text{as }r\to\infty
\end{cases}
$$
has a known solution interms of the Fourier coefficients of $f\circ\mathbf{x}$, where 
$\mathbf{x}:[0,2\pi]\to\mathbb{R}^2$ is a parametrization of the circle:

$$
\theta\mapsto\mathbf{x}(\theta)= \left(R\cos t,R\sin t\right) 
$$


$$
f(\mathbf{x}(\theta)) = \sum_{n=-\infty}^\infty f_n e^{in\theta}
$$

Let 
$$
u(\mathbf{x})=\sum_{n=-\infty}^\infty \frac{a_n}{H_n^{(1)}\left(kR\right)}H_n^{(1)}\left(k\|\mathbf{x}\|\right)e^{in\theta(\mathbf{x})}
$$
where $\theta(\mathbf{x})$ is the angle that vector $\mathbf{x}$ forms with te $Ox$ axis. Then Dirichlet condition reads
$$
\sum_{n=-\infty}^\infty a_n e^{in\theta(\mathbf{x})} = \sum_{n=-\infty}^\infty f_n e^{in\theta}
$$
that is $a_n\equiv f_n$.

Hence, the Dirichlet to Neumann operator $\mathrm{DtN}:L^2(\partial\mathrm{B}_R)\to L^2(\partial\mathrm{B}_R)$:
$$
f\mapsto \partial_{\mathbf{n}}u_f\vert_{R}
$$
can be computed as the, non local, expression:

$$
u(\mathbf{x})=\sum_{n=-\infty}^\infty \int_0^{2\pi}f(\mathbf{x}(s))e^{-ins}\,\mathrm{d}s \frac{k}{\pi}\frac{H'^{(1)}_n\left(kR\right)}{H_n^{(1)}\left(kR\right)}e^{in\theta(\mathbf{x})}
$$

## Neumann to Dirichlet map.

In the same manner, the Neumann problem
$$
\begin{cases}
\Delta u + k^2u = 0 & \text{in }\mathbb{R}^2 \setminus\overline{\mathrm{B}_R}\\
\partial_\mathbf{n}u = f & \text{on }\partial\mathrm{B}_R\\
\partial_r u - iku = o\left( \frac{1}{\sqrt{r}} \right)\text{as }r\to\infty
\end{cases}
$$
also admits a solution expressed as a sum of Hankel functions: 

$$
u(\mathbf{x})=\sum_{n=-\infty}^\infty \frac{a_n}{H_n^{(1)}\left(kR\right)}H_n^{(1)}\left(k\|\mathbf{x}\|\right)e^{in\theta(\mathbf{x})}
$$

Now the boundary condition reads: 
$$
\sum_{n=-\infty}^\infty \frac{a_nk}{H_n^{(1)}\left(kR\right)}H_n^{p(1)}\left(kR\right)e^{in\theta(\mathbf{x})} = \sum_{n=-\infty}^{\infty}f_ne^{in\theta}
$$
