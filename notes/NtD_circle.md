# NtD fluxes for a circular boundary
## Dirichlet to Neumann map
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

