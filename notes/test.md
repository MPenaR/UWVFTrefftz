# NtD fluxes for a circular boundary
The Dirichlet problem on the complement of a circle:

$$
\begin{cases}
\Delta u + k^2u = 0 & \text{in }\mathbb{R}^2 \setminus\overline{\mathrm{B}_R}\\
u = f & \text{on }\partial\mathrm{B}_R\\
\partial_r u - iku = o\left( \frac{1}{\sqrt{r}} \right)\text{as }r\to\infty
\end{cases}
$$
has a known solution interms of the Fourier coefficients of $f\circ\mathbf{x}$, where 
$\mathbf{x}:[0,2\pi]\to\mathbb{R}^2$ is a parametrization of the circle.

Let 
$$
u(\mathbf{x})=\sum_{n=-\infty}^\infty \frac{a_n}{H_n^{(1)}\left(kR\right)}H_n^{(1)}\left(k\|\mathbf{x}\|\right)e^{in\theta(\mathbf{x})}
$$
where $\theta(\mathbf{x})$ is the angle that vector $\mathbf{x}$ forms with te $Ox$ axis. Then Dirichlet condition reads
$$
\sum_{n=-\infty}^\infty a_n e^{in\theta(\mathbf{x})} = f(\mathbf{x})
$$


