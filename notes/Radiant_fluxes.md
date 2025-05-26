# Radiant fluxes
## Waveguide

### Right hand side term

The RHS flux is of the form: 

$$
\ell(v)=\int_E\left(\left(\mathrm{NtD}(\partial_\mathbf{n}u_\mathrm{i}) - u_\mathrm{i}\right)\overline{\partial_\mathbf{n}v} - d_2ik\left(\mathrm{NtD}\left(\partial_\mathbf{n}u_\mathrm{i}\right) - u_\mathrm{i}\right)\overline{\left(\mathrm{NtD}\left(\partial_\mathbf{n}v\right)-v\right)} \right)\,\mathrm{d}\ell
$$

Terms which do not involve the $\mathrm{NtD}$ are local, i.e. do not comple degrees of freedom from different triangles. We compute them first: 


$$
-\int_E\left( u_\mathrm{i}\overline{\partial_\mathbf{n}v}+ d_2ik u_\mathrm{i}\overline{v} \right)\,\mathrm{d}\ell
$$

Now we are assuming $v_m(\mathbf{x})=e^{ik\mathbf{d}_m\cdot\mathbf{x}}$ and that we parametrize the edge as:

$$
\mathbf{x}(t)=\mathbf{x}_M + l \boldsymbol{\tau} t,\quad t\in[-\frac{1}{2},\frac{1}{2}],
$$
where $\mathbf{x}_M =\frac{\mathbf{x}_A + \mathbf{x}_B}{2}$ is the mid point of the segment $[\mathbf{x}_A,\mathbf{x}_B]$, $l=\|\mathbf{x}_B - \mathbf{x}_A\|$ is its length and $\boldsymbol{\tau}=\frac{\mathbf{x}_B -\mathbf{x}_A}{l}$ is the unit tangent vector. Then:

$$
-\int_E\left( u_\mathrm{i}\overline{\partial_\mathbf{n}v_m}+ d_2ik u_\mathrm{i}\overline{v_m} \right)\,\mathrm{d}\ell=
$$

$$
-\int_{-\frac{1}{2}}^{\frac{1}{2}}\left( u_\mathrm{i}\overline{ik\mathbf{d}_m\cdot\mathbf{n}e^{ik\mathbf{d}_m \cdot\mathbf{x}(t)}}+ d_2ik u_\mathrm{i}\overline{e^{ik\mathbf{d}_m \cdot\mathbf{x}(t)}} \right)\,\mathrm{d}\ell=
$$


$$
\int_{-\frac{1}{2}}^{\frac{1}{2}}\left( u_\mathrm{i}ik\mathbf{d}_m\cdot\mathbf{n}e^{-ik\mathbf{d}_m \cdot\mathbf{x}(t)}- d_2ik u_\mathrm{i}e^{-ik\mathbf{d}_m \cdot\mathbf{x}(t)}\right)\,\mathrm{d}\ell=
$$


$$
\int_{-\frac{1}{2}}^{\frac{1}{2}}\left( ik\mathbf{d}_m\cdot\mathbf{n}- d_2ik\right)u_\mathrm{i}e^{-ik\mathbf{d}_m \cdot\mathbf{x}(t)}\,\mathrm{d}\ell=
$$


$$
 \left( ik\mathbf{d}_m\cdot\mathbf{n}- d_2ik\right)\int_{-\frac{1}{2}}^{\frac{1}{2}}u_\mathrm{i}(\mathbf{x}(t))e^{-ik\mathbf{d}_m \cdot\left(\mathbf{x}_M+l\boldsymbol{\tau}t\right)}\,\mathrm{d}\ell=
$$


$$
 \left( ik\mathbf{d}_m\cdot\mathbf{n}- d_2ik\right)e^{-ik\mathbf{d}_m \cdot\mathbf{x}_M}\int_{-\frac{1}{2}}^{\frac{1}{2}}u_\mathrm{i}(\mathbf{x}(t))e^{-ikl\mathbf{d}_m \cdot\boldsymbol{\tau}t}\,\mathrm{d}\ell=
$$

and this derivation is valid for any type of incident field and even for the circular domain.
