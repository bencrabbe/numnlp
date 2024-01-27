---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Appendix to Euclidean spaces

This appendix provides additional proofs left underspecified in the main chapter.

## The Law of cosines (full version)

As pointed out in the main chapter, the *law of cosines* is the basis for justifying the cosine definition in the two dimensional case:

$$
\cos\theta = \frac{\mathbf{x}^\top \mathbf{y}}{|| \mathbf{x} ||\, ||\mathbf{y} || }
$$

Proving the law, $c^2=a^2+b^2-2ab\cos\theta$ , breaks down in three cases:
1. $\theta$ is an acute angle (proven in the main chapter)
2. $\theta$ is a right angle, in which case $\cos\theta=0$ and $c^2=a^2+b^2-2ab\cos\theta$ reduces to Pythagoras theorem
3. $\theta$ is an obtuse angle, which is the case left to prove 


To prove (3) let us consider the black triangle and its prolongation as a right triangle in blue:

```{code-cell}
:tags: ["remove-input"]
from myst_nb import glue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


f,ax = plt.subplots()
ax.set_xlim([-0.1,3])
ax.set_ylim([-0.1,1.1])
ax.set_axis_off()
ax.plot([1,2],[0,0],color='black')
ax.plot([0,1],[1,0],color='black')
ax.plot([0,2],[1,0],color='black')
ax.plot([0,0],[0,1],'--',color='b')
ax.plot([0,1],[0,0],'--',color='b')


ax.annotate("a",(0.5,0.4))
ax.annotate("b", (1.5,0),verticalalignment="top")
ax.annotate("c", (1,0.5))
ax.annotate(r"$\theta$", (1,0.0825),color="red")
ax.annotate(r"$\theta'$", (0.8,0.04),color="blue")

ax.annotate(r"$h$", (0.,0.5),color="blue", horizontalalignment="right", verticalalignment="center")
ax.annotate(r"$e$", (0.5,0.),color="blue", horizontalalignment="center" , verticalalignment="top")
pac = mpatches.Arc([1, 0], 0.15, 0.15, angle=0, theta1=0, theta2=135.,color="red")
ax.add_patch(pac)

pac = mpatches.Arc([1, 0], 0.15, 0.15, angle=135, theta1=0, theta2=45.,color="blue")
ax.add_patch(pac)

glue("geom_vector", f, display=False)
```

````{prf:lemma} Law of Cosines, obtuse angle case
:label: lawcosinesobtuse
We prove that $c^2 = a^2 + b^2 - 2ab\cos\theta$ when $\theta$ is an obtuse angle between sides of length $a$ and $b$.
To establish the proof we make explicit an additional right angled triangle (dotted blue line) with sides $h$ and $e$.  Then we apply Pythagoras on this triangle and develop:

$$
\begin{align}
c^2 &= (b+e)^2 + h^2& (\text{Pythagoras})\\
&= b^2+2be + e^2 + h^2\\
&= a^2 + b^2+ 2 be & (b^2 + h^2) = a^2\\
&=  a^2 + b^2- 2 ab\cos\theta &
\end{align}
$$

where the last line is obtained by observing that $\cos\theta' = \frac{e}{a}$. Consequently the obtuse angle $\theta$ has cosine $\cos\theta = -\frac{e}{a}$ and $e = -a\cos\theta$.
````




## Proof of Cauchy Schwarz

The Cauchy Schwarz inequality justifies the definition of cosines for the n-dimensional case.
Several proofs exist, but are in general non intuitive. We provide a proof that relies on two algebraic properties
of norms:

1. The squared norm is positive or null:  $||\mathbf{v}||^2\geq 0$ for $\mathbf{v} \in \mathbb{R}^n$
2. $||\mathbf{x}-\mathbf{y}||^2  = || \mathbf{x}^2 || - 2\mathbf{x}^\top \mathbf{y} + ||\mathbf{y}||^2$

````{prf:theorem} Cauchy Schwarz
:label: cs

We prove that $|\mathbf{x}^\top\mathbf{y}| \leq ||\mathbf{x}|| \, || \mathbf{y} ||$. Considering the rescaled vectors $||\mathbf{x}||\, \mathbf{y}$ and $||\mathbf{y}||\, \mathbf{x}$, whose difference
is $||\mathbf{x}||\, \mathbf{y} - ||\mathbf{y}||\, \mathbf{x}$. Then the squared norm  of this vector evaluates as:

$$
\begin{align*}
\big\lVert\, ||\mathbf{x}||\, \mathbf{y} - ||\mathbf{y}||\, \mathbf{x} \,\big\rVert^2 &=(||\mathbf{x}||\, \mathbf{y} - ||\mathbf{y}||\, \mathbf{x})^\top (||\mathbf{x}||\, \mathbf{y} - ||\mathbf{y}||\, \mathbf{x} )\\ 
&= ||\mathbf{x}||^2 (\mathbf{y}^\top\mathbf{y}) - 2 ||\mathbf{x} || \, ||\mathbf{y} ||  (\mathbf{x}^\top \mathbf{y})+ (\mathbf{x}^\top \mathbf{x}) ||\mathbf{y}||^2\\
&=||\mathbf{x}||^2 ||\mathbf{y}||^2- 2 ||\mathbf{x} || \, ||\mathbf{y} || (\mathbf{x}^\top \mathbf{y})+ ||\mathbf{x}||^2||\mathbf{y}||^2\\ 
&= 2 ||\mathbf{x}||^2 ||\mathbf{y}||^2 - 2 ||\mathbf{x} || \, ||\mathbf{y} || (\mathbf{x}^\top \mathbf{y}) 
\end{align*}
$$
Since the squared norm is $\geq 0$ we have:

$$
\begin{align*}
2 ||\mathbf{x}||^2 ||\mathbf{y}||^2 - 2 ||\mathbf{x} || \, ||\mathbf{y} || (\mathbf{x}^\top \mathbf{y})& \geq 0\\
2 ||\mathbf{x}||^2 ||\mathbf{y}||^2 & \geq 2 ||\mathbf{x} || \, ||\mathbf{y} || (\mathbf{x}^\top \mathbf{y})\\
 ||\mathbf{x}||^2 ||\mathbf{y}||^2 & \geq  ||\mathbf{x} || \, ||\mathbf{y} || (\mathbf{x}^\top \mathbf{y})\\
 ||\mathbf{x}||\, ||\mathbf{y}|| & \geq  \mathbf{x}^\top \mathbf{y}\\
\end{align*}
$$

The reader can check that  mirroring the same development from the norm $ \big\lVert\, ||\mathbf{x}||\, \mathbf{y} + ||\mathbf{y}||\, \mathbf{x} \,\big\rVert^2$ leads to the inequality $||\mathbf{x}||\, ||\mathbf{y}|| \geq  -\mathbf{x}^\top \mathbf{y}$. Thus we conclude:

$$
| \mathbf{x}^\top \mathbf{y} | \leq ||\mathbf{x}||\, ||\mathbf{y}|| 
$$
````