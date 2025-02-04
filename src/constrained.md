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





# Constrained optimization

This chapter introduces some elementary notions of constrained
optimization with applications to natural language processing.

## Linear programming


Linear programs are a class of optimization problems that aim to
maximize a **linear objective function** subject to constraints stated as
**linear inequalities**.

Here is an example linear program for a bivariate function:

$$
\begin{align}
\text{maximize}\ & 3x_1+2x_2\\
\text{subject to}\ &\\
&x_1 \geq 0\\
                          &  x_2 \geq 0\\
						  & x_1 + 2x_2 \leq 6\\
						  & x_1+x_2 \leq 4  \\
						  &2x_1+x_2 \leq 7
\end{align}
$$



````{code-cell}
:tags: ["remove-input"]
from myst_nb import glue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


f,ax = plt.subplots()
ax.set_xlim([0.,3.5])
ax.set_ylim([0,3.])
ax.set_axis_off()

points    = np.array([(0,0),(0,3),(2,2),(3,1),(3.5,0)])
polygon = Polygon(points,closed=True, facecolor='y',alpha=0.2)

ax.plot([0,3.5],[0,0],color='black')
ax.plot([0,0],[0,3.5],color='black')
#inequalities
c1, = ax.plot([0,3.5],[3,1.25],color='red',label=r"$x_2 \leq -0.5x_1+3$")  #y <= -0.5x  + 3
c2,  =ax.plot([1,3.5],[3,0.5],color='green', label=r"$x_2 \leq -x_1+4$")    #y <= -x + 4
c3, = ax.plot([2,3.5],[3,0.],color='blue',label=r"$x_2 \leq -2x_1+7$")    #y <= -2x + 7
ax.add_patch(polygon)
ax.annotate("feasible region", (1.,1.))
ax.annotate(r"$x_1$", (1.5,0),verticalalignment='top')
ax.annotate(r"$x_2$", (0,1.5),horizontalalignment='right')

ax.legend(handles=[c1,c2,c3],loc="upper right")



glue("geom_vector", f, display=False)
````



simplex

## Integer Linear programming

applications to NLP

## Non linear constrained optimization

with equality constraints:

lagrange multipliers
