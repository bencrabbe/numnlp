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

Here is an example of linear program:

$$
\begin{align}
\text{maximize}\ & 3x+2y\\
\text{subject to}\ &\\
&x \geq 0\\
                          &  y \geq 0\\
						  & x + 2y \leq 6\\
						  & x+y \leq 4  \\
						  &2x+y \leq 7
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
polygon = Polygon(points,closed=True, facecolor='r',alpha=0.4)

ax.plot([0,3.5],[0,0],color='black')
ax.plot([0,0],[0,3.5],color='black')
#inequalities
c1, = ax.plot([0,3.5],[3,1.25],color='red',label=r"$y \leq -0.5x+3$")  #y <= -0.5x  + 3
c2,  =ax.plot([1,3.5],[3,0.5],color='green', label=r"$y \leq -x+4$")    #y <= -x + 4
c3, = ax.plot([2,3.5],[3,0.],color='blue',label=r"$y \leq -2x+7$")    #y <= -2x + 7
ax.add_patch(polygon)
ax.legend(handles=[c1,c2,c3],loc="upper right")

glue("geom_vector", f, display=False)
````



simplex

## Integer Linear programming

applications to NLP

## Non linear constrained optimization

with equality constraints:

lagrange multipliers
