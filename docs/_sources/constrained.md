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
\text{maximize}\ & f(\mathbf{x}) = 3x_1+2x_2\\
\text{subject to}\ &\\
&x_1 \geq 0\\
                          &  x_2 \geq 0\\
						  & x_1 + 2x_2 \leq 6\\
						  & x_1+x_2 \leq 4  \\
						  &2x_1+x_2 \leq 7
\end{align}
$$

As can be seen a linear program has an hyperplane as objective
function. The shape delimited by the constraints takes
the form of a *polytope* (the generalization of polygon to an
arbitrary number of dimensions)  the area inside the polytope is
called the **feasible region**.

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
ax.annotate("(0,0)", (0.01,0.01), verticalalignment='bottom',horizontalalignment='left')
ax.annotate("(2,2)", (2.01,2.01), verticalalignment='bottom',horizontalalignment='left')
ax.annotate("(3,1)", (3.01,1.01), verticalalignment='bottom',horizontalalignment='left')
ax.annotate("(0,3)", (0.01,2.89), verticalalignment='top',horizontalalignment='left')
ax.annotate("(3.5,0)", (3.39,0.01), verticalalignment='bottom',horizontalalignment='right')
 
ax.annotate(r"$x_1$", (1.5,0),verticalalignment='top')
ax.annotate(r"$x_2$", (0,1.5),horizontalalignment='right')

ax.legend(handles=[c1,c2,c3],loc="upper right")



glue("geom_vector", f, display=False)
````

Considering the favorable case, where the feasible region is non empty
and where the solution is unique (at least one slope coefficient is non null) one has to observe that the *solution
is located on a vertex of the polytope*.

A brute force algorithm for finding the maximum can basically evaluate
all these vertices and keep the one that maximizes $f(\mathbf{x})$.
It turns out that when solving real life problems the number of
constraints can become so huge that the brute force solution is not
practical anymore. The simplex algorithm aims to provide a method that
avoids such an exhaustive exploration. 

 
### Simplex algorithm

The simplex algorithm is an algorithm for solving linear programs that conceptually moves from
vertex to vertex on the polytope around the feasible region. This is a
pivoting algorithm of the same family as the Gaussian Pivot for
solving systems of linear equations. To solve a problem with the
simplex one follows two steps :

1. Rewrite the optimization problem in *standard form*
2. Run the algorithm on the standard form

#### Conversion to standard form

The conversion of the problem in standard form involves turning
inequalities into equalities. This transformation aims to fullfill the
requirements of the algorithm 

We assume the problem is given in canonical form, that is

$$
\begin{align}
\text{maximize}\ &\mathbf{c}^\top \mathbf{x}\\
\text{subject to}\ &\mathbf{Ax}\leq \mathbf{b}\\
&\mathbf{x}\geq k 
\end{align}
$$

There are two main cases. First, for constraints of the form $x_i \geq k$
specifying a lower bound to the variable different from 0 ($k \not =
0$), the transformation introduces the new variable $y_i = x_i - k$
such that $y_i \geq 0$. In this case $x_i = y_i + k$ and we can use
this equation to replace $x_i$ everywhere in the original problem.


`````{admonition} Example (replace non zero lower bound)
:class: tip

Suppose we have the constraint $x_1 \geq 7$. Then we introduce:

$$
\begin{align}
y_i &= x_i - 7\\
y_i &\geq 0
\end{align}
$$

and we use the equation $x_i = y_i + 7$ to replace $x_i$ everywhere in
the problem. 
 `````
 
 Second, the remaining inequality constraints are replaced by equality
 constraints. This transformation is achieved by introducing *slack variables*
such that each slack variable $s_i$ is lower bounded at 0 ($s_i \geq
 0$). Thus an inequation of the form $
 a_1x_1+\ldots + a_k x_k \leq b
 $ is turned into the equation:

$$
\begin{align}
 a_1x_1+\ldots + a_k x_k + s_i &= b\\
 s_i &\geq 0
\end{align}
$$


`````{admonition} Example (introduce slack variables)
:class: tip

In the example above we have three inequations for which slack
variables have to be introduced. Once slack variables are introduced we have the system of equations:

$$
\begin{align}
x_1+2x_2 + s_1&= 6\\
x_1 + x_2 + s_2&= 4\\
2x_1+x_2 + s_3&= 7\\
s_1,s_2,s_3&\geq 0
\end{align}
$$
 `````

#### The simplex pivoting algorithm

The simplex algorithm is a pivoting algorithm that moves from vertex
to vertex on the polytope. It starts by picking some vertex and then
seeks the next vertex to go. At each time step this vertex is chosen in such a way that
the objective increases. If no such vertex can be found then the
algorithm terminates, it has found the solution.


Encoded in standard form the maximization problem is turned into a
system of linear equations. On the one hand we have the constraints
that have now the form  $\mathbf{Ax} = \mathbf{b}$ (where $\mathbf{A}$
is the matrix of coefficients including those for the slack
variables). On the other hand we have the objective $z =
\mathbf{c}^\top\mathbf{x}$. To standardize the objective it is
rewritten from the functional form $z =  c_1x_1+\ldots+ c_kx_k$ to the
equation $z-c_1x_1- \ldots -
c_kx_k = 0$.  As a result the canonical system of equations can be organized in
the following simplex matrix 

$$
\begin{bmatrix}
1                & \mathbf{c} & 0\\
\mathbf{0} & \mathbf{A} &\mathbf{b} 
\end{bmatrix}
$$


`````{admonition} Example (simplex table)
:class: tip

Continuing the running example from this chapter, the simplex tableau built
from the standard form is the following.

$$
\begin{array}{c|cc|ccc|c|c}
z&x_1&x_2&s_1&s_2&s_3&b&\text{bfs}\\\hline 
1& -3&-2  & 0  & 0  & 0   &0 & z=0\\ 
0&1  &  2  & 1  &  0  & 0   &6 &s_1=6\\
0 &1 & 1  &0 &1 & 0&4&s_2 = 4\\
0&2&1&0&0&1&7&s_3=7
\end{array}
$$

The additional last column indicates the current basic feasible
solutions. Initially $z$ and the slack variables are part of the **basic feasible solution** (bfs). Variables mentioned in this column
have coefficient equal to 1 and their values is read off immediately
from the b column in a manner analog to reading off values from row
reduced echelon form used when solving systems of linear equations
with the Gauss pivot.
Variables outside the basic feasible solution have value 0. Thus the algorithm starts in a state where $\mathbf{x} =
\mathbf{0}$ and where the objective has value $z=0$.

`````

**Choosing the pivot** The simplex algorithm updates its state, the basic
feasible solution, from time step to time step. The update amounts to
drop one variable from the current basic feasible solution and to add a new one
instead. The choice of which variables to substitute in the bfs is
made by choosing a pivot in the table. 

The pivot column is chosen by
selecting the *most negative coefficient* on the objective line
indicating a positive partial slope in the objective function (this is
the most negative since signs are inverted when constructing the
table). 

In case there is no negative coefficient in the objective line,
optimization is over the maximum is found. 


The pivot row is chosen by evaluating every *positive coefficient*
$a_{ij}$ in the selected column and pretend it is part of the basis (in practice
use the row division $b_j/a_{ij}$ for selected pivot column $i$). The one that
yields the lowest value is chosen since it avoids progressing too fast
and triggering a sign change when updating the state. 

In case no row contains a positive coefficient, optimization is over
and there is no feasible solution.



**Update the state**  
The pivot is assigned the value 1 and all other rows in the
constraints are transformed using standard pivoting rules (see solving
linear equations systems). Allowed rules are line multiplication and adding line multiples to other lines.


`````{admonition} Example (simplex execution)
:class: tip

Here is an illustration of the execution of the algorithm on the
running example given so far. For convenience we repeat the initial
tableau:

$$
\begin{array}{c|cc|ccc|c|c}
z&x_1&x_2&s_1&s_2&s_3&b&\text{bfs}\\\hline 
1& -3&-2  & 0  & 0  & 0   &0 & z\\ 
0&1  &  2  & 1  &  0  & 0   &6 &s_1\\
0 &1 & 1  &0 &1 & 0&4&s_2 \\
0&{\color{red}\mathbf{2}}&1&0&0&1&7&s_3
\end{array}
$$
The pivot in (red) is chosen from column $x_1$ and last row since this
is the highest negative coefficient in the objective (-3) and the
ratio $7/2$ is the minimal among the rows. It means that we introduce
$x_1$ in the bfs and we drop $s_3$. 

The table is transformed using pivoting rules to set the pivot coefficient
at 1 and everything else at 0 in the pivoting column by adding
multiples of the dropped row.  The resulting table
is given below where actual transformation rules are annotated rightwards


$$
\begin{array}{c|cc|ccc|c|c|l}
z&x_1&x_2&s_1&s_2&s_3&b&\text{bfs}&\text{rules}\\\hline 
1& 0& -0.5  & 0  & 0   &1.5 &10.5 & z&O\gets O + 1.5 R_3 \\ 
0 &0 & 1.5  &1 &0 & -0.5&2.5&s_1& R_1\gets R_1 - 0.5 R_1 \\
0&0  & {\color{red} \mathbf{0.5}}  & 0  &  1  & -0.5   &0.5&s_2&R_2\gets R_2-0.5 R_3\\
0&1&0.5&0&0&0.5&3.5&x_1 & R_3\gets 0.5 R_3
\end{array}
$$

The iteration continues since the objective coefficient in column
$x_2$ is still negative. The pivot is in column $x_2$ and the lowest
ratio $b/a_2$ is 1 and given by row 2. In other words we introduce
variable $x_2$ and drop variable $s_2$ from the bfs. The application of the pivoting
rules yield the following table

$$
\begin{array}{c|cc|ccc|c|c|l}
z&x_1&x_2&s_1&s_2&s_3&b&\text{bfs}&\text{rules}\\\hline 
1& 0& 0 & 0  & 1   &1 & 11&z&O\gets O +  R_2 \\ 
0 &0 & 0  &1 &-3 & 1&1&s_1 & R_1\gets R_1 - 3 R_2 \\
0&0  & 1  & 0  &  2  & -1&1&x_2&R_2\gets 2 R_2\\
0&1&0&0&-1&0&3&x_1 & R_3\gets R_3 - R_2
\end{array}
$$

Since there is no more negative coefficient in the objective row, the
iteration stops and values of the variables in the bfs can be read off
from the b column. We thus have the solution $x_1 = 3, x_2=1, s_1=1$
and the value of the objective is $z=11$.  Usually solvers will
report only the values for the $\mathbf{x}$ vector. 




``````

### Linear programming in practice

In practice one uses a dedicated solver. Here we illustrate how to
implement the example given above with the `pulp`solver. 

````{code-cell}
from pulp import *
x1 = LpVariable("x1")
x2 = LpVariable("x2")
P  = LpProblem("example",LpMaximize)
P += 3 * x1 + 2 * x2  #objective
P += x1 + 2 * x2 <= 6 #1st constraint
P += x1 + x2 <= 4     #2nd constraint
P += 2 * x1 + x2 <= 7 #3rd constraint

status = P.solve() 
print("Solution status:", LpStatus[status])

print(f"Solution:  x1 : {value(x1)}, x2: {value(x2)}  ")

````

## Integer Linear programming


In the context of natural language processing a particular case of
linear programming is of interest. It is the case where variables are
integer (and even boolean in practice). This difference has some
consequences for the solver algorithm and the simplex cannot be
straightfowardly applied anymore.  To see why let us consider the
following example integer linear problem:




$$
\begin{align}
\text{maximize}\ & f(\mathbf{x}) = 3x_1+2x_2\\
\text{subject to}\ &\\
						  &  x_1 + 2 x_2 \leq 6\\
						  & x_1+x_2 \leq 3.5  \\
						  &1.5 x_1+x_2 \leq 4.5\\
						  &x_1 ,x_2 \geq 0\\
						 &  x_1,x_2 \in \mathbb{Z} \\
\end{align}
$$


````{code-cell}
:tags: ["remove-input"]
from myst_nb import glue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


f,ax = plt.subplots()
ax.set_xlim([0.,3.])
ax.set_ylim([0,3.])
ax.set_axis_off()

points    = np.array([(0,0),(0,3),(1,2.5),(2,1.5),(3,0)])
polygon = Polygon(points,closed=True, facecolor='y',alpha=0.2)

ax.plot([0,3.5],[0,0],color='black')
ax.plot([0,0],[0,3.5],color='black')
#inequalities
c1, = ax.plot([0,3.5],[3,1.25],color='red',label=r"$x_2 \leq -0.5x_1+3$")  #y <= -0.5x  + 3
c2,  =ax.plot([0,3.5],[3.5,0],color='green', label=r"$x_2 \leq -x_1+3.5$")    #y <= -x + 3.5
c3, = ax.plot([0,3.5],[4.5,-0.75],color='blue',label=r"$x_2 \leq -1.5x_1+4.5$")    #y <= -1.5x + 4.5
ax.add_patch(polygon)
ax.annotate("feasible region", (1.,1.))
ax.annotate("(0,0)", (0.01,0.01), verticalalignment='bottom',horizontalalignment='left')
ax.annotate("(1,2.5)", (1.01,2.51), verticalalignment='bottom',horizontalalignment='left')
ax.annotate("(2,1.5)", (2.01,1.51), verticalalignment='bottom',horizontalalignment='left')
ax.annotate("(0,3)", (0.01,2.89), verticalalignment='top',horizontalalignment='left')
ax.annotate("(3,0)", (2.89,0.01), verticalalignment='bottom',horizontalalignment='right')
 
ax.annotate(r"$x_1$", (1.5,0),verticalalignment='top')
ax.annotate(r"$x_2$", (0,1.5),horizontalalignment='right')

ax.legend(handles=[c1,c2,c3],loc="upper right")

ax.scatter([0,0,0,0,1,1,1,2,2,3],[0,1,2,3, 0,1,2,0,1,0],color='red')


glue("geom_vector", f, display=False)
````

The dots show where the feasible integer values are located. As can be
observed these values are not anymore on the edges of the polytope. 
In other words algorithms for solving linear problems like the simplex
are not anymore suitable for solving integer linear problems. The
design of algorithms for solving such problems is beyond the scope of
these notes. 

```{warning}

Solving linear programs is in practice solved in polynomial time (as a
function of the number of variables). Solving integer linear programs
falls in the class of NP-hard problems. In other words ILP may face
scalability issues when the number of variables is too high
```


````{admonition} Solving integer linear problems

Solving integer problems is usually more complicated (higher complexity) than solving linear problems. 
Note that solving the linear relaxation in the first place may help. It means that
solving the linear problem may help to approach the integer solution. But naive methods such as
rounding off the solution to the nearest integer is invalid.
````

### ILP in practice

Solving integer linear programs is made with dedicated libraries. Here
is how to solve the above example with the `pulp` library.



````{code-cell}
from pulp import *
x1 = LpVariable("x1",cat="Integer")
x2 = LpVariable("x2",cat="Integer")
P   = LpProblem("example",LpMaximize)
P += 3*x1 + 2 * x2
P += x1 + 2 * x2 <= 6
P += x1 + x2  <= 3.5
P += 1.5 * x1 + x2 <= 4.5

status = P.solve()
print("Solution status:",LpStatus[status])
print(f"Solution: x1 : {value(x1)}  x2: {value(x2)} ")

````

### ILP for Natural Language Processing

In natural language processing, Integer Linear Programming is mostly
used for decoding data structures with a rich set of constraints using
boolean variables. 

Here is a simple ILP formulation illustrating how to constrain a graph representing
coreference relations with a set of basic constraints. Consider the
sentence:

> **Anna** is a student at **Paris Cité**. **She** loves **the University**. **Her**
hobbies include blogging and singing.

Words bolded indicate referring expressions. The task of coreference
resolution aims to link together expressions referring to the same entity.
Assuming that weighted edges have been predicted between mentions the
task is to predict the highest weighted graph linking the nodes. 


````{code-cell}
:tags: ["remove-input"]

import netgraph as ng
import matplotlib.pyplot as plt



pos=  {
	0 : (0,0),
	1 : (2,0),
	2 : (3,0),
	3 : (4,0),
	4 : (5,0)
}

f,ax = plt.subplots()
ax.set_axis_off()


#node_layout=pos

G  = ng.ArcDiagram([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)],ax=ax,pos=pos,arrows=True, above=True,edge_width=0.2,node_alpha=0.1,edge_layout_kwargs={'rad':0.5})
G1 = ng.ArcDiagram([(1,0),(2,0),(3,0),(4,0),(2,1),(3,1),(4,1),(3,2),(4,2),(4,3)],ax=ax,pos=pos,above=False,arrows=True,node_labels={0:"Anna",1:"Paris Cité",2:"She",3:"the University",4:"Her"}, node_label_fontdict={'size':12},edge_width=0.2, node_alpha=0.1, edge_layout_kwargs={'rad':0.5})


f.canvas.draw()

glue("cgraph", f, display=False)
````
**Plain English statement** Given a complete weighted directed graph as illustrated above the task is to select the subgraph with maximal weight that satisfies the following constraints:

- Anaphoric pronouns must have at least an antecedent.
- Only anaphoric pronouns have antecedents. Otherwise the referring expression has no antecedent 

**ILP Formulation** To formulate these constraints as an ILP problem, we first define two kinds of **boolean variables**:
- One boolean variable for each mention: $\mathbf{m} =  m_1,m_2,m_3,m_4,m_5$ indicating whether the mention is anaphoric or not. 
Given the example, $m_3 = 1$ and $m_5 = 1$ the the other variables are set to 0.

- One boolean variable for each potential anaphoric edge : $\mathbf{e} = e_{12} \ldots e_{45}$ indicating whether the edge is included in the solution or not (totalling 20 such variables).
We further assume a vector $\mathbf{w}$ of weights representing the edge weights.

The ILP formulation is the following:

$$
\begin{align}
\text{maximize}\ & \mathbf{w}^\top \mathbf{e}\\
\text{subject to}\ \\
& m_j \leq \sum_{i} e_{ij} & (\forall_{m_{j}})\\
&  e_{ij} \leq m_j & (\forall_{e_{ij}})
\end{align}
$$

To understand the formulation, the overall objective states that the edge scores should be maximal. 
The solver will include as many edges with positive weight as possible.
The first set of constraints states that the number of coreferents is greater or equal to 1 if $m_j$ is anaphoric (greater or equal to 0 otherwise).
The second set of constraints states that arcs can leave from an mention marked as anaphor 

````{code-cell}
#Here is how to implement the coreference problem with pulp

from pulp import *
from random import random

edges = { (i,j) : LpVariable( f"e{i}{j}",cat="Binary")  for i in range(5) for j in range(5) if i != j }

## Initalize and fix variables

mentions = [LpVariable( f"m{j}",cat="Binary")  for j in range(5) ]
for idx, variable in enumerate(mentions):
	if idx in [2,4]:
		variable.setInitialValue(1)
	else:
		variable. setInitialValue(0)
	variable.fixValue()



## Objective (we arbitrarily assign positive random weights to edges)

P   = LpProblem("coreference",LpMaximize)
P += lpSum( [ e * random()   for  (varname, e) in  edges.items() ] )


## 1st  set of constraints

for jdx,varm in enumerate(mentions):
	E = [varedge for (i,j),varedge in edges.items() if  j == jdx ]
	P+= varm <= lpSum(E)

## 2nd set of constraints

for edge,varedge in edges.items():
	(i,j) = edge
	P +=  varedge <= mentions[j]
	
## Results 
status = P.solve()
print("Solution status:",LpStatus[status])
print("Edges in solution")
for varname,Var in edges.items():
	if value(Var) == 1 :
		print(varname,value(Var))

````

````{warning}
In this exercise we arbitrarily assign positive weights to edges which is easier for understanding the properties of the formulation:
all edges that satisfy the constraints are included in the solution. 

The typical realistic scenario using a true machine learning model will assign positive and negative weights to the edges, discarding the less likely ones
with negative scores from the solution.
````










## Non linear constrained optimization

with equality constraints:

lagrange multipliers
