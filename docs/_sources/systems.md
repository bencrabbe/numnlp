# Systems of linear equations


Although not directly related to language applications, we introduce here an automatic method for solving systems of linear equations.
This may be of interest for the reader willing to better understand a related problem: that of matrix inversion and it also helps to better understand
constrained optimization algorithms like the simplex.


A system of linear equations is a collection of equations involving the same variables and all these equations are linear.
The key initial observation is that such a system can be written in matrix vector form.

$$
\mathbf{Ax} = \mathbf{b} 
$$

`````{admonition} Example
:class: tip

Here is an example of a system with three equations:

$$
\left\{
\begin{array}{rcl}
3x_1+2x_2-x_3&=&1\\
2x_1-2x_2+4x_3&=&-2\\
-x_1+\frac{x_2}{2}-x_3 &=& 0 
\end{array}
\right.
$$

such a system can be rewritten in matrix form given the following vectors and matrices:

$$
\underbrace{
\begin{bmatrix}
3 & 2 &-1\\
2 &-2 &4\\
-1&\frac{1}{2} & -1
\end{bmatrix}
}_\mathbf{A}
\underbrace{
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
}_\mathbf{x}
=
\underbrace{
\begin{bmatrix}
1\\
-2\\
0
\end{bmatrix}
}_\mathbf{b}
$$

``````

The naive way to solve such a system is by variable substitution: one expresses a variable as a function of all the other variables and substitutes it in the other equations.
We introduce here another method that relies on the fundamental observation: in case the matrix of coefficients $\mathbf{A}$ is the identity matrix, 
the solution is trivially read off:

$$
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
=
\begin{bmatrix}
b_1\\
b_2\\
b_3
\end{bmatrix}
$$

since it reduces to the system:

$$
\begin{array}{rcl}
x_1 &=&b_1\\
x_2 &=& b_2\\
x_3 &=&b_3
\end{array}
$$

The gauss pivot method for solving systems of linear equations actually transforms any square matrix of coefficients $\mathbf{A}$ and the vector $\mathbf{b}$ into an equivalent matrix
$\mathbf{A}'$ and vector $\mathbf{b}'$ using structural transformation rules that preserve the equivalence of the system : the solutions remain the same.


## Row transformation rules

The row transformation rules (or pivoting rules) are rules to rewrite equations in a different form that preserve the solutions of the equation.
given an equation of the form $\mathbf{ax} = b$ the rules are

1. **Swap rule**. Two equations in a system can be swapped without changing the set of solutions
2. **Multiplication rule.** Let $k \not = 0$ the equation $\mathbf{a}^\top\mathbf{x} = b$ may be rewritten  $k \mathbf{a}^\top\mathbf{x} = k b$ 
3. **Addition rule** Let the equations $E_1 : \mathbf{a}_1^\top\mathbf{x}_1 =  b_1 $ and $E_2 : \mathbf{a}_2^\top\mathbf{x}_2 = b_2 $ we can replace $E_1$ 
by $E_1+E_2 :  \mathbf{a}_1^\top\mathbf{x}_1  +  \mathbf{a}_2^\top\mathbf{x}_2= b_1+b_2$. This justified since we add an equal quantity to each side of the equality.


Note that in practice the addition rule is commonly combined with the multiplication rule. 
The notation $E_i \gets \alpha E_i + \beta E_j$ means that equation $E_i$ is updated by multiplying $E_i$ and $E_j$ by non zero constants before adding them

## Gauss Jordan Pivot

The Gauss pivot algorithm solves a system of linear equation by reformulating it and transforming it until the matrix of coefficients is the identity matrix.
From there the solution is immediately read off. Given a system of linear equations with a square coefficient matrix $\mathbf{A}$ and vector of values $\mathbf{b}$
the algorithm starts from the augmented matrix:
$ [\mathbf{A} | \mathbf{b}]$

The augmented matrix is then transformed iteratively until it either reaches the row reduced form or it fails. 
At each iteration step $i$ the pivot column is column $\mathbf{a}_i$.

1. Select a pivot row such that it has a non zero coefficient in the pivoting column and its row has not been chosen earlier. In case no such pivot is found the system is not solvable.
2. Swap rows such that the pivot row becomes the $i$-th row in $\mathbf{A}$. The **pivot** is now element $a_{ii}$ 
3. Use row transformations to get $a_{ii} = 1$. 
4. Use row transformations of the form $E_j \gets E_j +  k E_i$ to get $a_{ij} = 0$ everywhere in the pivoting column except when $i=j$. $k$ is chosen to be $-\frac{a_{ij}}{a_{ii}}$


`````{admonition} Example
:class: tip

Here is an example for a system with three equations.

$$
\left\{
\begin{array}{rcl}
2x_1+x_2+2x_3&=&10\\
x_1+2x_2+x_3  &=&8\\
3x_1+x_2-x_3  &=& 2 
\end{array}
\right.
$$

and its coding in an augmented matrix:

$$
\left[
\begin{array}{ccc|c}
 2 & 1&2&10\\
{\color{red}\mathbf{1}} &2& 1& 8\\ 
3&1&-1&2
\end{array}\right]
\stackrel{swap}{\Rightarrow}
\left[
\begin{array}{ccc|c}
{\color{red}\mathbf{1}} &2& 1& 8\\ 
2 & 1&2&10\\
3&1&-1&2
\end{array}\right]

$$



The pivot in the first column is chosen to be 1 (the choice is motivated only to simplify  calculations) and after swapping we apply row transformations to get to the next iteration:
 
$$
\left[
\begin{array}{ccc|c}
1 & 2& 1& 8\\ 
0 & {\color{red} \mathbf{-3}}&0&-6\\
0&-5&-4&-22
\end{array}
\right]
\begin{array}{l}
\\
E_2\gets E_2 - 2 E_1\\
E_3\gets E_3 - 3  E_1
\end{array}
$$

Moving to the second column, we choose the pivot to be -3. We first divide the pivot row by -3 to get a 1 

$$
\left[
\begin{array}{ccc|c}
1 & 2& 1& 8\\ 
0 & {\color{red} 1}&0&2\\
0&-5&-4&-22
\end{array}
\right]
\begin{array}{l}
\strut\\
E_2\gets - \frac{1}{3}E_2\\
\strut
\end{array}
$$


then we apply linear transformations to zero out the coefficients in the pivot row.

$$
\left[
\begin{array}{ccc|c}
1 & 0& 1& 4\\ 
0 & 1&0&2\\
0&0&{\color{red} \mathbf{-4}}&-12
\end{array}
\right]
\begin{array}{l}
E_1\gets E_1 - 2E_2\\
\\
E_3 \gets E_3 +5 E_2 
\end{array}
$$

Finally we reach the third column where the only possible pivot is -4. 
By dividing the pivot row by -4 we get

$$
\left[
\begin{array}{ccc|c}
1 & 0& 1& 4\\ 
0 & 1&0&2\\
0&0&{\color{red} \mathbf{1}}&3
\end{array}
\right]
\begin{array}{l}
\strut\\
\strut\\
E_3 \gets -\frac{1}{4} E_4 
\end{array}
$$

and finally we zero out other coefficients in the pivot column to get:

$$
\left[
\begin{array}{ccc|c}
1 & 0& 0& 1\\ 
0 & 1&0&2\\
0&0&{\color{red} \mathbf{1}}&3
\end{array}
\right]
\begin{array}{l}
E_1 \gets  E_1 - E_3 \\
\strut\\
\strut
\end{array}
$$

The iteration is now over and the matrix is in row reduced form. We can read off the solution : $x_1 = 1, x_2 = 2, x_3 = 3$
`````

## Matrix inversion

The Gauss Jordan method can also be used to solve matrix inversion problems. Suppose we want 
to compute the inverse $\mathbf{A}^{-1}$ of matrix $\mathbf{A}$. We instanciate an augmented matrix of
the form $[\mathbf{A}|\mathbf{I}]$ with $\mathbf{I}$ the identity matrix. The inversion operation applies the pivoting methodology until the 
$\mathbf{A}$ matrix becomes the identity matrix. The $\mathbf{I}$ matrix will be transformed into the inverse once the process is complete.

 
`````{admonition} Example
:class: tip

Given the starting augmented matrix

$$
[\mathbf{A}|\mathbf{I}]
=
\left[
\begin{array}{ccc|ccc}
{\color{red}\mathbf{2}}&3&1&1&0&0\\
3&3&1&0&1&0\\
2&4&1&0&0&1
\end{array}
\right]
$$

The first pivot is the upper left 2. Then we update the matrix such that the pivot equals 1 and the remaining pivot column coefficients are 0:

$$
\left[
\begin{array}{ccc|ccc}
{\color{red}\mathbf{1}}&\frac{3}{2}&\frac{1}{2}&\frac{1}{2}&0&0\\
0&-\frac{3}{2}&-\frac{1}{2}&-\frac{3}{2}&1&0\\
0&1&0&-1&0&1
\end{array}
\right]
\begin{array}{l}
E_1 \gets  \frac{1}{2}E_1 \\
E_2 \gets E_2 - \frac{3}{2} E_1 \\
E_3 \gets E_3 -  E_1 \\
\end{array}
$$

For the second column we choose the next pivot to be 1 and swap rows accordingly:

$$
\left[
\begin{array}{ccc|ccc}
1&\frac{3}{2}&\frac{1}{2}&\frac{1}{2}&0&0\\
0&{\color{red}\mathbf{1}}&0&-1&0&1\\
0&-\frac{3}{2}&-\frac{1}{2}&-\frac{3}{2}&1&0
\end{array}
\right]
$$

We move to the next iteration by setting non pivot coefficients to 0:

$$
\left[
\begin{array}{ccc|ccc}
1&0&\frac{1}{2}&2&0&-\frac{3}{2}\\
0&1&0&-1&0&1\\
0&0&{\color{red} -\mathbf{\frac{1}{2}}}&-3&1&\frac{3}{2}
\end{array}
\right]
\begin{array}{l}
E_1 \gets  E_1 - \frac{3}{2}E_2 \\
\strut \\
E_3 \gets E_3 +  \frac{3}{2} E_2 \\
\end{array}
$$

For the last iteration the only available pivot is  -1/2 and again we set the third column coefficients to 0 in order to get the result.

$$
\left[
\begin{array}{ccc|ccc}
1&0&0&-1&1&0\\
0&1&0&-1&0&1\\
0&0&{\color{red} \mathbf{1}}&6&-2&-3
\end{array}
\right]
\begin{array}{l}
E_1 \gets  E_1 + E_3 \\
\strut \\
E_3 \gets -2E_3  \\
\end{array}
$$ 

We conclude that the inverse of matrix $\mathbf{A}$ is the matrix

$$
\mathbf{A}^{-1} = 
\left[
\begin{array}{ccc}
-1&1&0\\
-1&0&1\\
6&-2&-3
\end{array}
\right]
$$


``````


















