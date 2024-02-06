
Difference and differential equations
====


This chapter provides overview elements to modeling processes that evolve with time.
Time may be represented as a sequence of discrete elements from $\mathbb{N}$ or as a continuous sequence from $\mathbb{R}$. We mainly focus
on discrete time in this chapter as it is the most common use case in computer science and computational language modeling.

When time is discrete such a sequence $a_0\, a_1\, \ldots a_t$ can be expressed as a function of $t$ with the t-th term written as $a_t = f(t)$.

```{admonition} Example
:class: tip

Here is an example of the first few terms of such a sequence :

` 2 1.8 1.62 1.458 1.312 1.18  1.06 0.956  0.861 0.775 `

In this example the first term $a_0=2$ and each $a_t = 0.9^t a_0$. In other words $f(t) = 0.9^t \times 2$
and therefore $f(1) = 0.9\times 2 = 1.8$ , $f(2)=0.9^2 \times 2 = 1.62$ etc.
```
This kind of sequence can be written alternatively by a **difference equation** of the form $a_{t+1} + \alpha a_t =  f(t)$. That is 
an equation where a term of the sequence is expressed as a function of one or more previous terms in the sequence. That is the equation
characterizes how the terms evolve with time.

```{admonition} Example
:class: tip

Let us consider the equation: 

$$
\begin{align*}
a_{t+1} - 0.9 a_t &= 0\\
a_{t+1} &= 0.9 a_t
\end{align*}
$$

Solving a difference equation amounts to find all sequences, or **trajectories**, that satisfy the equation.
For instance the sequence starting with

` 2 1.8 1.62 1.458 1.312 1.18  1.06 0.956  0.861 0.775 `

written analytically as $f(t) = 0.9^t \times 2$ satisfies the equation, where 2 is a constant called the base of the recurrence. However this is not the only one,
every sequence of the form $f(t) = 0.9^t \times \lambda$ $(\lambda \in \mathbb{R})$ actually satisfies the equation.
Here $\lambda$ is the constant that sets the base of the recurrence: $a_0 = \lambda$. Thus the set of solutions to the difference equation is written as:

$$
f(t) = 0.9^t \times \lambda \qquad (\lambda \in \mathbb{R})
$$
```



@see nice ref: https://math.dartmouth.edu/opencalc2/dcsbook/c1pdf/sec14.pdf


Where do time sequences come from ? 
-------

  - optimisation
  - dynamic systems
  - physics



Behavior at the limit
----------
Does the sequence converge or diverge ? limits


Difference equations
--------

Example (logistic equation)


Differential equations
---------


Markov chains
----------

@see for the matrix power method
https://ergodic.ugr.es/cphys/lecciones/fortran/power_method.pdf

Neural Network
-----------
 - Express them as function of time => difference and differential equations

