# Appendix to Euclidean spaces


This appendix provides additional proofs left underspecified in the main chapter.


## The Law of cosines (full version)



As pointed out in the main chapter, the *law of cosines* is the basis for justifying the cosine definition in the two dimensional case:



$$ \cos\theta = \frac{\mathbf{x}\top \mathbf{y}}{||\mathbf{x}||\,
||\mathbf{y}||}
$$

Proving the law $c^2 = a^2+b^2-2ab\cos\theta$ Breaks down in three cases:
 
1. $\theta$ is an acute angle (proven in the main chapter)
2. $\theta$ is a right angle, in which case $\cos\theta= 0$ and $c^2 = a^2+b^2-2ab\cos\theta$ reduces to Pythagoras theorem
3. $\theta$ is an obtuse angle, which is the case left to prove.

To prove (3) let us consider the black triangle and its prolongation
as a right triangle in blue.


````{prf:lemma} Law of Cosines, obtuse angles 

We prove that $c^2 = a^2+b^2-2ab\cos\theta$  when $\theta$ is an
obtuse angle between sides $a$ and $b$. To establish the proof we make
explicit an additional right angled triangle (dotted blue line), with
sides $h$ and $e$. Then we apply Pythagoras on, this triangle and develop: 

$$
\begin{align}
c^2 &= (b+e)^2+h^2\\
       &= b^2+2be+e^2+h^2\\
	   &=a^2+b^2+2be & (e^2+h^2) = a^2\\
	   &=a^2+b^2-2ab\cos\theta
\end{align}
$$


where the last line is obtained by observing that $\cos\theta' =
\frac{e}{a}$. Consequently the obtuse angle $\theta$ has cosine
$\cos\theta = - \frac{e}{a}$ and $e= -a\cos\theta$

````

## Proof of Cauchy Schwarz


The Cauchy Schwarz inequality justifies the definition for the
n-dimensional case. Several proofs exists, but are algebric and
generally less intuitive than the law of cosines. We provide such a
proof that relies on two algebraic properties of norms:

1. The squared norm is positive or null : $||\mathbf{v}||^2 \geq 0$
for $\mathbf{v}\in\mathbb{R}^n$
2. $|| \mathbf{x}+\mathbf{y}||^2 =
   ||\mathbf{x}||^2+2\mathbf{x}^\top\mathbf{y} +||\mathbf{y}||^2$

````{prf:theorem} Cauchy Schwarz

We prove that $|\mathbf{x}^\top \mathbf{y}| \leq ||\mathbf{x}||\,
||\mathbf{y}||$. Consider the vectors $\mathbf{x},\mathbf{y} \in \mathbb{R}^n$ and the scalar
$k\in \mathbb{R}$. The proof takes advantage of the properties of the
norm $|| \mathbf{x}+k\mathbf{y}||$:

$$
|| \mathbf{x}+k\mathbf{y}|| = ||\mathbf{x}||^2 + 2\mathbf{x}^\top \mathbf{y}k+k^2||\mathbf{y}||^2
$$


By definition $|| \mathbf{x}+k\mathbf{y}|| \geq 0$. Therefore we can
choose any real value for $k$. By choosing $k =
\frac{-\mathbf{x}^\top\mathbf{y} }{||\mathbf{y}||^2}$ we get:

$$
\begin{align}
||\mathbf{x}+k\mathbf{y} ||^2 &= ||\mathbf{x} ||^2+2(\mathbf{x}^\top\mathbf{y})^\top \frac{-\mathbf{x}^\top\mathbf{y}}{||\mathbf{y}||^2} + \frac{(\mathbf{x}^\top\mathbf{y})^2}{||\mathbf{y}||^4} ||\mathbf{y}||^2\\
                                                 &= ||\mathbf{x} ||^2 - \frac{2(\mathbf{x}^\top\mathbf{y})^2}{||\mathbf{y}||^2}+\frac{(\mathbf{x}^\top\mathbf{y})^2}{||\mathbf{y}||^2}\\
	&= ||\mathbf{x} ||^2 - \frac{(\mathbf{x}^\top\mathbf{y})^2}{||\mathbf{y}||^2}
\end{align}
$$

As the norm is positive or null, we further have:

$$
\begin{align}
0 &\leq ||\mathbf{x}||^2 -  \frac{(\mathbf{x}^\top\mathbf{y})^2}{||\mathbf{y}||^2}\\
    \frac{(\mathbf{x}^\top\mathbf{y})^2}{||\mathbf{y}||^2}
      &\leq ||\mathbf{x}||^2\\
    (\mathbf{x}^\top\mathbf{y})^2 & \leq ||\mathbf{x}||^2 \,||\mathbf{y}||^2\\
 |\mathbf{x}^\top\mathbf{y})| & \leq ||\mathbf{x}|| \,||\mathbf{y}||\\
\end{align}
$$
````





