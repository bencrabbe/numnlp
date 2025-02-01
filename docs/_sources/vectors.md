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


# Vectors and vector spaces

In this chapter, we focus on the notion of vector and motivate
their use through various examples. 

## Vectors

A **vector** is an $n-$tuple of real numbers. In this
course, we'll be dealing exclusively with $n$-tuples of $\mathbb{R}^n$, even if other possibilities exist.
For example, a vector $\mathbf{x} \in \mathbb{R}^4$ will be noted as follows:

$$
\mathbf{x} = 
\begin{bmatrix}
x_1\\x_2\\x_3\\x_4
\end{bmatrix}
$$
where each of the $x_i$ is an **element**, a **coefficient** or a **coordinate** of the vector

`````{admonition} Example
:class: tip
Let us consider the vector $\mathbf{x}\in\mathbb{R}^3$:

$$
\mathbf{x} = 
\begin{bmatrix}
1\\0\\\sqrt{2}
\end{bmatrix}
$$
We have that $x_1 = 1, x_2 = 0$ and $x_3 = \sqrt{2}$. Observe also
that the order of the vector elements matters:

$$
\begin{bmatrix}
1\\0\\\sqrt{2}
\end{bmatrix}
\not = 
\begin{bmatrix}
1\\\sqrt{2}\\0
\end{bmatrix}
$$
`````


Alternatively, we sometimes use a horizontal notation and we write
$\mathbf{x} = (x_1, x_2, x_3, x_4)$ or even abbreviated as $\mathbf{x} = (x_1 \ldots x_4)$

In general, we note a vector in bold and a scalar (a number) in
standard font. Then an element of a vector $\mathbf{x}$ at position $i$ is denoted by
$x_i$ and the **slice** of a vector $\mathbf{x}  = x_1 \ldots x_i
\ldots x_j \ldots x_n$ between indices
$i$ and $j$ is the vector $\mathbf{x}_{i:j} = x_i\ldots x_j$


```{warning} 
Contrary to classical mathematical notation, it should be noted that Python tuples, lists and most computer libraries
  like *numpy* start indexing at 0, example:
```

```{code-cell} Warning
x = (2.,4.,6)
print(x[0])
```

```{warning}
Programming languages and libraries, when they allow slices, they do
so by including $i$ and excluding $j$ from the slice, example:
```
```{code-cell}
x = (2.,4.,6)
print(x[0:2])
```

We also use a notation for the concatenation of vectors. If $a = (a_1
\ldots a_i),$ $b = (b_1 \ldots b_j),$ $c = (c_1 \ldots c_k)$ then the concatenation is written :

$$
\begin{bmatrix}
\mathbf{a} \\\mathbf{b}\\\mathbf{c}
\end{bmatrix}
=
\begin{bmatrix}
a_1\\\vdots\\a_i\\ b_1\\ \vdots\\ b_j\\\vdots\\ c_1\\\vdots\\ c_k
\end{bmatrix}
$$

We further distinguish a few remarkable vectors. The **null vector**,
written $\mathbf{0}$ is a vector whose all elements have the
value 0. We sometimes note the size $n$ of the vector in subscript $(
\mathbf{0}_n)$ to be more explicit. The **one-hot** vector
is the vector whose values are all zero except for one value at
position $i$, which is 1. Finally, we note with $\mathbf{1}$ the
vector whose elements are all equal to 1.


### Vectors and geometry

In 2 and 3 dimensions,  vectors may be viewed as a movement from
the origin of a coordinate system to its coordinates. It may also be
viewed as the coordinates of its destination. This is illustrated with the
following plot for the 2-dimensional vector $\mathbf{x} = \begin{bmatrix}3\\2\end{bmatrix}$:

```{code-cell}
:tags: ["remove-input"]
from myst_nb import glue
import numpy as np
import matplotlib.pyplot as plt
f,ax = plt.subplots()
ax.set_xlim([0,4])
ax.set_ylim([0,4])
ax.plot([3,3],[0,2],'--',color="blue")
ax.plot([0,3],[2,2],'--',color="blue")
V = [3,2]
ax.quiver(0, 0, V[0], V[1], angles='xy', scale_units='xy', scale=1, color='r')
ax.annotate("(3,2)",(3,2))
glue("geom_vector", f, display=False)
```

In this class we will manipulate vectors of dimensions definitely higher than 2
or 3. Although the visualisation of these vectors is less obvious, the
geometric notions (angle,distance) that we will define for those vectors are not only valid for the
specific 2D and 3D cases but also generalize for those higher dimensions.



### Where do vectors come from ?

Besides geometry, vectors are used in various fields of science. 
For language, they appear most often in a context of statistical
analysis or data representation. We list here a few cases where you
are likely to meet high dimensional vectors:
 
**As measures of an object** Let us start with some non-linguistic examples. The first comes from
Iris. The dataset contains a sequence of measurements on each line
measures that describe a flower instance:
```
  Sepal.Length Sepal.Width Petal.Length Petal.Width
1     5.1          3.5         1.4          0.2
2     4.9          3.0         1.4          0.2
3     4.7          3.2         1.3          0.2
4     4.6          3.1         1.5          0.2
```
Each of these measurement sequences is interpreted as a vector.
For example, $\mathbf{x}^{(1)} = (5.1,3.5 ,1.4,0.2)$ and
$\mathbf{x}^{(3)} = (4.7,3.2,1.3,0.2)$ are vectors

**Color coding** Vectors of size 3 can be used to encode colors
(RGB coding). Each element of the vector is an integer between 0
and 255, indicating respectively the quantity of red, green and blue
to create a color. Thus, vector $(255,0,0)$
represents red, vector $(0,0,0)$ represents black and
vector $(120,120,120)$ represents gray.


```{code-cell}
:tags: ["remove-input"]
from myst_nb import glue
import numpy as np
import matplotlib.pyplot as plt
f,ax = plt.subplots()
ax.set_xlim([0,6])
ax.set_ylim([0,2])
ax.scatter( 1 , 1 , s = 2500,color= np.array([240,240.,0.])/255. )
ax.scatter( 2 , 1 , s = 2500,color= np.array([0.,0.,0.])/255. )
ax.scatter( 3 , 1 , s = 2500,color= np.array([255,0.,120.])/255. )
ax.scatter( 4 , 1 , s = 2500,color= np.array([255,0.,0.])/255. )
ax.scatter( 5 , 1 , s = 2500,color= np.array([120,120.,120.])/255. )
ax.set_axis_off()
ax.annotate("(240,240,0)", (1,0.7),ha='center')
ax.annotate("(0,0,0)", (2,0.7),ha='center')
ax.annotate("(255,0,120)", (3,0.7),ha='center')
ax.annotate("(255,0,0)", (4,0.7),ha='center')
ax.annotate("(120,120,120)", (5,0.7),ha='center')
glue("geom_vector", f, display=False)
```



**Counting occurrences** In a language context, numbers naturally appear
when counting in texts. For example, we can
collect statistical counts on vowels in several texts, such as : 
```
  a    e    i    o    u
1 15   30   12   0    5
2 5    12   3    7    5
3 9    15   2    6    6
4 8    7    3    1    1 
```
Thus text number 1 is represented by a vector $\mathbf{x}^{(1)} =
(15,30,12,0,5)$ whose size equals the number of vowels, whereas text number 4 is represented by the vector $\mathbf{x}^{(4)} =
( 8,7,3,1,1)$

**Listing the vocabulary used in a text** A variant
of the previous case is when the vector is valued by
booleans. For instance each element of such a vector equals 1 if the corresponding letter appears in the text and 0
otherwise:

$$
x_i  = \left\{
\begin{array}{lcl}
1&\text{if}& \text{letter in text}\\
0&&\text{otherwise}
\end{array}
\right .
$$

Assuming the vowel  count data again, we'll code $\mathbf{x}^{(1)} = (1,1,1,0,1)$ and $\mathbf{x}^{(1)} = (1,1,1,0,1)$.
and $\mathbf{x}^{(4)} = (1,1,1,1,1)$. Such vectors are called
boolean vectors.

**One hot representation for written language** Besides counting problems, when it comes to analyzing
textual data analysis, data sets do not naturally include numerical
but rather discrete symbols such as letters, words and symbolic annotations.
It is very common in machine learning and statistics for language
to encode such symbols by vectors. This amounts to
building a dictionary (a map) that associates each such symbol with a
distinct vector.

The simplest case is that of one-hot coding, which we illustrate
for vowels. We assume a
dictionary that associates each letter with a vector:

$$
\begin{array}{lccccccc}
\mathbf{x}(a) &=& 1 & 0 &0 & 0 & 0\\ 
\mathbf{x}(e) &=& 0 & 1 &0 & 0 & 0\\ 
\mathbf{x}(i) &=& 0 & 0 &1 & 0 & 0\\
\mathbf{x}(o) &=& 0 & 0 &0 & 1 & 0\\ 
\mathbf{x}(u) &=& 0 & 0 &0 & 0 & 1\\ 
\end{array}
$$

To code a word like  *oui*  from the letters that make it up
for example, we can concatenate the vectors of each of its letters:

$$
( \mathbf{x}(o) , \mathbf{x}(u) , \mathbf{x}(i)) = (0,0,0,1,0,0 ,0,0,0,1, 0,0,1,0,0) 
$$
This coding underpins many models used in language language processing.

Most of the time words are encoded as one  hot vectors.
You can actually create a dictionary that encodes
each of the words in a vocabulary by a 
one-hot vector. This encoding underlies many language models used in language processing.

**Mathematical objects** Vectors are very general mathematical
objects that may arise in a variety of contexts: *Word embeddings* are nowadays ubiquitous in Natural Language
Processing and are nothing else than vectors ;  *Neural networks* are
models that operate often on vectors ; vectors may arise in other
mathematical contexts, for example as *gradient vectors* that are
vectors of partial derivatives.


## Vector addition

Just as arithmetic expressions allow you to perform calculations with
real numbers,  just as boolean expressions can be used to perform
calculations with booleans, we can also formalize a calculation with
vectors of such numbers.


Two vectors of the same size can be added together. The 
**vector addition** generates a vector in which each element at position $i$
 is the sum of the two elements at position $i$ in each argument.
 This operation is formalized as follows:

$$
\mathbf{x} + \mathbf{y} = 
\begin{bmatrix}
x_1 \\  \vdots \\ x_n
\end{bmatrix}
+
\begin{bmatrix}
y_1\\ \vdots \\y_n
\end{bmatrix}
=
\begin{bmatrix}
x_1+y_1\\ \vdots  \\ x_n+y_n
\end{bmatrix}
$$


`````{admonition} Example
:class: tip
For example, in RGB, adding red and green gives yellow:

$$
\begin{bmatrix}
255\\ 0\\ 0
\end{bmatrix}
+
\begin{bmatrix}
0 \\ 255\\ 0
\end{bmatrix}
=
\begin{bmatrix}
255 \\ 255 \\ 0
\end{bmatrix}
$$
`````


`````{admonition} Example
:class: tip
Assuming that we have counted the occurrences of vowels
in two different texts, the addition of the two vectors yields
the number of occurrences in both texts: 

$$
\begin{bmatrix}  
  1 \\15 \\ 30 \\ 12 \\ 0 \\ 5
\end{bmatrix}
+
\begin{bmatrix}  
  2 \\ 5\\ 12 \\ 3 \\ 7 \\ 5
\end{bmatrix}
=
\begin{bmatrix}  
  3 \\ 20\\ 42 \\ 15 \\ 7 \\ 10
\end{bmatrix}
$$
`````



### Geometric interpretation of the addition


The addition of two vectors in 2D as a geometric interpretation known
as the parallelogram law of vector addition which states that if two
vectors can be represented as two sides of a parallelogram, their
addition is represented by the diagonal of the parallelogram.

Here is an example:

$$
\begin{bmatrix}
2\\ 1
\end{bmatrix}
+
\begin{bmatrix}
1\\ 1
\end{bmatrix}
=
\begin{bmatrix}
3\\ 2
\end{bmatrix}
$$ 


```{code-cell}
:tags: ["remove-input"]
from myst_nb import glue
import numpy as np
import matplotlib.pyplot as plt
f,ax = plt.subplots()
ax.set_xlim([0,4])
ax.set_ylim([0,2])
x  = [2,1]
y  = [1,1]
xy = [3,2]

ax.quiver(2, 1, 1, 1, angles='xy', scale_units='xy', scale=1, color='b',width=0.001)
ax.quiver(1, 1, 2, 1, angles='xy', scale_units='xy', scale=1, color='r',width=0.001)
ax.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='r')
ax.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, color='b')
ax.quiver(0, 0, xy[0], xy[1], angles='xy', scale_units='xy', scale=1,color='m')

ax.annotate("(0,0)",(0,0),verticalalignment='bottom',horizontalalignment='right')
ax.annotate("(2,1)",(2,1))
ax.annotate("(1,1)",(1,1))
ax.annotate("(3,2)",(3,2))

ax.set_axis_off()

glue("geom_vector", f, display=False)
```





## Multiplication by a scalar


The other operation is the **multiplication of a vector
vector by a scalar**.
Let $r$ be a real number and $\mathbf{x}$ a vector.
$r\,\mathbf{x}$ is
formalized as follows:

$$
r\, \mathbf{x} =  r \,
\begin{bmatrix}
x_1\\ \vdots\\ x_n
\end{bmatrix}
=
\begin{bmatrix}
r x_1\\ \vdots\\ r x_n
\end{bmatrix}
$$


`````{admonition} Example
:class: tip
This is illustrated concretely on an example like:


$$
-2
\begin{bmatrix}
12\\6\\77
\end{bmatrix}
=
\begin{bmatrix}
-24\\-12\\-154
\end{bmatrix}
$$

``````




## Vector space

A vector space is a set $S^n$ of vectors (or tuples) whose elements are from set
$S$. The vectors can be added together and multiplied by scalars
$s\in S$.
- Vector addition is commutative, associative and has identity element
  the vector $\mathbf{0}$
- Scalar multiplication is commutative, associative and has identity
  element the scalar 1
- The scalar multiplication distributes over vector addition : $s
  (\mathbf{x}+\mathbf{y}) = s\mathbf{x} + s\mathbf{y}$. The
  vector multiplication distributes over scalar addition: $(s+t)
  \mathbf{x} = s\mathbf{x} + t \mathbf{x}$ 



### Linear combinations

Given these algebraic properties, let us focus on the case of **linear
 combination**.
 If $\mathbf{x}_1\ldots \mathbf{x}_k$ are vectors and $s_1\ldots s_k$
are scalars, then the vector :

$$
s_1 \mathbf{x}_1+\ldots + s_k \mathbf{x}_k
$$

is a  linear combination of the vectors $\mathbf{x}_1\ldots
\mathbf{x}_k$. The scalars $s_1\ldots s_k$ are called coefficients
of the linear combination.

Note that in the special case where the vectors $\mathbf{x}_1\ldots
\mathbf{x}_k$ are one hot, the coefficient $s\ldots s_k$ is the coefficient of the linear combination.
are unitary, the coefficient $s_i$ occupies position $i$ in the
resulting vector:

$$
-3
  \begin{bmatrix}
    1\\0\\0
  \end{bmatrix}
  +
  2
    \begin{bmatrix}
    0\\1\\0
  \end{bmatrix}
  +
  \frac{1}{5}
    \begin{bmatrix}
    0\\0\\1
  \end{bmatrix}
  =
    \begin{bmatrix}
    -3\\2\\ \frac{1}{5}
  \end{bmatrix}
$$
  
When each $s_i$ coefficient is equal to 1, the linear combination amounts
 to summing the vectors $\mathbf{x}_1\ldots
\mathbf{x}_k$. When each coefficient $s_i = \frac{1}{k}$ the linear
linear combination is equivalent to averaging the vectors $\mathbf{x}_1\ldots
\mathbf{x}_k$. Finally, if the coefficients $s_i$ can be interpreted as
probabilities ($s_1+\ldots + s_k = 1$, $s_i\geq 0$), we say that the
linear combination is a
**weighted sum** or **mixture**. 



### Basis of a vector space

A **basis** is a minimal set of vectors from which any vector in the space can be expressed as a linear combination of. 
To define this formally we rely on the properties of linear independence and spanning. 


A set of vectors is **linearly independent** if no vector in
 the set can be written as a linear combination of the other vectors
 in the set. This is generally stated as follows: The set of vectors ${\mathbf{x}_1,\ldots \mathbf{x}_n}$ 
 is linearly independent iff

$$
c_1 \mathbf{x}_1 + \ldots + c_n\mathbf{x}_n = \mathbf{0}
$$  

then the set of scalar coefficients $c_1, \ldots c_n$ is the set of zero coefficients: $c_1 = 0, \ldots c_n = 0$.
A set of vectors that is not linearly independent is **linearly dependent**.



A set of vectors ${\mathbf{x}_1,\ldots \mathbf{x}_n}$ **spans** a vector space  if any vector $\mathbf{x}\in S$ can be expressed as the linear 
combination of these vectors, that is:

$$
\mathbf{v} = c_1 \mathbf{x}_1 + \ldots + c_n\mathbf{x}_n \qquad \forall \mathbf{v} \in S
$$


A set of vectors ${\mathbf{x}_1,\ldots \mathbf{x}_n}$ that are both linearly dependent and that span the vector space is a basis of the vector space.






 
  
 




## Exercises

1. Define explicitly the operation of substraction $\mathbf{x} -
  \mathbf{y}$. Choose a 2D example and draw the parallelogram law for vector substraction.
2. What is the 2D geometric interpretation of the scalar
  multiplication $r\,\mathbf{x}$ ? Illustrate with an example.
3. Given the vectors 

$$
\begin{align} 
\mathbf{x} ^{(1)} & = \begin{bmatrix}1\\0\\0\end{bmatrix}\\
\mathbf{x} ^{(2)} &=  \begin{bmatrix}4\\0\\4\end{bmatrix}\\
\mathbf{x} ^{(3)} & = \begin{bmatrix}0\\2\\0\end{bmatrix}\\
\mathbf{x}^{(4)} & = \begin{bmatrix}0\\0\\3\end{bmatrix}
\end{align}
$$ 

Three of them are linearly independent. Which ones ? Find a vector
that is linearly dependent of other vectors in this set and explain
why.  Is this vector unique ?
   
4. Create a naive one hot encoding in numpy
