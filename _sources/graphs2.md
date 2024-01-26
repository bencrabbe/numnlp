# Applications to graph analysis : the PageRank algorithm


This chapter illustrates an application of eigenvalues and eigenvectors to graph analytics with the PageRank algorithm. PageRank is an algorithm that maps a set of web pages to real numbers that can be used to **rank** them.

The algorithm views the web as a graph whose nodes are web pages and whose edges are hyperlinks leading from one page to another. It views the structure of the web graph as a matrix. The algorithm outputs a vector of scores, with one score for each web page and it turns out that the output vector is actually the first eigenvector of the graph adjacency matrix. 


## Representation: the web as a graph

A set of web pages can be represented as a directed graph $G=\langle V, E\rangle$ whose vertices are the actual web pages.
The edges relating the vertices are inferred from hyperlinks: a link sending an internet surfer from page $p_1$ to page $p_2$ is thus an edge $(p_1,p_2) \in E$ of the directed graph.

PageRank is an algorithm that takes advantage of the graph structure to rank the pages rathen than the page contents. It  builds upon the idea
that a web page is important if many pages are pointing to it. In graph terminology we could use the in degree of the nodes to compute their importance. But PageRank goes one step further. It not only counts the incoming edges on the nodes but also weights their importance by the importance of the source node. Thus the page rank is a global computation on the network

```{figure} figs/pr.png
---
height: 250px
name: pagerank-fig
---
The PageRank algorithm weights each nodes as a function of the importance of its incoming edges
```

## Markov chains and random walks

Graphs can be conveniently represented by matrices. 
A graphs's **adjacency matrix** can be used to encode the graph structure.

Let us consider the graph

````{code-cell}
---
height: 250px
name: simple-graph
tags: remove-input
---
import networkx as nx
import matplotlib as plt
plt.rcParams['figure.dpi'] = 50

G = nx.DiGraph()
G.add_nodes_from(["1","2","3","4"])
G.add_edge("1","2")
G.add_edge("1","3")
G.add_edge("2","3")
G.add_edge("2","4")
G.add_edge("3","4")

pos = {"1":(0,0),"2":(1,0),"3":(1,1),"4":(2,1)}

nx.draw(G,pos=pos,with_labels=True,node_color="skyblue",alpha=0.8,node_size=500)
````

with adjacency matrix:

```{admonition} Exemple
:class:tip

$$
\mathbf{A} =
\begin{bmatrix}
0&1&1&0\\
0&0&1&1\\
0&0&0&1\\
0&0&0&0
\end{bmatrix}
$$

```


## PageRank



import wikipedia
import networkx as nx
from collections import Counter

nodeset = ["Luke Skywalker","Admiral Ackbar","Boba Fett","C-3PO","Chewbacca","Count Dooku","Darth Maul","Darth Vader","General Grievous",
        "Greedo","Han Solo","Jabba the Hutt","Jango Fett","Jar Jar Binks","Lando Calrissian","Leia Organa","Mace Windu",
        "Obi-Wan Kenobi","Padmé Amidala","Palpatine","Qui-Gon Jinn","R2-D2","Saw Gerrera","Yoda"]

nodes2pages = {node:wikipedia.page(node,auto_suggest=False) for node in nodeset}

graph = nx.DiGraph()

for node in nodeset:
    graph.add_node(node,title=node)
    
for node in nodeset:
    
    cpage = nodes2pages[node]
    for target_node in nodeset:
        if target_node != node:
            if target_node.lower() in cpage.summary.lower():
                graph.add_edge(node,target_node)
            


import matplotlib as plt
plt.rcParams['figure.dpi'] = 150

pos = {nidx:(2*x,y) for nidx,(x,y) in nx.shell_layout(graph).items()}

nx.draw(graph,pos=pos,with_labels = True,alpha=0.8,node_color="skyblue",edge_color="lightgray")



pr = [(pr,node) for node,pr in nx.pagerank(graph).items()]
pr.sort(reverse=True)
for rank,node in pr:
    print(node,rank)
