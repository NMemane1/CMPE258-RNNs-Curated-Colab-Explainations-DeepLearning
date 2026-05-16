# Graph Neural Networks: Fundamentals — Code Walkthrough

This document explains the notebook `final_gnn_fundamentals_tutorial.ipynb`
cell by cell, and serves as the script for the accompanying video.

**Notebook goal:** build up Graph Neural Networks from absolute scratch — what
graphs are, how to represent them, why ordinary ML fails on them, the message-
passing paradigm, and finally a Graph Convolutional Network (GCN) implemented
twice (NumPy with hand-written backprop, then PyTorch) and trained on a real
social network.

Mostly NumPy + NetworkX + Matplotlib, with a PyTorch section at the end. Runs on
CPU fine.

---

## Setup

### Cell 1 — Dependencies
A commented-out `pip install` line (torch, torch-geometric, networkx, etc.) for
Colab.

### Cell 2 — Imports
NumPy, Matplotlib, NetworkX, typing. Sets `np.random.seed(42)`.

## Chapter 1 — What are graphs?

### Cell 4 — `visualize_graph_example`
Draws four graph types side by side with NetworkX: **undirected**, **directed**
(arrows), **weighted** (edge labels), and **bipartite** (two node groups, like a
user–item recommendation graph). Establishes the vocabulary visually.

## Chapter 2 — Graph representations

### Cell 7 — `demonstrate_adjacency_matrix`
Builds a small graph and shows its **adjacency matrix** `A` — `A[i,j]=1` if an edge
connects `i` and `j`. Draws the graph next to the matrix as a heatmap and verifies
the matrix is symmetric (undirected graph).

### Cell 9 — `compare_representations`
Shows the same graph in four formats: **edge list**, **adjacency matrix**,
**adjacency list**, and **COO format** (`edge_index` — the two-row source/target
array PyTorch Geometric uses). Ends with a memory comparison: a dense matrix is
O(n²) while edge/COO formats are O(E) — the reason real GNN libraries use sparse
formats.

### Cell 11 — `Graph` class
A reusable graph class used by the rest of the notebook. It stores the adjacency
matrix, an adjacency list, and a **node feature matrix `X`** (`[num_nodes,
feature_dim]`, defaulting to one-hot if none given). Provides `get_neighbors`,
`degree`, and `degree_matrix`. Node features are the crucial part — GNNs operate on
them.

## Chapter 3 — Why do we need GNNs?

### Cell 13 — `demonstrate_permutation_problem`
The central motivation. It takes one graph, **permutes the node ordering**
(`A_permuted = P·A·Pᵀ`), and shows that the two adjacency matrices look different
even though they are the *same graph*. A naive model that flattens the matrix
would give different answers for the same graph. The lesson: a good graph model
must be **permutation-invariant/equivariant** — which is exactly what message
passing guarantees.

## Chapter 4 — The message-passing paradigm

### Cell 16 — `visualize_message_passing`
A three-panel diagram of one round of message passing for a single node:
(1) initial features, (2) **aggregate** — the node collects messages from its
neighbors, (3) **update** — the node computes a new representation. This
aggregate-then-update loop is the core of every GNN.

### Cell 18 — `visualize_receptive_field`
Uses Zachary's Karate Club graph to show that with each layer a node's
**receptive field** grows by one hop — after L layers a node "sees" all nodes
within L hops. This is why stacking too many GNN layers causes *over-smoothing*.

### Cell 20 — `simple_message_passing`
Message passing with **no learnable parameters**. For each node it aggregates
neighbor features (`mean`, `sum`, or `max`) and blends them 50/50 with the node's
own features. Running it for 3 rounds shows features becoming more similar as
information propagates — a concrete demo of both message passing and
over-smoothing.

## Chapter 5 — Graph Convolutional Network in NumPy

### Cell 23 — `compute_normalized_adjacency`
The GCN normalization: `Â = D̃^(−1/2) · Ã · D̃^(−1/2)` where `Ã = A + I` adds
**self-loops** (so a node keeps its own features) and the symmetric degree
normalization stops high-degree nodes from dominating. The output confirms the row
sums are ≈1.

### Cell 25 — `GCNLayerNumPy`
One GCN layer, `H' = σ(Â·H·W + b)`, in raw NumPy — including a **hand-written
backward pass**. Forward: aggregate (`Â·H`), linear transform, activation. Backward:
gradient through the activation, then `dW`, `db`, and `dH`, with an SGD weight
update. He initialization for the weights. This is the cell that demystifies "what
a GCN layer actually computes."

### Cell 27 — `GCNNumPy`
Stacks `GCNLayerNumPy` objects into a full network (ReLU on hidden layers, linear
output). Also defines a numerically stable `softmax` and a `cross_entropy_loss`
that returns both loss and gradient. Demo: a 4→8→3 network forward pass.

### Cell 29 — `train_gcn_numpy`
A full training run in pure NumPy. Builds a 9-node graph with three planted
communities (3 nodes each), gives each community a feature pattern, and trains the
GCN for 200 epochs on a few labeled nodes — **semi-supervised node
classification**. Plots loss, train/test accuracy, and the final graph colored by
predicted class.

## Chapter 6 — GCN in PyTorch

### Cell 31 — PyTorch imports
`torch`, `nn`, `functional`, `optim`; device setup.

### Cell 32 — `GCNLayerPyTorch` and `GCN`
The same layer as a proper `nn.Module`: a learnable `weight` parameter (Glorot/
Xavier init) and optional bias; forward is `Â·(H·W) + b` — **autograd handles the
backward pass automatically**, so no manual gradients. `GCN` stacks layers with
ReLU and dropout between them (dropout only in training mode). Also a Torch version
of the normalized-adjacency helper.

### Cell 34 — `train_gcn_pytorch`
Trains the PyTorch GCN on the *same* 9-node, 3-community graph. Uses the Adam
optimizer with `weight_decay` (L2 regularization), `F.cross_entropy`, and the
standard `zero_grad → forward → loss → backward → step` loop. Plots the same three
panels. This cell is the direct comparison: identical math, far less code than the
NumPy version.

## Chapter 7 — Real data

### Cell 36 — `karate_club_demo`
Node classification on **Zachary's Karate Club** — a famous 34-node social network
that historically split into two factions. Ground-truth labels are the two clubs.
The model is trained on only 6 labeled nodes (one-hot node IDs as features) and
must classify the rest — true semi-supervised learning. Plots the loss, test
accuracy, and the network colored by prediction with training nodes outlined. The
GCN typically classifies nearly all nodes correctly, recovering the real social
split from almost no labels.

---

## Key takeaways
- A **graph** is nodes + edges; it can be stored as a matrix, edge list, adjacency
  list, or COO `edge_index`.
- Ordinary ML fails on graphs because flattening is **not permutation-invariant**.
- **Message passing** (aggregate neighbors → update) is the universal GNN
  operation and is permutation-equivariant by construction.
- A **GCN layer** is `σ(Â·H·W)` with `Â` the symmetric-normalized adjacency
  (self-loops included).
- Implementing it in NumPy (manual backprop) then PyTorch (autograd) shows the
  *same math* with and without a framework.
- GCNs do **semi-supervised node classification** well — Karate Club is recovered
  from a handful of labels.
