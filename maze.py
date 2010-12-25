""" Script to generate mazes using low degree spanning trees"""


import pylab as pl
import pymc as mc
import networkx as nx
import random

import model
import graphics

def my_grid_graph(n):
    G = nx.grid_graph([n,n])
    for u,v in G.edges():
        G[u][v]['weight'] = random.random()

    G.pos = {}
    for v in G:
        G.pos[v] = [v[0], n-1-v[1]]

    return G

def random_maze(n=25):
    G = my_grid_graph(n)

    T = nx.minimum_spanning_tree(G)
    P = graphics.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    pl.figure(1)
    pl.clf()
    D, pos = graphics.maze(G, T, fast=True)
    pl.show()

    pl.figure(2)
    pl.clf()
    nx.draw_networkx_edges(D, pos, width=2, edge_color='k')
    nx.draw_networkx_edges(P, G.pos, width=3, alpha=1, edge_color='g')
    pl.show()
