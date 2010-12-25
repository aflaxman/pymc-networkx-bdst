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

    D, pos = graphics.maze(G, T, fast=True)
    plot_maze(D, pos, P, G.pos)


def ld_maze(n=25):
    G = my_grid_graph(n)

    T = model.LDST(G, beta=10)
    mod_mc = mc.MCMC([T])
    mod_mc.use_step_method(model.STMetropolis, T)
    mod_mc.sample(100, burn=99)
    T = T.value

    P = graphics.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    D, D_pos = graphics.maze(G, T, fast=False)
    plot_maze(D, D_pos, P, G.pos)


def plot_maze(D, D_pos, P, P_pos):
    pl.figure(1)
    pl.clf()
    nx.draw_networkx_edges(D, D_pos, width=2, edge_color='k')
    graphics.undecorate_plot(pl.sqrt(len(P_pos)))
    pl.show()

    pl.figure(2)
    pl.clf()
    nx.draw_networkx_edges(D, D_pos, width=2, edge_color='k')
    nx.draw_networkx_edges(P, P_pos, width=3, alpha=1, edge_color='g')
    graphics.undecorate_plot(pl.sqrt(len(P_pos)))
    pl.show()
