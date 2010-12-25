""" Script to generate mazes using low degree spanning trees"""


import pylab as pl
import pymc as mc
import networkx as nx
import random

import model
import graphics
reload(model); reload(graphics)

def random_maze(n=25):
    G = model.my_grid_graph(n)

    T = nx.minimum_spanning_tree(G)
    P = graphics.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    D, pos = graphics.maze(G, T, fast=True)
    plot_maze(D, pos, P, G.pos)


def hidden_image_maze(fname, n=25):
    H = model.image_grid_graph(fname, n=n)
    G = H.base_graph
    for u,v in G.edges():
        if H.has_edge(u,v):
            G[u][v]['weight'] = 0
    T = nx.minimum_spanning_tree(G)
    P = graphics.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    D, pos = graphics.maze(G, T, fast=True)
    plot_maze(D, pos, P, G.pos)

    for v in G:
        pl.plot([G.pos[v][0]], [G.pos[v][1]], '.', alpha=.5, color=G.node[v]['color'])


def ld_maze(n=25):
    G = model.my_grid_graph(n)

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
