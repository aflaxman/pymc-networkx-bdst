""" Script to generate mazes using grid graphs and spanning trees"""

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
    P = model.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    D = model.dual_grid(G, T)
    pos = graphics.layout_maze(D, fast=True)
    graphics.plot_maze(D, pos, P, G.pos)


def hidden_image_maze(fname, n=25, style='jittery'):
    """ Supported styles: jittery, smooth, sketch"""
    H = model.image_grid_graph(fname, n=n)  # get a subgraph of the grid corresponding to edges between black pixels
    G = H.base_graph

    # for every edge in H, make the corresponding edge in H have weight 0
    for u,v in H.edges():
        G[u][v]['weight'] = 0

    # find a minimum spanning tree on G (which will include the maze solution)
    T = nx.minimum_spanning_tree(G)

    # find the maze solution in the spanning tree
    P = model.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    # generate the dual graph, including edges not crossed by the spanning tree
    D = model.dual_grid(G, T)
    graphics.add_maze_boundary(D, n)
    graphics.make_entry_and_exit(D, n)
    pos = graphics.layout_maze(D, fast=(style == 'jittery'))
    graphics.plot_maze(D, pos, P, G.pos)

    # make it stylish if requested
    if style == 'sketch':
        pl.figure(1)
        D_pos = graphics.layout_maze(D, fast=True)
        nx.draw_networkx_edges(D, D_pos, width=1, edge_color='k')
        D_pos = graphics.layout_maze(D, fast=True)
        nx.draw_networkx_edges(D, D_pos, width=1, edge_color='k')

    
    # show the pixel colors loaded from the file, for "debugging"
    pl.figure(2)
    for v in G:
        pl.plot([G.pos[v][0]], [G.pos[v][1]], '.', alpha=.5, color=G.node[v]['color'])


def ld_maze(n=25):
    """ having many low-degree vertices makes for hard mazes

    unfortunately, finding them is slow"""

    # start with an nxn square grid
    G = model.my_grid_graph(n)

    # make a pymc model of a low-degree spanning tree on this
    T = model.LDST(G, beta=10)
    mod_mc = mc.MCMC([T])
    mod_mc.use_step_method(model.STMetropolis, T)
    mod_mc.sample(100, burn=99)
    T = T.value

    P = model.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    D = model.dual_grid(G, T)
    graphics.add_maze_boundary(D, n)
    graphics.make_entry_and_exit(D, n)
    D = graphics.split_edges(D)
    D = graphics.split_edges(D)
    D_pos = graphics.layout_maze(D, fast=False)
    graphics.plot_maze(D, D_pos, P, G.pos)


def border_maze(fname='jessi.png', n=100):
    G = model.image_grid_graph(fname, colors=set([(255,255,255,255), (0,0,0,255)]), n=n)  # get a subgraph of the grid corresponding to edges between black and white
    H = model.image_grid_graph(fname, colors=set([(0,0,0,255)]), n=n)  # get a subgraph of the grid corresponding to edges between black pixels
    
    # for every edge in H, make the corresponding edge in G have weight 0
    for u,v in G.edges():
        G[u][v]['weight'] = (H.has_edge(u,v) and -1.) or G.base_graph[u][v]['weight']

    # find a minimum spanning tree on G (which will include the maze solution)
    T = nx.minimum_spanning_tree(G)

    # add border edges to G
    B = model.image_grid_graph(fname, colors=set([(255,255,255,255), (0,0,0,255), (255,0,0,255)]), n=n)
    print B.number_of_edges()
    for u,v in B.edges():
        G.add_edge(u,v)

    # find the maze solution in the spanning tree
    P = model.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))
    G_pos = G.base_graph.pos

    # generate the dual graph, including edges not crossed by the spanning tree
    D = model.dual_grid(G, T)
    pos = graphics.layout_maze(D, fast=True)
    graphics.plot_maze(D, pos, P, G_pos)

    # show the pixel colors loaded from the file, for "debugging"
    pl.figure(2)
    for v in G:
        pl.plot([G_pos[v][0]], [G_pos[v][1]], '.', alpha=.5, color=G.base_graph.node[v]['color'])

    return G, H, T, B
