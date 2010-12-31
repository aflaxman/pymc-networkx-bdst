""" Script to generate mazes using grid graphs and spanning trees"""

import pylab as pl
import pymc as mc
import networkx as nx
import random

import models
import views
reload(models); reload(views)

def random_maze(n=25):
    G = models.my_grid_graph([n,n])

    T = nx.minimum_spanning_tree(G)
    P = models.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    D = models.dual_grid(G, T)
    pos = views.layout_maze(D, fast=True)
    views.plot_maze(D, pos, P, G.pos)


def hidden_image_maze(fname, style='jittery'):
    """ Supported styles: jittery, smooth, sketch"""
    H = models.image_grid_graph(fname)  # get a subgraph of the grid corresponding to edges between black pixels
    G = H.base_graph

    # for every edge in H, make the corresponding edge in H have weight 0
    for u,v in H.edges():
        G[u][v]['weight'] = 0

    # find a minimum spanning tree on G (which will include the maze solution)
    T = nx.minimum_spanning_tree(G)

    # find the maze solution in the spanning tree
    P = models.my_path_graph(nx.shortest_path(T, (0,0), max(H.nodes())))

    # generate the dual graph, including edges not crossed by the spanning tree
    D = models.dual_grid(G, T)
    views.add_maze_boundary(D, max(G.nodes()))
    views.make_entry_and_exit(D, max(G.nodes()))
    pos = views.layout_maze(D, fast=(style == 'jittery'))
    views.plot_maze(D, pos, P, G.pos)

    # make it stylish if requested
    if style == 'sketch':
        pl.figure(1)
        D_pos = views.layout_maze(D, fast=True)
        nx.draw_networkx_edges(D, D_pos, width=1, edge_color='k')
        D_pos = views.layout_maze(D, fast=True)
        nx.draw_networkx_edges(D, D_pos, width=1, edge_color='k')

    
    # show the pixel colors loaded from the file, for "debugging"
    pl.figure(2)
    for v in G:
        pl.plot([G.pos[v][0]], [G.pos[v][1]], '.', alpha=.5, color=G.node[v]['color'])


def ld_maze(n=25):
    """ having many low-degree vertices makes for hard mazes

    unfortunately, finding them is slow"""

    # start with an nxn square grid
    G = models.my_grid_graph([n,n])

    # make a pymc model of a low-degree spanning tree on this
    T = models.LDST(G, beta=10)
    mod_mc = mc.MCMC([T])
    mod_mc.use_step_method(models.STMetropolis, T)
    mod_mc.sample(100, burn=99)
    T = T.value

    P = models.my_path_graph(nx.shortest_path(T, (0,0), (n-1, n-1)))

    D = models.dual_grid(G, T)
    views.add_maze_boundary(D, [n,n])
    views.make_entry_and_exit(D, [n,n])
    D = views.split_edges(D)
    D = views.split_edges(D)
    D_pos = views.layout_maze(D, fast=False)
    views.plot_maze(D, D_pos, P, G.pos)


def border_maze(fname='jessi.png', fast=True):
    G = models.image_grid_graph(fname, colors=set([(255,255,255,255), (0,0,0,255)]))  # get a subgraph of the grid corresponding to edges between black and white
    H = models.image_grid_graph(fname, colors=set([(0,0,0,255)]))  # get a subgraph of the grid corresponding to edges between black pixels
    
    # for every edge in H, make the corresponding edge in G have weight 0
    for u,v in G.edges():
        G[u][v]['weight'] = (H.has_edge(u,v) and .1) or (1.+G.base_graph[u][v]['weight'])

    # find a minimum spanning tree on G (which will include the maze solution)
    T = nx.minimum_spanning_tree(G)

    # add border edges to G
    B = models.image_grid_graph(fname, colors=set([(255,255,255,255), (0,0,0,255), (255,0,0,255)]))
    for u,v in B.edges():
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    # find the maze solution in the spanning tree
    for u,v in G.edges():
        if len(G[u]) < 4 and len(G[v]) < 4:
            G[u][v]['weight'] = 1
        else:
            G[u][v]['weight'] = 1e6
    for u,v in T.edges():
        G[u][v]['weight'] = 0
    P = models.my_path_graph(nx.shortest_path(G, (0,0), max(G.nodes()), weighted=True))
    G_pos = G.base_graph.pos

    # generate the dual graph, including edges not crossed by the spanning tree
    D = models.dual_grid(G, T)
    D = views.split_edges(D)
    pos = views.layout_maze(D, fast=fast)
    views.plot_maze(D, pos, P, G_pos)

    # show the pixel colors loaded from the file, for "debugging"
    pl.figure(2)
    for v in G:
        pl.plot([G_pos[v][0]], [G_pos[v][1]], '.', alpha=.5, color=G.base_graph.node[v]['color'])

    return dict(G=G, H=H, T=T, P=P, B=B)
