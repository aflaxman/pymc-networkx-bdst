import networkx as nx
import pylab as pl
import model

def split_edges(G):
    """ Make a new graph with edges split
    Assumes that all edges are named (x1,y1), (x2,y2))
    """
    H = nx.Graph()
    for u,v in G.edges():
        x1,y1=u
        x2,y2=v
        x3,y3=.5*(x1+x2), .5*(y1+y2)
        H.add_edge((x1,y1), (x3,y3))
        H.add_edge((x2,y2), (x3,y3))
    return H

def plot_graph_and_tree(G, T, time):
    """ Plot a graph G and a tree T on top of it

    Assumes that G has an embedding in the plane, represented as a dictionary G.pos"""

    pl.clf()
    nx.draw_networkx_edges(G, G.pos, alpha=.75, width=.5, style='dotted')
    nx.draw_networkx_edges(T, G.pos, alpha=.5, width=2)
    X = pl.array(G.pos.values())
    pl.plot(X[:,0], X[:,1], 'bo', alpha=.5)
    pl.plot([G.pos[T.root][0]], [G.pos[T.root][1]], 'bo', ms=12, mew=4, alpha=.95)

    # display the most recently swapped edges
    P = model.my_path_graph(T.path)
    nx.draw_networkx_edges(P, G.pos, alpha=.25 + (1-time)*.5, width=4, edge_color='c')
    P = model.my_path_graph([T.u_new, T.v_new])
    P.add_edge(T.u_old, T.v_old)
    nx.draw_networkx_edges(P, G.pos, alpha=.25 + (1-time)*.5, width=4, edge_color='y')

    # find and display the current longest path
    path = nx.shortest_path(T, T.root)
    furthest_leaf = max(path, key=lambda l: len(path[l]))
    P = model.my_path_graph(path[furthest_leaf])
    if len(path[furthest_leaf]) <= T.k:
        col = 'g'
    else:
        col = 'r'
    nx.draw_networkx_edges(P, G.pos, alpha=.5, width=4, edge_color=col)
    pl.text(G.pos[furthest_leaf][0], G.pos[furthest_leaf][1], '%d hops from root'%len(path[furthest_leaf]), color=col, alpha=.8, fontsize=9)
    T.depth = len(path[furthest_leaf])

def add_maze_boundary(D, shape):
    for i in pl.arange(shape[0]):
        D.add_edge((i-.5, -.5), (i+.5, -.5))
        D.add_edge((i-.5, shape[1]-.5), (i+.5, shape[1]-.5))
    for i in pl.arange(shape[1]):
        D.add_edge((-.5, i-.5), (-.5, i+.5))
        D.add_edge((shape[0]-.5, i-.5), (shape[0]-.5, i+.5))

def make_entry_and_exit(D, shape):
    D.remove_edge((-.5,-.5), (-.5, .5))
    D.remove_edge((shape[0]-.5,shape[1]-1.5), (shape[0]-.5, shape[1]-.5))

def layout_maze(D, fast=True):
    """ Generate position dict for points in D
    fast : bool, optional, random perturbation (fast) or spring embedding (slow)?
    """
    pos = {}
    for v in D.nodes():
        pos[v] = (v[0], -v[1])
        
    # adjust node positions so they don't look so square
    if not fast:
        spring_pos = nx.spring_layout(D, pos=pos, fixed=set(D.nodes()) & set([(2*i-.5, 2*j-.5) for i in pl.arange(len(D)) for j in pl.arange(len(D))]), iterations=10)

    eps = .99
    my_avg = lambda x, y: (x[0]*(1.-eps) + y[0]*eps, x[1]*(1.-eps)+y[1]*eps)
    for v in pos:
        if fast:
            # splitting and jittering looks like shakey pen
            pos[v] = [pos[v][0] + .05*eps*pl.randn(), pos[v][1] + .05*eps*pl.randn()]
        else:
            # splitting and springing looks pretty and curvy
            pos[v] = my_avg(pos[v], spring_pos[v])

    return pos

def plot_maze(D, D_pos, P, P_pos):
    pl.figure(1)
    pl.clf()
    nx.draw_networkx_edges(D, D_pos, width=2, edge_color='k')
    undecorate_plot(max(P.nodes()))
    pl.show()

    pl.figure(2)
    pl.clf()
    nx.draw_networkx_edges(D, D_pos, width=2, edge_color='k')
    nx.draw_networkx_edges(P, P_pos, width=3, alpha=1, edge_color='g')
    undecorate_plot(max(P.nodes()))
    pl.show()


def undecorate_plot(shape):
    pl.axis([-1, shape[0], -shape[1], 1])
    pl.axis('off')
    pl.subplots_adjust(.01, .01, .99, .99)
