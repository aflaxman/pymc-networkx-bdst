import networkx as nx
import pylab as pl

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

def my_path_graph(path):
    G = nx.Graph()
    G.add_path(path)
    return G

def plot_graph_and_tree(G, T, time):
    pl.clf()
    nx.draw_networkx_edges(G, G.pos, alpha=.75, width=.5, style='dotted')
    nx.draw_networkx_edges(T, G.pos, alpha=.5, width=2)
    X = pl.array(G.pos.values())
    pl.plot(X[:,0], X[:,1], 'bo', alpha=.5)
    pl.plot([G.pos[T.root][0]], [G.pos[T.root][1]], 'bo', ms=12, mew=4, alpha=.95)

    # display the most recently swapped edges
    P = my_path_graph(T.path)
    nx.draw_networkx_edges(P, G.pos, alpha=.25 + (1-time)*.5, width=4, edge_color='c')
    P = my_path_graph([T.u_new, T.v_new])
    P.add_edge(T.u_old, T.v_old)
    nx.draw_networkx_edges(P, G.pos, alpha=.25 + (1-time)*.5, width=4, edge_color='y')

    # find and display the current longest path
    path = nx.shortest_path(T, T.root)
    furthest_leaf = max(path, key=lambda l: len(path[l]))
    P = my_path_graph(path[furthest_leaf])
    if len(path[furthest_leaf]) <= T.k:
        col = 'g'
    else:
        col = 'r'
    nx.draw_networkx_edges(P, G.pos, alpha=.5, width=4, edge_color=col)
    pl.text(G.pos[furthest_leaf][0], G.pos[furthest_leaf][1], '%d hops from root'%len(path[furthest_leaf]), color=col, alpha=.8, fontsize=9)
    T.depth = len(path[furthest_leaf])

def dual_edge(u, v):
    """ Helper function to map an edge in a lattice to corresponding
    edge in dual lattice (it's just a rotation)

    >>> dual_edge((0,0),
    (0,1)) ((-0.5, 0.5), (0.5, 0.5))
    """
    mx = .5 * (u[0] + v[0])
    my = .5 * (u[1] + v[1])
    dx = .5 * (u[0] - v[0])
    dy = .5 * (u[1] - v[1])
    return ((mx+dy, my+dx), (mx-dy, my-dx))

def maze(G, T, fast=True):
    """ Make a maze from the dual of the base graph minus the dual of the tree

    Assumes that G is the base graph is a grid with integer labels
    Note that T doesn't have to be a tree
    """

    n = pl.sqrt(T.number_of_nodes())

    D = nx.Graph()

    # add dual complement edges
    for v in T.nodes():
        for u in G[v]:
            if not T.has_edge(u,v):
                D.add_edge(*dual_edge(u,v))

    # add boundry edges
    for i in range(n):
        D.add_edge((-.5, i-.5), (-.5, i+.5))
        D.add_edge((n-.5, i-.5), (n-.5, i+.5))
        D.add_edge((i-.5, -.5), (i+.5, -.5))
        D.add_edge((i-.5, n-.5), (i+.5, n-.5))

    # remove edges for start and end
    D.remove_edge((-.5,-.5), (-.5, .5))
    D.remove_edge((n-.5,n-1.5), (n-.5, n-.5))
    
    D = split_edges(D)
    D = split_edges(D)

    pos = {}
    for v in D.nodes():
        pos[v] = (v[0], n-1-v[1])
        
    # adjust node positions so they don't look so square
    if not fast:
        spring_pos = nx.spring_layout(D, pos=pos, fixed=set(D.nodes()) & set([(2*i-.5, 2*j-.5) for i in range(n/2) for j in range(n/2)]), iterations=10)

    eps = .99
    my_avg = lambda x, y: (x[0]*(1.-eps) + y[0]*eps, x[1]*(1.-eps)+y[1]*eps)
    for v in pos:
        if fast:
            # splitting and jittering looks like shakey pen
            pos[v] = [pos[v][0] + .05*eps*pl.randn(), pos[v][1] + .05*eps*pl.randn()]
        else:
            # splitting and springing looks pretty and curvy
            pos[v] = my_avg(pos[v], spring_pos[v])
        
    nx.draw_networkx_edges(D, pos, alpha=1., width=2, edge_color='k')
    pl.axis([-1, n, -1, n])
    pl.axis('off')
    pl.subplots_adjust(0, 0, 1, 1)

    return D, pos
