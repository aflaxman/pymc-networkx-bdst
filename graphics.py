import networkx as nx
import pylab as pl


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

def maze(T):
    """ Make a maze from the dual of the base graph minus the dual of the tree

    Assumes that base graph is a grid with integer labels"""

    n = pl.sqrt(T.number_of_nodes())

    D = nx.Graph()

    # add dual complement edges
    for v in T.nodes():
        for u in T.base_graph[v]:
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

    pos = {}
    for v in D.nodes():
        pos[v] = v

    nx.draw_networkx_edges(D, pos, alpha=1., width=2, edge_color='k')
    pl.axis([-1, n, -1, n])
