""" A question on cstheory stackexchange [1] got me curious about MCMC for spanning trees

Here is a model to explore the issue

[1] http://cstheory.stackexchange.com/questions/3913/how-can-i-randomly-generate-bounded-height-spanning-trees
"""

import pylab as pl
import pymc as mc
import networkx as nx
import random

def BDST(G, root=0, k=5, beta=1.):
    """ Create a PyMC Stochastic for a Bounded Depth Spanning Tree on
    base graph G

    Parameters
    ----------
    G : nx.Graph, base graph to span
    k : int, depth bound parameter
    beta : float, "inverse-temperature parameter" for depth bound
    """

    T = nx.minimum_spanning_tree(G)
    T.base_graph = G

    @mc.stoch(dtype=nx.Graph)
    def bdst(value=T, root=root, k=k, beta=beta):
        path_len = pl.array(nx.shortest_path_length(value, root).values())
        return -beta * pl.sum(path_len > k)

    return bdst

class BDSTMetropolis(mc.Metropolis):
    """ A PyMC Step Method that walks on spanning trees by adding a
    uniformly random edge not in the tree, removing a uniformly random
    edge from the cycle created, and keeping it with the appropriate
    Metropolis probability (no Hastings factor necessary, because the
    chain is reversible, right?)

    Parameters
    ----------
    stochastic : nx.Graph that is a tree and has a base_graph which it
                 spans
    """
    def __init__(self, stochastic):
        # Initialize superclass
        mc.Metropolis.__init__(self, stochastic, scale=1., proposal_sd='custom', proposal_distribution='custom', verbose=None, tally=False)

    def propose(self):
        """ Add an edge and remove an edge from the cycle that it creates"""
        T = self.stochastic.value

        T.u_new, T.v_new = T.edges()[0]
        while T.has_edge(T.u_new, T.v_new):
            T.u_new, T.v_new = random.sample(T.base_graph.edges(), 1)[0]

        T.path = nx.shortest_path(T, T.u_new, T.v_new)
        i = random.randrange(len(T.path)-1)
        T.u_old, T.v_old = T.path[i], T.path[i+1]
        
        T.remove_edge(T.u_old, T.v_old)
        T.add_edge(T.u_new, T.v_new)
        self.stochastic.value = T

    def reject(self):
        """ Restore the graph to its state before more recent edge swap"""
        T = self.stochastic.value
        T.add_edge(T.u_old, T.v_old)
        T.remove_edge(T.u_new, T.v_new)
        self.stochastic.value = T

def my_path_graph(path):
    G = nx.Graph()
    G.add_path(path)
    return G

def anneal_experiment(n=11, depth=10):
    beta = mc.Uninformative('beta', value=1.)

    G = nx.grid_graph([n, n])
    root = (5,5)
    bdst = BDST(G, root, depth, beta)

    @mc.deterministic
    def max_depth(T=bdst, root=root):
        shortest_path_length = nx.shortest_path_length(T, root)
        T.max_depth = max(shortest_path_length.values())
        return T.max_depth

    mod_mc = mc.MCMC([beta, bdst, max_depth])
    mod_mc.use_step_method(BDSTMetropolis, bdst)
    mod_mc.use_step_method(mc.NoStepper, beta)

    ni = 5

    for i in range(ni):
        beta.value = i*5
        mod_mc.sample(1000, thin=10)
        print 'cur depth', max_depth.value
        print 'pct of trace with max_depth <= depth', pl.mean(max_depth.trace() <= depth)
    return bdst.value

def anneal_graphics(n=11, depth=10):
    beta = mc.Uninformative('beta', value=1.)

    G = nx.grid_graph([n, n])
    G.orig_pos = dict([[v, v] for v in G.nodes_iter()])
    G.pos = dict([[v, v] for v in G.nodes_iter()])

    root = (5,5)
    bdst = BDST(G, root, depth, beta)

    mod_mc = mc.MCMC([beta, bdst])
    mod_mc.use_step_method(BDSTMetropolis, bdst)
    mod_mc.use_step_method(mc.NoStepper, beta)


    ni = 5
    nj = 100
    nk = 5

    for i in range(ni):
        beta.value = i*5
        for j in range(nj):
            mod_mc.sample(1)
            T = bdst.value
            
            for k in range(nk):
                if random.random() < .75:
                    delta_pos = nx.spring_layout(T, pos=G.pos, fixed=[root], iterations=1)
                else:
                    delta_pos = G.orig_pos
                eps=.01
                my_avg = lambda x, y: (x[0]*(1.-eps) + y[0]*eps, x[1]*(1.-eps)+y[1]*eps)
                for v in G.pos:
                    G.pos[v] = my_avg(G.pos[v], delta_pos[v])

                pl.clf()
                nx.draw_networkx_edges(G, G.pos, alpha=.75, width=.5, style='dotted')
                nx.draw_networkx_edges(T, G.pos, alpha=.5, width=2)
                X = pl.array(G.pos.values())
                pl.plot(X[:,0], X[:,1], 'bo', alpha=.5)
                pl.plot([G.pos[root][0]], [G.pos[root][1]], 'bo', ms=12, mew=4, alpha=.95)

                # display the most recently swapped edges
                P = my_path_graph(T.path)
                nx.draw_networkx_edges(P, G.pos, alpha=.25 + (nk-k)*.5/nk, width=4, edge_color='g')
                P = my_path_graph([T.u_new, T.v_new])
                P.add_edge(T.u_old, T.v_old)
                nx.draw_networkx_edges(P, G.pos, alpha=.25 + k*.5/nk, width=4, edge_color='y')

                # find and display the current longest path
                path = nx.shortest_path(bdst.value, root)
                furthest_leaf = max(path, key=lambda l: len(path[l]))
                P = my_path_graph(path[furthest_leaf])
                nx.draw_networkx_edges(P, G.pos, alpha=.5, width=4, edge_color='r')
                pl.text(G.pos[furthest_leaf][0], G.pos[furthest_leaf][1], '%d hops from root'%len(path[furthest_leaf]), color='r', alpha=.8, fontsize=9)
                str = ''
                str += ' beta: %.1f\n' % beta.value
                str += ' cur depth: %d (target: %d)\n' % (len(path[furthest_leaf]), depth)
                sm = mod_mc.step_method_dict[bdst][0]
                str += ' accepted: %d of %d\n' % (sm.accepted, sm.accepted + sm.rejected)
                pl.figtext(0, 0, str)
                pl.figtext(1, 0, 'healthyalgorithms.wordpress.com \n', ha='right')
                pl.axis([-1, n, -1, n])
                pl.axis('off')
                pl.subplots_adjust(0, 0, 1, 1)
                pl.savefig('t%06d.png' % (i*nj*nk + j*nk + k))
            print 'accepted:', mod_mc.step_method_dict[bdst][0].accepted
            print 'cur depth:', len(path[furthest_leaf])

    import subprocess
    subprocess.call('mencoder mf://*.png -mf w=800:h=600 -ovc x264 -of avi -o G_%d_d_%d.avi' % (n, depth), shell=True)
    subprocess.call('mplayer -loop 0 G_%d_d_%d.avi' % (n, depth), shell=True)

    return bdst.value
