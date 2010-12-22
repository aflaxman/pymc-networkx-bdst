""" A question on cstheory stackexchange [1] got me curious about MCMC for spanning trees

Here is a model to explore the issue

[1] http://cstheory.stackexchange.com/questions/3913/how-can-i-randomly-generate-bounded-height-spanning-trees
"""

import pylab as pl
import pymc as mc
import networkx as nx
import random

def BDST(G, k=5, beta=1.):
    """ Create a PyMC Stochastic for a Bounded Depth Spanning Tree on
    base graph G

    Parameters
    ----------
    G : nx.Graph, base graph to span
    k : int, depth bound parameter
    beta : float, "inverse-temperature parameter" for depth bound
    """

    T = nx.minimum_spanning_tree(G)

    @mc.stoch(dtype=nx.Graph)
    def bdst(value=T, G=G, k=k, beta=beta):
        path_len = pl.array(nx.shortest_path_length(value, 0).values())
        return -beta * pl.sum(path_len > k)

    return bdst

class BDSTMetropolis(mc.Metropolis):
    def __init__(self, stochastic):
        # Initialize superclass
        mc.Metropolis.__init__(self, stochastic, scale=1., proposal_sd='custom', proposal_distribution='custom', verbose=None, tally=False)

    def propose(self):
        """ Add an edge and remove an edge from the cycle that it creates"""
        T = self.stochastic.value
        G = self.stochastic.parents['G']

        T.u_new, T.v_new = random.sample(T.edges(), 1)[0]

        path = nx.shortest_path(T, T.u_new, T.v_new)
        i = random.randrange(len(path)-1)
        T.u_old, T.v_old = path[i], path[i+1]
        
        T.remove_edge(T.u_old, T.v_old)
        T.add_edge(T.u_new, T.v_new)
        self.stochastic.value = T

    def reject(self):
        """ Restore the graph to its state before more recent edge swap"""
        T = self.stochastic.value
        T.add_edge(T.u_old, T.v_old)
        T.remove_edge(T.u_new, T.v_new)
        self.stochastic.value = T
        
def anneal(n=25, k=5):
    beta = mc.Uninformative('beta', value=1.)

    G = nx.complete_graph(n)
    G.pos = nx.spring_layout(G, fixed=[0], iterations=1)
    bdst = BDST(G, k, beta)

    mod_mc = mc.MCMC([beta, bdst])
    mod_mc.use_step_method(BDSTMetropolis, bdst)
    mod_mc.use_step_method(mc.NoStepper, beta)


    ni = 5
    nj = 100
    nk = 3

    for i in range(1, ni):
        beta.value = i*5
        for j in range(nj):
            mod_mc.sample(1)
            T = bdst.value
            
            for k in range(nk):
                if k < .9*nk:
                    G.pos = nx.spring_layout(T, pos=G.pos, fixed=[0], iterations=1)

                pl.clf()
                nx.draw_networkx_edges(T, G.pos, alpha=.5, width=2)
                X = pl.array(G.pos.values())
                pl.plot(X[:,0], X[:,1], 'o', alpha=.5)

                pl.plot(X[[T.u_old, T.v_old], 0], X[[T.u_old, T.v_old], 1], 'r-', linewidth=4, alpha=(nk - k) * .5 / nk)
                pl.plot(X[[T.u_new, T.v_new], 0], X[[T.u_new, T.v_new], 1], 'b-', linewidth=4, alpha=.25 + k*.5/nk)
                
                pl.figtext(0, 0, 'cur depth: %d\naccepted: %d\n' % (max(nx.shortest_path_length(bdst.value, 0).values()), mod_mc.step_method_dict[bdst][0].accepted))
                pl.axis([-3, 3, -3, 3])
                pl.axis('off')
                pl.subplots_adjust(0, 0, 1, 1)
                pl.savefig('t%06d.png' % (i*nj*nk + j*nk + k))
            print 'accepted:', mod_mc.step_method_dict[bdst][0].accepted

    return bdst.value
