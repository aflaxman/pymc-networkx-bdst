""" A question on cstheory stackexchange [1] got me curious about MCMC for spanning trees

Here is a model to explore the issue

[1] http://cstheory.stackexchange.com/questions/3913/how-can-i-randomly-generate-bounded-height-spanning-trees
"""

import pylab as pl
import pymc as mc
import networkx as nx
import random

def BDST(n=25, k=5, beta=1.):
    """ Create a PyMC Stochastic for a Bounded Depth Spanning Tree
    Parameters
    ----------
    n : int, size of graph
    k : int, depth bound parameter
    beta : float, "inverse-temperature parameter" for depth bound
    """

    G = nx.Graph()
    G.add_path(range(n))  # start with a path for now
    G.pos = nx.spring_layout(G)

    @mc.stoch(dtype=nx.Graph)
    def bdst(value=G, k=k, beta=beta):
        path_len = pl.array(nx.shortest_path_length(value, 0).values())
        return -beta * pl.sum(path_len > k)

    return bdst

class BDSTMetropolis(mc.Metropolis):
    def __init__(self, stochastic):
        # Initialize superclass
        mc.Metropolis.__init__(self, stochastic, scale=1., proposal_sd='custom', proposal_distribution='custom', verbose=None, tally=False)

    def propose(self):
        """ Add an edge and remove an edge from the cycle that it creates"""
        G = self.stochastic.value

        self.u_new, self.v_new = random.sample(G.nodes(), 2)

        path = nx.shortest_path(G, self.u_new, self.v_new)
        i = random.randrange(len(path)-1)
        self.u_old, self.v_old = path[i], path[i+1]
        
        G.remove_edge(self.u_old, self.v_old)
        G.add_edge(self.u_new, self.v_new)
        self.stochastic.value = G

    def reject(self):
        """ Restore the graph to its state before more recent edge swap"""
        G = self.stochastic.value
        G.add_edge(self.u_old, self.v_old)
        G.remove_edge(self.u_new, self.v_new)
        self.stochastic.value = G
        
def anneal(n=25, k=5):
    beta = mc.Uninformative('beta', value=1.)
    bdst = BDST(n=n, k=k, beta=beta)

    mod_mc = mc.MCMC([beta, bdst])
    mod_mc.use_step_method(BDSTMetropolis, bdst)
    mod_mc.use_step_method(mc.NoStepper, beta)


    ni = 5
    nj = 10
    nk = 10

    for i in range(1, ni):
        beta.value = i
        for j in range(nj):
            mod_mc.sample(1)
            G = bdst.value
            G.pos = nx.spring_layout(G, pos=G.pos, fixed=[0], iterations=1)
            
            for k in range(nk):
                pl.clf()
                nx.draw_networkx_edges(G, G.pos, alpha=.5, width=2)
                X = pl.array(G.pos.values())
                pl.plot(X[:,0], X[:,1], 'o', alpha=.5)
                pl.axis([-3, 3, -3, 3])
                pl.axis('off')
                pl.subplots_adjust(0, 0, 1, 1)
                pl.savefig('t%3d.png' % (i*nj*nk + j*nk + k))
        print 'accepted:', mod_mc.step_method_dict[bdst][0].accepted
        print 'cur depth:', max(nx.shortest_path_length(bdst.value, 0).values())

    return bdst.value
