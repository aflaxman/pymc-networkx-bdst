""" Script to generate movie of MCMC search for a bounded-depth spanning tree
"""

import matplotlib.pylot as plt
import pymc as mc
import networkx as nx
import random
import views

from models import *

def anneal_w_graphics(n=11, depth=10):
    """ Make an animation of the BDST chain walking on an nxn grid and play it
    """
    ni = 5
    nj = 100
    nk = 5

    beta = mc.Uninformative('beta', value=1.)

    G = nx.grid_graph([n, n])
    G.orig_pos = dict([[v, v] for v in G.nodes_iter()])
    G.pos = dict([[v, v] for v in G.nodes_iter()])

    root = (5,5)
    bdst = BDST(G, root, depth, beta)

    mod_mc = mc.MCMC([beta, bdst])
    mod_mc.use_step_method(STMetropolis, bdst)
    mod_mc.use_step_method(mc.NoStepper, beta)

    for i in range(ni):
        beta.value = i*5
        for j in range(nj):
            mod_mc.sample(1)
            T = bdst.value
            
            for k in range(nk):
                if random.random() < .95:
                    delta_pos = nx.spring_layout(T, pos=G.pos, fixed=[root], iterations=1)
                else:
                    delta_pos = G.orig_pos
                eps=.01
                my_avg = lambda x, y: (x[0]*(1.-eps) + y[0]*eps, x[1]*(1.-eps)+y[1]*eps)
                for v in G.pos:
                    G.pos[v] = my_avg(G.pos[v], delta_pos[v])
                views.plot_graph_and_tree(G, T, time=1.*k/nk)
                str = ''
                str += ' beta: %.1f\n' % beta.value
                str += ' cur depth: %d (target: %d)\n' % (T.depth, depth)
                sm = mod_mc.step_method_dict[bdst][0]
                str += ' accepted: %d of %d\n' % (sm.accepted, sm.accepted + sm.rejected)
                plt.figtext(0, 0, str)
                plt.figtext(1, 0, 'healthyalgorithms.wordpress.com \n', ha='right')
                plt.axis([-1, n, -1, n])
                plt.axis('off')
                plt.subplots_adjust(0, 0, 1, 1)
                plt.savefig('bdst%06d.png' % (i*nj*nk + j*nk + k))
            print 'accepted:', mod_mc.step_method_dict[bdst][0].accepted

    import subprocess
    subprocess.call('mencoder mf://bdst*.png -mf w=800:h=600 -ovc x264 -of avi -o bdst_G_%d_d_%d.avi' % (n, depth), shell=True)
    subprocess.call('mplayer -loop 0 bdst_G_%d_d_%d.avi' % (n, depth), shell=True)
    subprocess.call('rm bdst*.png')

    return bdst.value

if __name__ == '__main__':
    anneal_w_graphics()
