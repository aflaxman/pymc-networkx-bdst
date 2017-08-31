""" Tests for the spanning trees and mazes"""

# matplotlib will open windows during testing unless you do the following
import matplotlib
matplotlib.use("AGG") 


import models as model
import views as graphics
import maze

class TestClass:
   def setup(self):
       self.G = model.my_grid_graph([5, 5])

   def test_bdst(self):
       bdst = model.BDST(self.G, beta=0.)
       assert bdst.logp == 0., 'at infinite temperature, all bounded-depth spanning tree should be equally likely'

   def test_ldst(self):
       ldst = model.LDST(self.G, beta=0.)
       assert ldst.logp == 0., 'at infinite temperature, low-degree spanning tree should be equally likely'
       
   def test_ldst_anneal(self):
       model.anneal_ldst(n=5, phases=2, iters=10)

   def test_bdst_anneal(self):
       model.anneal_bdst(n=5, phases=2, iters=10)

   def test_graph_utils(self):
       P = model.my_path_graph(model.nx.shortest_path(self.G, (0,0), (4,4)))
       H = model.image_grid_graph('test.png')

       d = model.dual_grid_edge((0,0), (0,1))
       assert d == ((-0.5, 0.5), (0.5, 0.5)), 'dual of integer lattice should be offset by .5s'

       D = model.dual_grid(H.base_graph, H)
       graphics.add_maze_boundary(D, [5,5])
       graphics.make_entry_and_exit(D, [5,5])
       HH = graphics.split_edges(H)

   def test_maze_graphics(self):
       T = model.anneal_ldst(n=5, phases=1, iters=1)
       D = model.dual_grid(self.G, T)
       D_pos = graphics.layout_maze(D)
       graphics.plot_maze(D, D_pos, T, self.G.pos)
 
       T.root = (0,0)
       T.k = 5

       graphics.plot_graph_and_tree(self.G, T, 0)

   def test_random_maze(self):
      maze.random_maze(5)

   def test_hidden_image_maze(self):
      maze.hidden_image_maze('test.png', style='jittery')
      maze.hidden_image_maze('test.png', style='smooth')
      maze.hidden_image_maze('test.png', style='sketch')

   def test_ld_maze(self):
      maze.ld_maze(n=5)

   def test_border_maze(self):
      maze.border_maze('test.png')
