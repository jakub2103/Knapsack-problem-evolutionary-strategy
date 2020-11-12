import numpy as np
import random

class ES(object):
     """
     Atribiutes:
     adam_and_eve - amount of first parents
     dims - number of analyzed dims, each dimmension represent one character
     scope - it's possible to limit range of each dimmension
     """

     def __init__(self, adam_and_eve, dims, scope=None):
          self.num_of_population = adam_and_eve
          self.dims = dims
          self.scope = scope
          if self.scope is None:
               self.scope = [100 for _ in range(self.dims)]
          elif len(self.scope) <= self.dims:
               [self.scope.append(100) for _ in range(len(self.scope), self.dims)]
          else:
               self.scope = self.scope[:self.dims]
          # generate random first population
          self.population = []
          for _ in range(adam_and_eve):
               self.population.append(np.multiply(np.random.rand(self.dims), self.scope))

     def generate_children(self, amount_of_offspring):
          # Select (randomly) ro parents from population mi - if ro == mi take all
          ro = np.random.randint(self.num_of_population)
          np.random.shuffle(self.population)
          selected_parent = self.population[:ro]
          # Recombine the ro selected parents a to form a recombinant individual r
          np.random.shuffle(selected_parent)
          first_part = np.array(selected_parent[:ro//2])
          second_part = np.array(selected_parent[ro//2:])
          if len(second_part) == len(first_part):
               paired = np.array(list(zip(first_part, second_part)))
          elif len(second_part) > len(first_part):
               paired = np.array(list(zip(first_part, second_part[:-1])))
          else:
               paired = np.array(list((zip(first_part[:-1], second_part))))

temp = ES(16, 2, scope=[14, 32])
temp.generate_children(10)
