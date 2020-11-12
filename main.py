import numpy as np
from operator import itemgetter
import copy


class Specimen(object):
     def __init__(self, dims, scope):

          self.x = copy.deepcopy(scope)
          self.dims = dims
          if scope is None:             self.x = [10 for i in range(dims)]
          elif len(self.x) <= dims:     [self.x.append(10) for _ in range(dims - len(self.x))]
          else:                         self.x = self.x[:dims]
          for i in range(len(self.x)):
               if self.x[i] > 0:
                    self.x[i] = np.random.randint(0, self.x[i])
          self.s = np.random.random(len(self.x))

     def mutate(self):
          s_gaussian = np.random.normal(loc=0.0, scale=1., size=(self.dims, ))
          self.s = np.multiply(self.s, np.exp(s_gaussian))
          x_gaussian = np.random.normal(loc=0.0, scale=self.s, size=(self.dims,))
          self.x = np.add(self.x, x_gaussian)

     def crossing(self, x, s, cross_chance=0.5):
          """
          cross_chance - represents probability for corossing each feature
          """
          for i in range(len(x)):
               if np.random.rand() < cross_chance:
                    self.x[i], x[i] = x[i], self.x[i]
                    self.s[i], s[i] = s[i], self.s[i]
          return x, s

     def optimize_function(self, max_weight):
          result = 0
          element_weight = 10
          for x in self.x:
               result += x*element_weight
               element_weight += 10
          if result > max_weight:
               return 0
          return result


class ES(object):
     """
     We choose (mi/ro, lambda) strategy
     For now each dimmension represent next 10 kg and value is amount of this element
     e.g. x = [2, 5, 1] means that we want to pack 2*10 + 5 * (10+10) + 1 *(10+10+10) kg
     Atribiutes:
     adam_and_eve - amount of first parents
     dims - number of analyzed dims, each dimmension represent one character
     scope - it's possible to limit range of each dimmension
     """

     def __init__(self, adam_and_eve, dims, scope=None, max_weight=500):
          self.num_of_population = adam_and_eve
          self.dims = dims
          self.scope = scope
          # generate random first population
          self.population = []
          for _ in range(adam_and_eve):
               self.population.append(Specimen(dims, scope))
          self.max_weight = max_weight

     def generate_children(self):
          # Select (randomly) ro parents from population mi - if ro == mi take all
          ro = np.random.randint(self.num_of_population)    # random amount of offspring
          np.random.shuffle(self.population)
          selected_parent = self.population[:ro]
          # Recombine the ro selected parents a to form a recombinant individual r
          np.random.shuffle(selected_parent)
          # cross only 25% of selected parent
          temp_list_of_parents = selected_parent[:len(selected_parent)//4]
          new_population = selected_parent[len(selected_parent)//4:]
          while len(temp_list_of_parents) > 0:
               if len(temp_list_of_parents) > 1:
                    parent_1 = temp_list_of_parents.pop(np.random.randint(len(temp_list_of_parents)))
                    parent_2 = temp_list_of_parents.pop(np.random.randint(len(temp_list_of_parents)))
                    parent_2.x, parent_2.s = parent_1.crossing(parent_2.x, parent_2.s)
                    new_population.append(parent_1)
                    new_population.append(parent_2)
               else:
                    new_population.append(temp_list_of_parents.pop(np.random.randint(len(temp_list_of_parents))))
          # mutate all new population
          for new_pop in range(len(new_population)):
               new_population[new_pop].mutate()
          self.population = np.append(self.population, new_population)
          results = []
          for pop in self.population:
               results.append(pop.optimize_function(self.max_weight))
          results = np.array(results)
          best_of_index = results.argsort()[-self.num_of_population:][::-1]
          f = itemgetter(best_of_index)
          self.population = list(f(self.population))


temp = ES(500, 8, scope=[50, 42])
temp.generate_children()

