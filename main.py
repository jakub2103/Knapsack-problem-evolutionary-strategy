import numpy as np
from operator import itemgetter
import copy
import matplotlib.pyplot as plt
import pandas as pd
from generate_dataset_txt import generate
import time
import scipy.stats as ss
import operator


class Specimen(object):
    """
     Single object representing single knapsack.
     """

    def __init__(self, scope):
        self.scope = copy.deepcopy(scope)
        self.dims = 3
        self.x = None
        self.s = None
        if self.scope is not None:  # scope == 3-dimmensional array [[1_amount, 1_weight, 1_value],[2_amout, ...],]
            self.__gen_specimen_values__(self.scope)
        self.sum_weight = 0
        self.sum_value = 0

    def __gen_specimen_values__(self, scope):
        self.scope = copy.deepcopy(scope)
        self.x = []
        for i in self.scope:
            self.x.append(np.random.randint(0, i[0] + 1))  # random x
        self.s = np.random.random(len(self.x))  # random s

    def __mutate__(self, discrete):
        s_gaussian = np.random.normal(loc=0.0, scale=1., size=(len(self.s),))
        self.s = np.multiply(self.s, np.exp(s_gaussian))
        # discrete gaussian distribution
        if discrete:
            x = np.arange(-len(self.x) // 2, -len(self.x) // 2 + 1)
            xU, xL = x + 0.5, x - 0.5
            prob = ss.norm.cdf(xU, scale=10) - ss.norm.cdf(xL, scale=10)
            prob = prob / prob.sum()
            x_gaussian = np.random.choice(x, size=(len(self.x),), p=prob)

        # cont gaussain distribution
        else:
            x_gaussian = np.random.normal(loc=0.0, scale=self.s, size=(len(self.x),))

        if np.all(np.add(self.x, x_gaussian) < self.scope[:, 0]) and \
                np.all(np.add(self.x, x_gaussian) >= 0):
            if np.any(np.add(self.x, x_gaussian) > self.scope[:, 0]) and np.any(np.add(self.x, x_gaussian) > 0):
                self.x = np.add(self.x, x_gaussian)

    def __update_weight__(self):
        self.sum_weight = copy.deepcopy(np.sum(np.multiply(self.x, self.scope[:, 1])))

    def __update_value__(self):
        self.sum_value = copy.deepcopy(np.sum(np.multiply(self.x, self.scope[:, 2])))

    def __crossing__(self, x, s, cross_chance=0.5):
        """
          cross_chance - represents probability for corossing each feature
          """
        for i in range(len(x)):
            if np.random.rand() < cross_chance:
                self.x[i], x[i] = x[i], self.x[i]
                self.s[i], s[i] = s[i], self.s[i]
        return x, s

    def __optimize_function__(self, max_weight):
        if self.sum_weight > max_weight:
            return 0
        return self.sum_value


class ES(object):
    """
     We choose (mi/ro, lambda) strategy
     For now each dimmension represent next 10 kg and value is amount of this element
     e.g. x = [2, 5, 1] means that we want to pack 2*10 + 5 * (10+10) + 1 *(10+10+10) kg
     input: three dimmensional array with elements representing - max amount, weight, value each must be from 0 to inf

     Atribiutes:
     adam_and_eve - amount of first parents
     dims - number of analyzed dims, each dimmension represent one character
     scope - it's possible to limit range of each dimmension
     """

    def __init__(self, adam_and_eve, scope=None, max_weight=90000000, error=5e-5, max_value=90000000):
        self.num_of_population = adam_and_eve
        self.dims = 3  # number of dimmensions [amount, weight, value]
        self.scope = scope  # represents the maximum quantity and weight
        self.error = error  # maximum allowable error
        # generate random first population
        self.population = []
        self.best_result = 0
        """Initial dataset represent example solution"""
        for _ in range(adam_and_eve):
            self.population.append(Specimen(scope))
        self.max_weight = max_weight  # maximum weight of knapsack
        self.max_value = max_value
        if scope is not None:
            self.__define_max_real_weight__()
            self.__define_max_real_velue__()

    def __define_max_real_weight__(self):
        if self.max_weight > np.sum(np.multiply(self.scope[:, 0], self.scope[:, 1])):
            self.max_weight = np.sum(np.multiply(self.scope[:, 0], self.scope[:, 1]))
            print("Max possible weight was changed into", self.max_weight)

    def __define_max_real_velue__(self):
        if self.max_value > np.sum(np.multiply(self.scope[:, 0], self.scope[:, 2])):
            self.max_value = np.sum(np.multiply(self.scope[:, 0], self.scope[:, 2]))
            print("Max possible value was changed into", self.max_value)

    def load_from_file(self, file_path):
        """
          :param file_path: path to file
          :param type: possible formats [txt, xlsx, csv]
          """
        _, type = file_path.split(".")
        if type == 'txt':
            self.scope = np.loadtxt(file_path)
        elif type == 'xlsx':
            self.scope = pd.read_excel(file_path).to_numpy()
        elif type == 'csv':
            self.scope = pd.read_excel(file_path).to_numpy()
        else:
            print("Unsupported format")
            return 0
        print("File loaded")
        for population in self.population:
            population.__gen_specimen_values__(self.scope)
        self.__define_max_real_weight__()
        self.__define_max_real_velue__()
        return 0

    def __generate_children__(self, discrete, more_crossing=False):
        # Select (randomly) ro parents from population mi - if ro == mi take all
        ro = np.random.randint(self.num_of_population)  # random amount of offspring
        np.random.shuffle(self.population)
        selected_parent = copy.deepcopy(self.population[:ro])
        # Recombine the ro selected parents a to form a recombinant individual r
        np.random.shuffle(selected_parent)
        # cross only 25% of selected parent
        if more_crossing:
            temp_list_of_parents = selected_parent[:len(selected_parent) // 3]
            new_population = selected_parent[len(selected_parent) // 3:]
        # if there is problem with max value try to cross more parent
        else:
            temp_list_of_parents = selected_parent[:len(selected_parent) // 4]
            new_population = selected_parent[len(selected_parent) // 4:]
        while len(temp_list_of_parents) > 0:
            if len(temp_list_of_parents) > 1:
                parent_1 = temp_list_of_parents.pop(np.random.randint(len(temp_list_of_parents)))
                parent_2 = temp_list_of_parents.pop(np.random.randint(len(temp_list_of_parents)))
                parent_2.x, parent_2.s = parent_1.__crossing__(parent_2.x, parent_2.s)
                new_population.append(parent_1)
                new_population.append(parent_2)
            else:
                new_population.append(temp_list_of_parents.pop(np.random.randint(len(temp_list_of_parents))))
        # mutate all new population
        for new_pop in range(len(new_population)):
            new_population[new_pop].__mutate__(discrete)
            new_population[new_pop].__update_value__()
            new_population[new_pop].__update_weight__()
        self.population = np.append(self.population, new_population)
        results = []
        for population in self.population:
            results.append(population.__optimize_function__(self.max_weight))
        results = np.array(results)
        best_of_index = results.argsort()[-self.num_of_population:][::-1]
        if self.best_result == results[best_of_index[0]]:
            more_crossing = True
        self.best_result = results[best_of_index[0]]
        f = itemgetter(best_of_index)
        self.population = list(f(self.population))
        return more_crossing

    def train(self, epochs=50, is_plot=True, discrete=True):
        fig, ax = plt.subplots()
        ep = 0
        more_crossing = False
        for epoch in range(epochs):
            start = time.time()
            print("Epoch: ", epoch)
            ep = epoch
            more_crossing = self.__generate_children__(discrete, more_crossing)
            print("Max value in epoch: ", self.best_result)
            if is_plot:
                self.__plot__(ax, epoch, pause=0.05)
            if self.best_result / self.max_value >= 1 - self.error:
                print("Time for epoch: ", time.time() - start, "s \n")
                print("Found optimal value")
                break
            print("Time for epoch: ", time.time() - start, "s \n")
        self.__plot__(ax, ep, pause=20.)

        # return best specimen
        return sorted(self.population, key=operator.attrgetter('sum_value'), reverse=True)[0]

    def __plot__(self, ax, epoch, pause=0.15):
        ax.cla()
        point_list = []
        ax.set_xlim(0, self.max_weight*1.1)
        ax.set_ylim(0, self.max_value*1.1)
        ax.set_title("Epoch: " + str(epoch))
        ax.set_ylabel('Total value')
        ax.set_xlabel('Total weight')
        for population in self.population:
            point_list.append([population.sum_weight, population.sum_value])
        point_list = np.array(point_list)
        color_list = point_list[:, 1] / np.max(point_list[:, 1])
        ax.scatter(point_list[:, 0], point_list[:, 1], c=color_list)
        plt.pause(pause)


if __name__ == '__main__':
    data = generate("bbal", dims=500, save_to_file=False, restricted_amount=[1, 1], restricted_weight=[0, 10])
    es = ES(1000, error=5e-7, scope=data)
    es.train(epochs=400, is_plot=True, discrete=True)
