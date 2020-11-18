import numpy as np


def generate(name, seed=1, restricted_amount=None, restricted_weight=None, restricted_value=None, length=100,
             save_to_file=True):
     if restricted_amount is None:
          restricted_amount = [0, 100]
     if restricted_value is None:
          restricted_value = [0, 1000]
     if restricted_weight is None:
          restricted_weight = [0, 1000]
     name = name.split(".")[0]
     if save_to_file:
          f = open(name+".txt", 'w')
     np.random.seed(seed)
     matrix = []
     for _ in range(length):
          row = [np.random.randint(restricted_amount[0], restricted_amount[1]),
                    np.random.randint(restricted_weight[0], restricted_weight[1]),
                    np.random.randint(restricted_value[0], restricted_value[1])]
          matrix.append([row[0], row[1]/10, row[1]/10])
          for i, r in enumerate(row):
               if i > 0:
                    r = r/10
               if save_to_file:
                    f.write(str(r) + "\t")
          if save_to_file:
               f.write("\n")
     if save_to_file:
          f.close()
     return np.array(matrix)


generate("example_dataset", length=10)