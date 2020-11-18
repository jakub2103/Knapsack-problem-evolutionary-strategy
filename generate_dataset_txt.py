import numpy as np


def generate(name, seed=1, restricted_amount=None, restricted_weight=None, restricted_value=None, length=100):
     if restricted_amount is None:
          restricted_amount = [0, 100]
     if restricted_value is None:
          restricted_value = [0, 1000]
     if restricted_weight is None:
          restricted_weight = [0, 1000]
     name = name.split(".")[0]
     f = open(name+".txt", 'w')
     np.random.seed(seed)
     for _ in range(length):
          row = [np.random.randint(restricted_amount[0], restricted_amount[1]),
                    np.random.randint(restricted_weight[0], restricted_weight[1]),
                    np.random.randint(restricted_value[0], restricted_value[1])]
          for i, r in enumerate(row):
               if i > 0:
                    r = r/10
               f.write(str(r) + "\t")
          f.write("\n")
     f.close()

generate("example_dataset", length=10)