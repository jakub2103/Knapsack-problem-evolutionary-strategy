import numpy as np


def generate(name, seed=1, restricted=None, length=100):
     if restricted is None:
          restricted = [0, 10000]
     name = name.split(".")[0]
     f = open(name+".txt", 'w')
     np.random.seed(seed)
     for _ in range(length):
          row = np.random.randint(restricted[0], restricted[1], 3)
          for i, r in enumerate(row):
               if i > 0:
                    r = r/100
               f.write(str(r) + "\t")
          f.write("\n")
     f.close()



generate("example_dataset", length=200)