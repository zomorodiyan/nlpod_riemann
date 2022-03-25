import matplotlib.pyplot as plt
import numpy as np
import os

N = 51
# read_write x,y
x = np.genfromtxt("Results/csv/bc_1/X.csv", delimiter=",")
np.save("Results/npz/X.npy", x)
y = np.genfromtxt("Results/csv/bc_1/Y.csv", delimiter=",")
np.save("Results/npz/Y.npy", y)

# write field(s)
for varName in ["P"]:
    for bc in range(1, 6):
        if not os.path.exists("./Results/npz/bc_" + str(bc)):
            os.makedirs("./Results/npz/bc_" + str(bc))
        for ss in range(N):
            # ---- read ------------------------------------------------------
            dir_ = "Results/csv/bc_" + str(bc) + "/" + varName + str(ss) + ".csv"
            z = np.genfromtxt(dir_, delimiter=",")

            # ---- write -----------------------------------------------------
            dir_ = "Results/npz/bc_" + str(bc) + "/" + varName + str(ss) + ".npy"
            np.save(dir_, z)

        print("bc_" + str(bc) + "is done")
