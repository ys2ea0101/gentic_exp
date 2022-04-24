import matplotlib.pyplot as plt
import pandas as pd


df1 = pd.read_csv("data_archive/mc_400_400.dat")
df2 = pd.read_csv("data_archive/mc_200_200_case3.dat")
df3 = pd.read_csv("data_archive/swap_400_400.dat")
df4 = pd.read_csv("data_archive/both_100_100.dat")
df5 = pd.read_csv("data_archive/swap_200_200_case3.dat")

plt.plot(df1.values[:,1], label="Loop mutation", linewidth=2)
plt.plot(df3.values[:,1], label="Exchange mutation", linewidth=2)

#opt_val = -4294.25
#optimal = [opt_val for _ in range(len(df2))]
#plt.plot(optimal, label="Hungarian solution", linestyle='dashed', linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Best objective")
plt.legend()
plt.show()
