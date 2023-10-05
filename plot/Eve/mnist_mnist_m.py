from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
# rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


FL_rounds = np.array([10, 20, 30, 40, 50, 100])

rl_acc_FedAvg = np.array([0.24, -0.046, -0.144, -0.157, -0.15, -0.2])
rl_acc_coef_0001 = np.array([0.414, -0.031, -0.011, -0.128, -0.113, -0.222])
rl_acc_coef_001 = np.array([0.16,0.265,0.243,-0.132,-0.142,-0.204])

DT_acc_coef_0001 = [0.174, 0.015, 0.133, 0.029, 0.037, -0.022]
DT_acc_coef_001 = [-0.08, 0.311, 0.387, 0.025, 0.008, -0.004]

# plt.scatter(FL_rounds, DT_acc_coef_0001, label = "1e-3")
# plt.scatter(FL_rounds, DT_acc_coef_001, label = "1e-2")

plt.figure(figsize=(4,4))
plt.tight_layout()

ax = plt.axes()
ax.set_facecolor("lavender")
plt.grid(color='white', linestyle='-', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(bottom=False, top = False, left = False,right = False)
ax.set_axisbelow(True)

frame = plt.gca()



plt.scatter(FL_rounds[1:-1], rl_acc_coef_001[1:-1],s=50, c = "blue", label = "1e-2")
plt.scatter(FL_rounds[1:-1], rl_acc_coef_0001[1:-1],s = 35, c = "blue", label = "1e-3")
plt.scatter(FL_rounds[1:-1], rl_acc_FedAvg[1:-1], s=20, c = "red", label = "0 (FedAvg)")
plt.title("MNIST "+ r'$\rightarrow$'+  " MNIST-M")
plt.xlabel("Federate Pretrain Round")
plt.ylabel("Relative DT Acc (\%)")
ax.yaxis.set_label_coords(-0.1,0.05)

plt.xticks(np.arange(20, 51, step=10))

legend = plt.legend(title=r'$\xi_J$')
                    # fontsize='small', fancybox=True)







plt.savefig("./test.jpg")