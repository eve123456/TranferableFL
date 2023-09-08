import numpy as np
import matplotlib.pyplot as plt

random_finetune_path = "performance/reg_J/random-finetune/ft20_ftBS_1024/"
FL_prms = "Fr200_Fe100_fe20_ftBS1024"

seed = [0,100,200,25,52]
res_path = [random_finetune_path +"seed"+ str(i) + ".npy" for i in seed]
r = []
for i in res_path:
    r.append(np.load(i).tolist())
    
    
    
    
# plot for regJ_0
r_regJ_0 = [np.load("performance/reg_J/Fr200_Fe100_fe20_ftBS1024/regJ_0/seed0.npy").tolist()]
m_regJ_0 = np.mean(r_regJ_0,axis = 0)
s_regJ_0 = np.std(r_regJ_0,axis = 0)
plt.plot(range(len(m_regJ_0)), m_regJ_0, label = "regJ_0", color = "blue")
plt.fill_between(range(len(m_regJ_0)), m_regJ_0-s_regJ_0, m_regJ_0+s_regJ_0, alpha=0.2, color = "blue")


# plot for regJ_e-4
r_regJ_4 = [np.load("performance/reg_J/Fr200_Fe100_fe20_ftBS1024/regJ_e-4/seed0.npy").tolist()]
m_regJ_4 = np.mean(r_regJ_4,axis = 0)
s_regJ_4 = np.std(r_regJ_4,axis = 0)
plt.plot(range(len(m_regJ_4)), m_regJ_4, label = "regJ_4", color = "red")
plt.fill_between(range(len(m_regJ_4)), m_regJ_4-s_regJ_4, m_regJ_4+s_regJ_4, alpha=0.2, color = "red")


# plot for baseline (random - finetune)
m = np.mean(r,axis = 0)
s = np.std(r,axis = 0)


plt.plot(range(len(m)), m, label = "random-finetune", color = "grey")
plt.fill_between(range(len(m)), m-s, m+s, alpha=0.2, color = "grey" )

plt.xlabel("epoch")
plt.ylabel("acc")
plt.ylim(0,1)
plt.title(f"MNIST_MNIST-M_LeNet,"+FL_prms)
plt.legend()

plt.savefig("plot/MNIST_MNIST-M_LeNet/reg_J/Fr200_Fe100_fe20_ftBS_1024/regJ.jpg")