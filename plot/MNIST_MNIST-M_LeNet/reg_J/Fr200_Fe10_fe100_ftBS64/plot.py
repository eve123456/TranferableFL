import numpy as np
import matplotlib.pyplot as plt

random_finetune_path = "performance/reg_J/random-finetune/ft100_ftBS_64/"

seed = [0,100,200,25,52]
res_path = [random_finetune_path +"seed"+ str(i) + ".npy" for i in seed]
r = []
for i in res_path:
    r.append(np.load(i).tolist())

FL_prms = "Fr200_Fe10_fe100_ftBS64"
regJ_path = "performance/reg_J/Fr200_Fe10_fe100_ftBS64/"
    
    
# plot for regJ_0
r_regJ_0 = [np.load(regJ_path + "regJ_0/seed0.npy").tolist()]
m_regJ_0 = np.mean(r_regJ_0,axis = 0)
s_regJ_0 = np.std(r_regJ_0,axis = 0)
plt.plot(range(len(m_regJ_0)), m_regJ_0, label = "regJ_0", color = "blue")
plt.fill_between(range(len(m_regJ_0)), m_regJ_0-s_regJ_0, m_regJ_0+s_regJ_0, alpha=0.2, color = "blue")

# plot for regJ_e-2
r_regJ_2 = [np.load(regJ_path + "regJ_e-2/seed0.npy").tolist()]
m_regJ_2 = np.mean(r_regJ_2,axis = 0)
s_regJ_2 = np.std(r_regJ_2,axis = 0)
plt.plot(range(len(m_regJ_2)), m_regJ_2, label = "regJ_2", color = "orange")
plt.fill_between(range(len(m_regJ_2)), m_regJ_2-s_regJ_2, m_regJ_2+s_regJ_2, alpha=0.2, color = "orange")

# plot for regJ_e-4
r_regJ_4 = [np.load(regJ_path + "/regJ_e-4/seed0.npy").tolist()]
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
plt.title(f"MNIST_MNIST-M_LeNet, "+FL_prms)
plt.legend()

plt.savefig("plot/MNIST_MNIST-M_LeNet/reg_J/"+FL_prms+"/regJ.jpg")