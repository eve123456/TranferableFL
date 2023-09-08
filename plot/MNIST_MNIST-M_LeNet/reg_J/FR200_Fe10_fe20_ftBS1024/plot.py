import numpy as np
import matplotlib.pyplot as plt

random_finetune_path = "performance/reg_J/random-finetune/ft20_ftBS_1024/"

seed = [0,100,200,25,52]
res_path = [random_finetune_path +"seed"+ str(i) + ".npy" for i in seed]
r = []
for i in res_path:
    r.append(np.load(i).tolist())

FL_prms = "FR200_Fe10_fe20_ftBS1024"
regJ_path = "performance/reg_J/Fr200_Fe10_fe100_ftBS64/"
    
    
# plot for regJ_0
regJ_0_seed0 = [0.3065,0.3510,0.3641,0.3571,0.3463,0.3310,0.3708,0.3514,0.3608,0.3828,0.3548,0.3613,0.3768,0.3837,0.3748,0.3852,0.3926,0.3996,0.4051,0.3822]

r_regJ_0 = [regJ_0_seed0]
m_regJ_0 = np.mean(r_regJ_0,axis = 0)
s_regJ_0 = np.std(r_regJ_0,axis = 0)
plt.plot(range(len(m_regJ_0)), m_regJ_0, label = "regJ_0", color = "blue")
plt.fill_between(range(len(m_regJ_0)), m_regJ_0-s_regJ_0, m_regJ_0+s_regJ_0, alpha=0.2, color = "blue")



# # plot for regJ_e-2
# r_regJ_2 = [np.load(regJ_path + "regJ_e-2/seed0.npy").tolist()]
# m_regJ_2 = np.mean(r_regJ_2,axis = 0)
# s_regJ_2 = np.std(r_regJ_2,axis = 0)
# plt.plot(range(len(m_regJ_2)), m_regJ_2, label = "regJ_2", color = "orange")
# plt.fill_between(range(len(m_regJ_2)), m_regJ_2-s_regJ_2, m_regJ_2+s_regJ_2, alpha=0.2, color = "orange")

# plot for regJ_e-4
regJ_4_seed0 = [0.3908,0.3860,0.4253,0.4358,0.4516,0.4536,0.4190,0.4468,0.4539,0.4354,0.4614,0.4263,0.4382,0.4689,0.4743,0.4506,0.4537,0.4643,0.4736,0.4662]

r_regJ_4 = [regJ_4_seed0]
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





