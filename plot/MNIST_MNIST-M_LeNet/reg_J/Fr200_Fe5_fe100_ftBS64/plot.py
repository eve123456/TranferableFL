

import numpy as np
import matplotlib.pyplot as plt

random_finetune_path = "performance/reg_J/random-finetune/ft100_ftBS_64/"

seed = [0,100,200,25,52]
res_path = [random_finetune_path +"seed"+ str(i) + ".npy" for i in seed]
r = []
for i in res_path:
    r.append(np.load(i).tolist())

FL_prms = "Fr200_Fe5_fe100_ftBS64"
regJ_path = "performance/reg_J/Fr200_Fe10_fe100_ftBS64/"
    
    
# plot for regJ_0
regJ_0_seed0 = np.array([0.3387,0.3376,0.3252,0.3685,0.3643,0.2993,0.3773,0.3900,0.4114,0.3203,0.3562,0.4027,0.3557,0.3824,0.3840,0.3990,0.3666,0.3981,0.3901,0.4107,0.3876,0.3546,0.3813,0.3936,0.3788,0.3573,0.4045,0.3910,0.3878,0.4050,0.4182,0.3066,0.3851,0.3783,0.3732,0.3443,0.3874,0.3512,0.4040,0.3787,0.3898,0.4313,0.4082,0.4041,0.3761,0.4042,0.3813,0.3738,0.3953,0.3562,0.3981,0.3921,0.3860,0.3967,0.3887,0.3374,0.3293,0.3912,0.3674,0.4172,0.3678,0.3692,0.3446,0.3856,0.4037,0.4070,0.3744,0.4044,0.4374,0.3657,0.4220,0.3942,0.3867,0.3652,0.3931,0.4042,0.4215,0.3933,0.3372,0.4476,0.3613,0.3673,0.3751,0.3320,0.3958,0.3477,0.3792,0.3644,0.3904,0.3840,0.3905,0.4064,0.4186,0.3753,0.4234,0.3780,0.4104,0.3455,0.4211,0.3844])

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

# # plot for regJ_e-4
# r_regJ_4 = [np.load(regJ_path + "/regJ_e-4/seed0.npy").tolist()]
# m_regJ_4 = np.mean(r_regJ_4,axis = 0)
# s_regJ_4 = np.std(r_regJ_4,axis = 0)
# plt.plot(range(len(m_regJ_4)), m_regJ_4, label = "regJ_4", color = "red")
# plt.fill_between(range(len(m_regJ_4)), m_regJ_4-s_regJ_4, m_regJ_4+s_regJ_4, alpha=0.2, color = "red")


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