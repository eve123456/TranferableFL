import numpy as np
import matplotlib.pyplot as plt

random_finetune_path = "performance/reg_J/random-finetune/ft100_ftBS_64/"

seed = [0,100,200,25,52]
res_path = [random_finetune_path +"seed"+ str(i) + ".npy" for i in seed]
r = []
for i in res_path:
    r.append(np.load(i).tolist())

FL_prms = "Fr200_Fe100_fe100_ftBS64"
regJ_path = "performance/reg_J/Fr200_Fe10_fe100_ftBS64/"
    
    
# plot for regJ_0
regJ_0_seed0 = [0.3874,0.4083,0.4667,0.4390,0.4762,0.3658,0.4307,0.4888,0.4969,0.4545,0.4694,0.5058,0.4945,0.4577,0.4876,0.5014,0.4857,0.4627,0.4694,0.4788,0.4835,0.3942,0.4672,0.4807,0.4776,0.4755,0.4589,0.4878,0.4297,0.4765,0.5202,0.5398,0.4817,0.4733,0.4193,0.4566,0.4629,0.4505,0.4872,0.4998,0.5252,0.4522,0.4748,0.5031,0.4782,0.4871,0.5085,0.4638,0.4646,0.4973,0.5003,0.4584,0.4823,0.4629,0.4653,0.4842,0.4791,0.5294,0.5189,0.5042,0.3467,0.5017,0.4847,0.4955,0.4652,0.5057,0.4779,0.4982,0.5109,0.4557,0.4772,0.5038,0.5143,0.4861,0.4767,0.4623,0.5004,0.4793,0.5165,0.5097,0.4282,0.4347,0.5039,0.4828,0.4849,0.5145,0.4802,0.4232,0.4958,0.4949,0.4805,0.4751,0.4571,0.4944,0.5158,0.5093,0.4190,0.4728,0.4792,0.4511]
r_regJ_0 = [regJ_0_seed0]
m_regJ_0 = np.mean(r_regJ_0,axis = 0)
s_regJ_0 = np.std(r_regJ_0,axis = 0)
plt.plot(range(len(m_regJ_0)), m_regJ_0, label = "regJ_0", color = "blue")
plt.fill_between(range(len(m_regJ_0)), m_regJ_0-s_regJ_0, m_regJ_0+s_regJ_0, alpha=0.2, color = "blue")



# # plot for regJ_e-2
regJ_2_seed0 = [0.4579,0.4843,0.4665,0.4712,0.4798,0.4232,0.4435,0.4765,0.4271,0.4693,0.4447,0.4987,0.4394,0.4585,0.4592,0.4689,0.4423,0.4562,0.4386,0.4172,0.4743,0.4574,0.4783,0.4603,0.4641,0.4762,0.4726,0.4157,0.4754,0.4729,0.4652,0.4726,0.4222,0.4811,0.4805,0.4848,0.4942,0.4527,0.4784,0.4528,0.4679,0.4687,0.4666,0.4996,0.4616,0.4310,0.4115,0.4759,0.4896,0.4474,0.4656,0.4789,0.4593,0.4145,0.4636,0.4643,0.4678,0.4514,0.4815,0.4430,0.4207,0.4534,0.4632,0.4889,0.4732,0.4569,0.4490,0.4701,0.4729,0.4611,0.4351,0.4578,0.4637,0.4517,0.4722,0.4779,0.4376,0.4270,0.4707,0.4205,0.4667,0.4558,0.4838,0.4794,0.4523,0.4447,0.4603,0.4864,0.4113,0.4598,0.4593,0.4201,0.4674,0.4088,0.4372,0.4248,0.4301,0.4338,0.4431,0.4493]
r_regJ_2 = [np.load(regJ_path + "regJ_e-2/seed0.npy").tolist()]
m_regJ_2 = np.mean(r_regJ_2,axis = 0)
s_regJ_2 = np.std(r_regJ_2,axis = 0)
plt.plot(range(len(m_regJ_2)), m_regJ_2, label = "regJ_2", color = "orange")
plt.fill_between(range(len(m_regJ_2)), m_regJ_2-s_regJ_2, m_regJ_2+s_regJ_2, alpha=0.2, color = "orange")

# plot for regJ_e-4
regJ_4_seed0 = [0.2823,0.3464,0.3923,0.3897,0.3612,0.4010,0.4008,0.3991,0.4323,0.4181,0.3983,0.3666,0.4054,0.3855,0.4226,0.4545,0.3956,0.4828,0.3552,0.3833,0.3515,0.3831,0.4613,0.3625,0.4102,0.4436,0.4536,0.3845,0.3990,0.4957,0.4467,0.4334,0.4223,0.4114,0.3223,0.4886,0.4344,0.4722,0.4553,0.5196,0.3982,0.4084,0.4175,0.4844,0.4993,0.4050,0.4503,0.3763,0.3431,0.4662,0.4351,0.3844,0.3893,0.4832,0.3305,0.4196,0.4108,0.4699,0.4376,0.4306,0.4351,0.4177,0.4445,0.5013,0.4783,0.4332,0.4623,0.3961,0.4641,0.4689,0.4101,0.3120,0.4313,0.4977,0.4935,0.4959,0.4007,0.4313,0.4605,0.3533,0.3945,0.4765,0.4435,0.4385,0.4677,0.4735,0.3527,0.4697,0.4440,0.3652,0.4142,0.3479,0.4511,0.4140,0.4340,0.4288,0.4895,0.4138,0.4595,0.4534]
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


