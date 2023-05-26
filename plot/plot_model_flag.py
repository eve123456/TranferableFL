import numpy as np
import matplotlib.pyplot as plt
import time


r= np.array([[0.3491,0.3504,0.3586,0.3647,0.3716,0.3645,0.3836,0.3762,0.3842,0.3666,0.3826,0.3755,0.3717,0.3568,0.3632,0.3567,0.3821,0.3624,0.3743,0.3706,0.3648,0.3774,0.3678,0.3822,0.3793,0.3816,0.3621,0.3527,0.3541,0.3650,0.3698,0.3500,0.3555,0.3750,0.3774,0.3751,0.3787,0.3828,0.3570,0.3464,0.3695,0.3591,0.3686,0.3564,0.3677,0.3633,0.3718,0.3662,0.3416,0.3556,0.3765,0.3724,0.3520,0.3641,0.3602,0.3698,0.3733,0.3514,0.3691,0.3610],\
            [0.3605,0.3718,0.3708,0.3492,0.3677,0.3723,0.3744,0.3653,0.3678,0.3605,0.3842,0.3790,0.3654,0.3726,0.3800,0.3574,0.3560,0.3540,0.3677,0.3668,0.3554,0.3668,0.3678,0.3642,0.3642,0.3611,0.3636,0.3721,0.3624,0.3700,0.3642,0.3726,0.3595,0.3767,0.3825,0.3792,0.3461,0.3413,0.3797,0.3783,0.3453,0.3475,0.3634,0.3757,0.3453,0.3741,0.3715,0.3701,0.3648,0.3656,0.3666,0.3648,0.3626,0.3660,0.3844,0.3847,0.3651,0.3725,0.3698,0.3685],\
            [0.3718,0.3807,0.3772,0.3731,0.3616,0.3473,0.3753,0.3587,0.3680,0.3674,0.3745,0.3525,0.3756,0.3575,0.3714,0.3585,0.3762,0.3643,0.3733,0.3617,0.3486,0.3682,0.3637,0.3540,0.3818,0.3774,0.3455,0.3687,0.3721,0.3698,0.3590,0.3695,0.3655,0.3501,0.3492,0.3758,0.3693,0.3746,0.3503,0.3692,0.3512,0.3701,0.3842,0.3811,0.3713,0.3596,0.3683,0.3665,0.3667,0.3807,0.3563,0.3482,0.3668,0.3637,0.3546,0.3560,0.3731,0.3813,0.3768,0.3662],\
            [0.3568,0.3792,0.3571,0.3680,0.3598,0.3838,0.3676,0.3606,0.3673,0.3604,0.3796,0.3642,0.3712,0.3814,0.3678,0.3687,0.3844,0.3622,0.3598,0.3695,0.3584,0.3642,0.3665,0.3666,0.3757,0.3821,0.3620,0.3543,0.3651,0.3682,0.3736,0.3605,0.3593,0.3734,0.3652,0.3757,0.3622,0.3767,0.3685,0.3724,0.3702,0.3774,0.3603,0.3667,0.3671,0.3623,0.3495,0.3558,0.3654,0.3710,0.3735,0.3618,0.3723,0.3677,0.3740,0.3642,0.3715,0.3722,0.3651,0.3597],\
            [0.3486,0.3783,0.3586,0.3725,0.3634,0.3558,0.3720,0.3770,0.3694,0.3768,0.3833,0.3611,0.3648,0.3584,0.3776,0.3533,0.3853,0.3712,0.3525,0.3597,0.3754,0.3700,0.3284,0.3634,0.3734,0.3750,0.3698,0.3685,0.3603,0.3697,0.3766,0.3797,0.3846,0.3645,0.3716,0.3684,0.3505,0.3627,0.3586,0.3733,0.3487,0.3748,0.3684,0.3695,0.3763,0.3627,0.3431,0.3816,0.3806,0.3768,0.3521,0.3618,0.3677,0.3766,0.3780,0.3698,0.3751,0.3574,0.3657,0.3583]])



m = np.mean(r,axis = 0)
s = np.std(r,axis = 0)


plt.plot(range(len(m)), m)
plt.fill_between(range(len(m)), m-s, m+s, alpha=0.2, color = "green" )

plt.xlabel("epoch")
plt.ylabel("acc")
plt.ylim(0,1)
plt.title(f"MNIST-M_model_flag: mean +- std by running {len(r)} tests.")


plt.savefig("./model_flag.jpg")



