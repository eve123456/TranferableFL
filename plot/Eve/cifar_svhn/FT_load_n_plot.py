import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

mtrx_nm = ["trn_loss", "trn_acc", "tst_loss", "tst_acc"]

mtrx_pos = {"trn_loss":0,
            "trn_acc":1,
            "tst_loss":2,
            "tst_acc":3}


def trunc_len(data):
    max_len = len(data[0])-1
    for i_rep in data:
        for j in range(len(i_rep)):
            if i_rep[j]==0:
                if j-1 < max_len:
                    max_len = j-1
                    break
    return max_len


def load_data(data_path, mtx_nm):
    f = open(data_path)
    data = json.load(f)
    data = list(data['model_ft_whole'].values())
    data = np.array(data)

    pos = mtrx_pos[mtx_nm]
    data = data[:,:,pos]

    max_len = trunc_len(data)
    data = data[:,:max_len+1]
    
    

    return data.mean(axis = 0), data.std(axis = 0)



def plot_contrast(coef_paths, mtx_nm, DT_flag, FL_bsline = None):

    _,ax = plt.subplots()
    ax.set_facecolor('lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_axisbelow(True)
    plt.grid(color='white', linestyle='-')

    for i_coef in coef_paths:
        i_path = coef_paths[i_coef]
        if not i_path:
            continue

        mtrx_mean, mtrx_std = load_data(i_path, mtx_nm)
        if DT_flag:
            mtrx_mean -= FL_bsline[mtx_nm][i_coef]
        l = len(mtrx_mean)
        # x_axis = np.linspace(1,len(mtrx_mean), len(mtrx_mean))
        x_smooth = np.linspace(1, l, l-5)  
        mtrx_mean_smooth =  np.array(mtrx_mean[:5].tolist() + [np.mean(mtrx_mean[i:i+5]) for i in range(5,l-5)])
        mtrx_std_smooth = np.array(mtrx_std[:5].tolist() +[np.mean(mtrx_std[i:i+5]) for i in range(5,l-5)])
    
        if i_coef == "0":
            plt.plot( x_smooth, mtrx_mean_smooth,"--", alpha=0.9, label = i_coef)
        else:
            plt.plot( x_smooth, mtrx_mean_smooth, alpha=0.9, label = i_coef)
        plt.fill_between(x_smooth, mtrx_mean_smooth - mtrx_std_smooth, mtrx_mean_smooth + mtrx_std_smooth, alpha=0.2)
    

        # plt.plot( x_axis, mtrx_mean, alpha=0.9, label = i_coef)
        # plt.fill_between(x_axis, mtrx_mean - mtrx_std, mtrx_mean + mtrx_std, alpha=0.2 )
        plt.xlabel("Finetune Epochs", fontsize=18)
        plt.ylabel(mtx_nm , fontsize=18)
        


    legend = plt.legend(title=r'$\xi_J$', fontsize = 20)
                    # fontsize='small', fancybox=True)
    plt.setp(legend.get_title(),fontsize=20)
    if DT_flag:
        plt.title("DT Finetune Performance on Target Domain", fontsize=18)
    else:
        plt.title("Finetune Performance on Target Domain", fontsize=18)
    print("check")
    plt.savefig(mtx_nm+".jpg")
    plt.cla()


if __name__ == "__main__":
    cof_pths = {"0": "./result/cifar10_all_data_equal_niid/20231116150737.json",
                "1e-4": "./result/cifar10_all_data_equal_niid/20231116150316.json",
                "1e-3": "./result/cifar10_all_data_equal_niid/20231116151058.json"} 

    # FL_bsline = {"tst_loss":{"0": 1.33,
    #                                 "1e-4": 1.32,
    #                                 "5e-4": 1.41,
    #                                 "1e-3": 1.60},
    #                     "tst_acc":{"0":0.53,
    #                                 "1e-4": 0.53,
    #                                 "5e-4": 0.49,
    #                                 "1e-3": 0.42}}

    for i_nm in mtrx_nm:
        plot_contrast(cof_pths, i_nm, False, None)

    # plot_contrast(cof_pths, "tst_acc", True, FL_bsline)
    # plot_contrast(cof_pths, "tst_loss", True, FL_bsline)