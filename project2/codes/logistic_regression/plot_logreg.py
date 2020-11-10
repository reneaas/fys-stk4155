import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os

#Lambda = 0 for SGD, found lambda = for MOM and used that with ADAM.
#eta = 0.1 for all, and found epochs = 100 and batch size = 500 from SGD which is used for MOM and ADAM.


# Results from SGD

path_to_plot = "../../results/LogisticRegression/Plots/"
path_to_files = "../../results/LogisticRegression/"

infile_SGD = path_to_files +  "SGD_LogReg_final.txt"

SGD_acc_val = []
SGD_acc_test = []
SGD_epochs = []
SGD_batch = []

with open(infile_SGD, "r") as infile:
    line = infile.readline()
    lines = infile.readlines()
    for line in lines:
        val = line.split(" ")
        SGD_acc_val.append(float(val[0]))
        SGD_acc_test.append(float(val[1]))
        SGD_epochs.append(float(val[2]))
        SGD_batch.append(float(val[3]))


SGD_epochs = [1, 10, 20, 50, 100, 1000]
SGD_batch = [10, 100, 200, 300, 500]

SGD_acc_val = np.array(SGD_acc_val)
SGD_acc_test = np.array(SGD_acc_test)
SGD_epochs =  np.array(SGD_epochs)
SGD_batch = np.array(SGD_batch)



SGD_best_val = np.where(SGD_acc_val == np.max(SGD_acc_val))

SGD_acc_val_mat = np.zeros([len(SGD_epochs), len(SGD_batch)])
SGD_acc_val_mat.flat[:] = SGD_acc_val

figename_SGD = path_to_plot + "SGD_LogReg_epochs_vs_batchsz.pdf"

sb.set(font_scale=1.25)
heat_map = sb.heatmap(SGD_acc_val_mat.T,annot=True, cbar=True, cbar_kws={"label": "Accuracy", "orientation" : "vertical"})
heat_map.set_xlabel("Epochs")
heat_map.set_ylabel("Batch Size")
heat_map.set_xticklabels(SGD_epochs)
heat_map.set_yticklabels(SGD_batch)
heat_map.xaxis.tick_top()
heat_map.set_ylim(5.0,0.0)
heat_map.tick_params(length=0)
plt.savefig(figename_SGD)
plt.close()


#Results from SGD with momentum
infile_MOM = path_to_files +  "SGD_mom_LogReg_final.txt"

MOM_acc_val = []
MOM_acc_test = []
MOM_gamma = []
MOM_lambda = []

with open(infile_MOM, "r") as infile:
    line = infile.readline()
    lines = infile.readlines()
    for line in lines:
        val = line.split(" ")
        MOM_acc_val.append(float(val[0]))
        MOM_acc_test.append(float(val[1]))
        MOM_gamma.append(float(val[2]))
        MOM_lambda.append(float(val[3]))

MOM_gamma = [10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]
MOM_lambda = [10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0]

MOM_acc_val = np.array(MOM_acc_val)
MOM_acc_test = np.array(MOM_acc_test)
MOM_gamma = np.array(MOM_gamma)
MOM_lambda = np.array(MOM_lambda)




idx_MOM = np.where(MOM_lambda > 10**-5)


MOM_acc_val_mat = np.zeros([len(MOM_gamma), len(MOM_lambda)])
MOM_acc_val_mat.flat[:] = MOM_acc_val

MOM_best_val = np.where(MOM_acc_val_mat[:,4:] == np.max(MOM_acc_val_mat[:,4:]))
print(MOM_best_val)
# Gamma = 1e-7 and Lambda = 1e-4

figename_MOM = path_to_plot + "MOM_LogReg_gamma_vs_lambda.pdf"

sb.set(font_scale=1.25)
heat_map = sb.heatmap(MOM_acc_val_mat[:,4:].T, annot=True, cbar=True, cbar_kws={"label": "Accuracy", "orientation" : "vertical"})
heat_map.set_xlabel(r"$\gamma$")
heat_map.set_ylabel(r"$\lambda$")
heat_map.set_xticklabels(MOM_gamma, rotation = 45)
heat_map.set_yticklabels(MOM_lambda[4:])
heat_map.xaxis.tick_top()
heat_map.set_ylim(5.0,0.0)
heat_map.tick_params(length=0)
plt.savefig(figename_MOM)
plt.show()
plt.close()


#Results from ADAM
infile_ADAM = path_to_files +  "ADAM_LogReg_final.txt"

ADAM_acc_val = []
ADAM_acc_test = []
ADAM_beta1 = []
ADAM_beta2 = []

with open(infile_ADAM, "r") as infile:
    line = infile.readline()
    lines = infile.readlines()
    for line in lines:
        val = line.split(" ")
        ADAM_acc_val.append(float(val[0]))
        ADAM_acc_test.append(float(val[1]))
        ADAM_beta1.append(float(val[2]))
        ADAM_beta2.append(float(val[3]))

ADAM_beta1 = [0.80, 0.85, 0.90, 0.95, 0.99, 0.999]
ADAM_beta2 = [0.80, 0.85, 0.90, 0.95, 0.99, 0.999]

ADAM_acc_val = np.array(ADAM_acc_val)
ADAM_acc_test = np.array(ADAM_acc_test)
ADAM_beta1 = np.array(ADAM_beta1)
ADAM_beta2 = np.array(ADAM_beta2)

ADAM_best_val = np.where(ADAM_acc_val == np.max(ADAM_acc_val))
print(ADAM_best_val)
ADAM_acc_val_mat = np.zeros([len(ADAM_beta1), len(ADAM_beta2)])
ADAM_acc_val_mat.flat[:] = ADAM_acc_val

figename_ADAM = path_to_plot + "ADAM_LogReg_beta1_vs_beta2.pdf"

sb.set(font_scale=1.25)
heat_map = sb.heatmap(ADAM_acc_val_mat.T, annot=True, cbar=True, cbar_kws={"label": "Accuracy", "orientation" : "vertical"})
heat_map.set_xlabel(r"$\beta_1$")
heat_map.set_ylabel(r"$\beta_2$")
heat_map.set_xticklabels(ADAM_beta1)
heat_map.set_yticklabels(ADAM_beta2)
heat_map.xaxis.tick_top()
heat_map.set_ylim(6.0,0.0)
heat_map.tick_params(length=0)
plt.savefig(figename_ADAM)
plt.show()
plt.close()
