import numpy as np
import os

infile_SGD = "SGD_LogReg.txt"
infile_MOM = "SGD_mom_LogReg.txt"
infile_ADAM = "ADAM_LogReg.txt"


SGD_acc_val = []
SGD_acc_test = []
SGD_epochs = []
SGD_batch = []
SGD_eta = []

MOM_acc_val = []
MOM_acc_test = []
MOM_epochs = []
MOM_batch = []
MOM_eta = []
MOM_gamma = []

ADAM_acc_val = []
ADAM_acc_test = []
ADAM_epochs = []
ADAM_batch = []
ADAM_eta = []
ADAM_beta1 = []
ADAM_beta2 = []


with open(infile_SGD, "r") as infile:
    line = infile.readline()
    lines = infile.readlines()
    for line in lines:
        val = line.split(" ")
        SGD_acc_val.append(float(val[0]))
        SGD_acc_test.append(float(val[1]))
        SGD_epochs.append(float(val[2]))
        SGD_batch.append(float(val[3]))
        SGD_eta.append(float(val[4]))

with open(infile_MOM, "r") as infile:
    line = infile.readline()
    lines = infile.readlines()
    for line in lines:
        val = line.split(" ")
        MOM_acc_val.append(float(val[0]))
        MOM_acc_test.append(float(val[1]))
        MOM_epochs.append(float(val[2]))
        MOM_batch.append(float(val[3]))
        MOM_eta.append(float(val[4]))
        MOM_gamma.append(float(val[5]))

with open(infile_ADAM, "r") as infile:
    line = infile.readline()
    lines = infile.readlines()
    for line in lines:
        val = line.split(" ")
        ADAM_acc_val.append(float(val[0]))
        ADAM_acc_test.append(float(val[1]))
        ADAM_epochs.append(float(val[2]))
        ADAM_batch.append(float(val[3]))
        ADAM_eta.append(float(val[4]))
        ADAM_beta1.append(float(val[5]))
        ADAM_beta2.append(float(val[6]))

SGD_acc_val = np.array(SGD_acc_val)
SGD_acc_test = np.array(SGD_acc_test)
SGD_epochs = np.array(SGD_epochs)
SGD_batch = np.array(SGD_batch)
SGD_eta = np.array(SGD_eta)


MOM_acc_val = np.array(MOM_acc_val)
MOM_acc_test = np.array(MOM_acc_test)
MOM_epochs = np.array(MOM_epochs)
MOM_batch = np.array(MOM_batch)
MOM_eta = np.array(MOM_eta)
MOM_gamma = np.array(MOM_gamma)

ADAM_acc_val = np.array(ADAM_acc_val)
ADAM_acc_test = np.array(ADAM_acc_test)
ADAM_epochs = np.array(ADAM_epochs)
ADAM_batch = np.array(ADAM_batch)
ADAM_eta = np.array(ADAM_eta)
ADAM_beta1 = np.array(ADAM_beta1)
ADAM_beta2 = np.array(ADAM_beta2)

SGD_best_val = np.where(SGD_acc_val == np.max(SGD_acc_val))
MOM_best_val = np.where(MOM_acc_val == np.max(MOM_acc_val))
ADAM_best_val = np.where(ADAM_acc_val == np.max(ADAM_acc_val))

print(" ")
print("-------- Winners SGD --------")
print("Accuracy Validation = ", SGD_acc_val[SGD_best_val])
print("Accuracy Test = ", SGD_acc_test[SGD_best_val])
print("Epochs = ", SGD_epochs[SGD_best_val])
print("Batch Size = ", SGD_batch[SGD_best_val])
print("Eta = ", SGD_eta[SGD_best_val])
print(" ")
print("-------- Winners MOM --------")
print("Accuracy Validation = ", MOM_acc_val[MOM_best_val])
print("Accuracy Test = ", MOM_acc_test[MOM_best_val])
print("Epochs = ", MOM_epochs[MOM_best_val])
print("Batch Size = ", MOM_batch[MOM_best_val])
print("Eta = ", MOM_eta[MOM_best_val])
print("Gamma = ", MOM_gamma[MOM_best_val])
print(" ")
print("-------- Winners ADAM --------")
print("Accuracy Validation = ", ADAM_acc_val[ADAM_best_val])
print("Accuracy Test = ", ADAM_acc_test[ADAM_best_val])
print("Epochs = ", ADAM_epochs[ADAM_best_val])
print("Batch Size = ", ADAM_batch[ADAM_best_val])
print("Eta = ", ADAM_eta[ADAM_best_val])
print("Beta1 = ", ADAM_beta1[ADAM_best_val])
print("Beta2 = ", ADAM_beta2[ADAM_best_val])
print(" ")
