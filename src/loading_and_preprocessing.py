import pandas as pd
import numpy as np
from utils import make_psd, pretty_print_buffer

# ------- Load Data ------- #

mu1 = pd.read_csv('data/M1.csv',header=None).values.squeeze().reshape(-1) 
mu2 = pd.read_csv('data/M2.csv',header=None).values.squeeze().reshape(-1) 
s1 = pd.read_csv('data/Sigma1.csv',header=None).values
s2 = pd.read_csv('data/Sigma2.csv',header=None).values
s1=make_psd(s1)
s2=make_psd(s2)
P1,P2 = 0.35,0.65

# ------- Generate Data ------- #

n1,n2 = round(P1*1000), round(P2*1000)
gen = np.random.Generator(np.random.PCG64())
DATA1 = gen.multivariate_normal(mean=mu1, cov=s1, size=n1)
DATA2 = gen.multivariate_normal(mean=mu2, cov=s2, size=n2)

#------- Prepare Training Data ------- #

x_train= np.vstack((DATA1, DATA2))
y_train = np.hstack((np.ones(n1), np.zeros(n2)))

pretty_print_buffer()
#sanity check
print(x_train.shape)
print(y_train.shape)

permutation = gen.permutation(len(x_train)) # shuffle data

x_train , y_train = x_train[permutation], y_train[permutation]


# ------- Compute MLE'S ------- #

mu1_MLE = np.mean(x_train[y_train==1], axis=0)
mu2_MLE = np.mean(x_train[y_train==0], axis=0)
s1_MLE = np.cov(x_train[y_train==1], rowvar=False)
s2_MLE = np.cov(x_train[y_train==0], rowvar=False)

pretty_print_buffer()

print(f'MLE mu1: {mu1_MLE}')
print(f'MLE mu2: {mu2_MLE}')
print(f'MLE s1: {s1_MLE}')
print(f'MLE s2: {s2_MLE}')
pretty_print_buffer()

print("comparing with true values:")
print(f"True mu1: {mu1}")
pretty_print_buffer()

print(f"True mu2: {mu2}")
pretty_print_buffer()

print(f"True s1: {s1}")
pretty_print_buffer()

print(f"True s2: {s2}")

#compute errors:
mu1_error = np.linalg.norm(mu1 - mu1_MLE)
mu2_error = np.linalg.norm(mu2 - mu2_MLE)
s1_error = np.linalg.norm(s1 - s1_MLE,ord='fro')
s2_error = np.linalg.norm(s2 - s2_MLE,ord='fro')

print(f'Mu1 estimation error (L2 norm): {mu1_error}')
pretty_print_buffer()


print(f'Mu2 estimation error (L2 norm): {mu2_error}')
pretty_print_buffer()

print(f'S1 estimation error (Frobenius norm): {s1_error}')
pretty_print_buffer()

print(f'S2 estimation error (Frobenius norm): {s2_error}')
pretty_print_buffer()



# Generating Validation Data (2000 observations)

n1_val, n2_val = round(P1*2000), round(P2*2000)
DATA1_val = gen.multivariate_normal(mean=mu1, cov=s1, size=n1_val)
DATA2_val = gen.multivariate_normal(mean=mu2, cov=s2, size=n2_val)

# c. ------- Prepare Validation Data ------- #

x_val = np.vstack((DATA1_val, DATA2_val))
y_val = np.hstack((np.ones(n1_val), np.zeros(n2_val)))

# Shuffle validation data
permutation_val = gen.permutation(len(x_val))
x_val , y_val = x_val[permutation_val], y_val[permutation_val]
# sanity check
pretty_print_buffer()

print(x_val.shape)
print(y_val.shape)


