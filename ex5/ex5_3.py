import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X = np.loadtxt(fname='log_loss_data/X.csv', delimiter=',', usecols=[0,1], dtype=int)
X=np.array(X)
y = np.loadtxt(fname='log_loss_data/y.csv', delimiter=',', usecols=0, dtype=int)
y=np.array(y)
w=np.array([1,-1])

step_size=0.001
W=[]
accuacies = []
losses=[]

def grad(w,X,y):
    gradW=np.array([0,0])
    for j in range(y.size):
        up=np.exp(-y[j]*np.dot(w,X[j]))*y[j]*X[j]
        down=1+np.exp(-y[j]*np.dot(w,X[j]))
        gradW=gradW-(up/down)
    return gradW

def log_loss1(w,X,y):
    loss=0
    eps = 1e-15
    p = np.clip(y, eps, 1 - eps)
    loss = np.sum(- np.dot(X,w) * np.log(p) - (1 - np.dot(X,w)) * np.log(1 - p))
    return loss


accuracy=[]
Mean_accuacy=[]

for iteration in range(100):

    w= w-step_size*grad(w,X,y)

    loss_val= log_loss1(w,X,y)
    print("Iteration %d:w=%s(log-loss=%.2f)"%(iteration,str(w),loss_val))

    y_prob=1/(1+np.exp(-np.dot(X,w)))

    y_pred=(y_prob>0.5).astype(int)

    y_pred=2*y_pred-1

    accuracy.append(accuracy_score(y, y_pred))

    mean_accuacy=np.mean(accuracy)
    Mean_accuacy.append(mean_accuacy)

    print("Accuracy is %f" % mean_accuacy)

    W.append(w)

W=np.array(W)

plt.plot(W[:,0],W[:,1],'ro')
plt.ylabel('w0')
plt.xlabel('w1')
plt.grid()
plt.show()

plt.plot(Mean_accuacy,'b')
plt.ylabel('Accuracy/%')
plt.xlabel('iteration')
plt.grid()
plt.show()
