import numpy as np

# x1*w1+x2*w2+b>0 => return 1
# x1*w1+x2*w2+b<=0 => return 0
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(x*w)+b
    if tmp>0:
        return 1
    else:
        return 0

def NAND(x1,x2):
    return not AND(x1,x2)

def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.8,0.8])
    b=-0.7
    tmp=np.sum(x*w)+b
    if tmp>0:
        return 1
    else:
        return 0

# single perceptron can't implement XOR gate
# try use multi-layer perceptron

def XOR(x1,x2):
    return AND(NAND(x1,x2),OR(x1,x2))