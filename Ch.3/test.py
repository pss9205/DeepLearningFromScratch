import activationFunction as f
import matplotlib.pylab as plt
import numpy as np



def show_functionShape():
    x = np.arange(-5.0,5.0,0.1)
    yst=f.step_function(x)
    ysg=f.sigmoid(x)
    plt.plot(x,yst)
    plt.ylim(-0.1,1.1)
    plt.show()
    plt.plot(x,ysg)
    plt.ylim(-0.1,1.1)
    plt.show()
class NN3:
    def init_network(self):
        network={}
        network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
        network['b1']=np.array([0.1,0.2,0.3])
        network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
        network['b2']=np.array([0.1,0.2])
        network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
        network['b3']=np.array([0.1,0.2])
        return network
    def forward(self,x,network):
        W1,W2,W3=network['W1'],network['W2'],network['W3']
        b1,b2,b3=network['b1'],network['b2'],network['b3']

        a1=np.dot(x,W1)+b1
        z1=f.sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        z2=f.sigmoid(a2)
        a3=np.dot(z2,W3)+b3
        y=self.identity_function(a3)

        return y
    def identity_function(self,x):
        return x
    
    def showResult(self):
        x=np.array([1.0,0.5])
        y=self.forward(x,self.init_network())
        print(y)

test=NN3()
test.showResult()