import numpy as np
from scipy import special
from sklearn.datasets import load_iris,load_digits,load_breast_cancer
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import torch

class Layer():
	def __init__(self,inNodes, outNodes,activation='linear'):
		self.inNodes = inNodes
		self.outNodes = outNodes		
		self.activation = activation
		self.weights,self.bias = self.GetWeights()	
		self.activationF = self.GetActivationF()
		self.prev_a = None
		self.a = None 
		self.z = None
		self.dVW, self.dVb = (None,None)
		self.dSW, self.dSb = (None,None)

	def GetWeights(self):
		weights = np.random.randn(self.outNodes,self.inNodes)*2/np.sqrt(self.inNodes)
		bias = np.random.randn(self.outNodes,1)*2/np.sqrt(self.inNodes)
		return (weights,bias)

	def __repr__(self):
		return f'Weights : ({self.inNodes},{self.outNodes})\nBias : ({self.outNodes})\nActivation : {self.activation}'

	def GetActivationF(self):
		if self.activation == 'linear':
			return (lambda x : x)

		elif self.activation == 'relu':
			return lambda x : np.maximum(0,x)

		elif self.activation == 'sigmoid':
			return lambda x: special.expit(x)

		elif self.activation == 'tanh':
			return lambda x: np.tanh(x)

		elif self.activation == 'softmax':
			def f(x):
				t = np.exp(x)
				div = np.sum(t, axis=0, keepdims=True)
				r = t/div
				r[(r == np.inf) | (r==-np.inf)] = 0
				return r

			return f

		else:
			raise Exception(f'{self.activation} is invalid')

	def GetActivationDerivative(self):
		if self.activation == 'linear':
			return (lambda x : 1)(self.a)

		elif self.activation == 'relu':
			return (lambda x : np.where(x<0,0,1))(self.a)

		elif self.activation == 'sigmoid':
			return (self.a*(1-self.a))
		elif self.activation == 'tanh':
			return (1-(self.a*self.a))

		else:
			raise Exception(f'{self.activation} is invalid')		

	def __call__(self,X,make_graph=True):
		self.prev_a = X.copy() if make_graph else self.prev_a
		z = (self.weights@X + self.bias)
		a = self.activationF(z)
		self.z = z if make_graph else self.z
		self.a = a if make_graph else self.a
		return a

	def Update(self,alpha,optimizer='SGD',beta=.9):

		dw, db = OptimDeriv(self,optim=optimizer,beta=beta)
		self.weights = self.weights - alpha*dw
		self.bias = self.bias - alpha*db

	def reset_(self):
		self.weights,self.bias = self.GetWeights()
		self.prev_a = None
		self.a = None 
		self.z = None
		self.dVW, self.dVb = (None,None)
		self.dSW, self.dSb = (None,None)
		return self


def OptimDeriv(obj,optim,beta):
	if optim == 'SGD':
		return (obj.dW,obj.db)

	elif optim == 'Momentum':
		obj.dVW = np.zeros_like(obj.dW) if obj.dVW is None else obj.dVW
		obj.dVb = np.zeros_like(obj.db) if obj.dVb is None else obj.dVb

		obj.dVW = beta*obj.dVW + obj.dW
		obj.dVb = beta*obj.dVb + obj.db
		return (obj.dVW,obj.dVb)

	elif optim == 'RMSprop':
		obj.dSW = np.zeros_like(obj.dW) if obj.dSW is None else obj.dSW
		obj.dSb = np.zeros_like(obj.db) if obj.dSb is None else obj.dSb

		obj.dSW = beta*obj.dSW + (1-beta)*np.square(obj.dW)
		obj.dSb = beta*obj.dSb + (1-beta)*np.square(obj.db)
		eps = 1e-8

		return (obj.dW/(np.sqrt(obj.dSW)+eps), obj.db/(np.sqrt(obj.dSb)+eps))

	elif optim == 'ADAM':
		OptimDeriv(obj,optim='Momentum',beta=beta)
		OptimDeriv(obj,optim='RMSprop',beta=beta)
		eps = 1e-8

		return (obj.dVW/(np.sqrt(obj.dSW)+eps), obj.dVb/(np.sqrt(obj.dSb)+eps))



	else:
		raise Exception(f'{optim} not implemented.')



def BCELoss(pred,y):
	loss = -(y*np.log(pred) + (1-y)*np.log(1-pred)).sum()
	pred[pred>=.5]=1
	pred[pred <=.5]=0
	return (round(loss,3), round((pred == y).mean(),3))

def CELoss(pred,y):
	loss = -(y*np.log(pred)).sum()
	return (round(loss,3), round((np.argmax(pred,axis=0)==np.argmax(y,axis=0)).mean(),3))


class Model():
	def __init__(self,layers=[]):
		self.layers = layers
		self.isCompiled = False

	def addLayer(self,layer):
		self.layers.append(layer)

	def __call__(self,X):
		return self.predict(X)

	def predict(self,X,make_graph=True):
		out = X
		for layer in self.layers:
			out = layer(out,make_graph)
		return out

	def __repr__(self):
		for i,each in enumerate(self.layers):
			print('-'*50)
			print('Layer ' + str(i))
			print(each)
		return ''

	def backward(self,y):
		for i, each in reversed(list(enumerate(self.layers))):

			if i != len(self.layers)-1:
				each.dA = self.layers[i+1].weights.T@self.layers[i+1].dZ
				each.dZ = each.dA*each.GetActivationDerivative()
			else:
				each.dZ = each.a - y
			each.dW = each.dZ@each.prev_a.T
			each.db = np.sum(each.dZ,axis=1,keepdims=True)
		return self

	def compile(self,alpha=.01,Metric=CELoss,epochs=10,optimizer='SGD',beta=.9):
		self.alpha = alpha
		self.epochs = epochs
		self.optimizer = optimizer
		self.beta = beta
		self.isCompiled = True
		self.TrainLosses = []
		self.TestLosses = []
		self.Metric = Metric
		return self

	def train(self,X,y,batch_size = 32, TestSet: tuple = None):
		
		assert (self.isCompiled)
		batches = GetBatches(X,y,batch_size)
		for _ in range(self.epochs):
			tLoss = 0
			tBatches = 0
			tAcc = 0
			for ind, (Xb,yb) in enumerate(batches):
				pred = self.predict(Xb)

				TrainLoss,TAccuracy = self.Metric(pred,yb)
				tLoss += TrainLoss
				tBatches += 1
				tAcc += TAccuracy

				print_(_+1,round(tLoss/tBatches,2),round(tAcc/tBatches,2),None,None,ind)

				self.backward(yb)
				self.step()

			self.TrainLosses.append(tLoss/tBatches)

			if TestSet is not None:
				TestLoss, TeAccuracy = self.Metric(self.predict(TestSet[0],make_graph=False),TestSet[1])
				self.TestLosses.append(TestLoss)
			else:
				TestLoss, TeAccuracy = [None]*2
			
			print_(_+1,round(tLoss/tBatches,2),round(tAcc/tBatches,2),TestLoss,TeAccuracy,ind)
			print()
			
		return self


	def step(self):
		[each.Update(self.alpha,optimizer=self.optimizer,beta=self.beta) for each in self.layers]

def GetBatches(X,y,batch_size):
	if batch_size == 0:
		return [(X,y)]
	batches = []
	inds = np.random.permutation(X.shape[1])
	for i in range(batch_size,X.shape[1],batch_size):
		batches.append((X[:,i-batch_size:i],y[:,i-batch_size:i]))
	return batches


def print_(epoch,tloss,tacc,teloss,teacc,curBatch):
	te = f'Test:Loss :{teloss} Accuracy :{teacc}' if teloss is not None else ''
	print(f'EPOCH:{epoch} Batch: {curBatch} Train:Loss:{tloss} Accuracy:{tacc} {te}',end='\r')

def pipeline(imgs,labels):
	X = imgs.numpy().reshape(-1,28*28)
	lbls = labels.reshape(1,-1).numpy()
	oneHot = np.zeros((lbls.shape[1],10))
	oneHot[np.arange(len(X)), lbls.squeeze()] = 1
	y= oneHot.T
	X = X.T
	return X,y


if __name__ == '__main__':

	print('GETTING DATA...')
	d = MNIST(root='myTrain.pt',transform=transforms.ToTensor())
	d2 = MNIST(root='myTrain.pt',transform=transforms.ToTensor(),train=False)
	imgs = torch.stack([img for img,_ in d])
	labels =torch.stack([torch.tensor(_) for img,_ in d])
	imgs.squeeze_(1)

	Teimgs = torch.stack([img for img,_ in d2])
	Telabels =torch.stack([torch.tensor(_) for img,_ in d2])
	Teimgs.squeeze_(1)

	print('DONE')
	X,y = pipeline(imgs,labels)
	Xte,yte= pipeline(Teimgs,Telabels)


	print('STARTING TRAINING..')
	m = Model([Layer(28*28,128,activation='relu'),
	          Layer(128,64,activation='relu'),
	          Layer(64,10,activation='softmax')])

	m.compile(alpha=1e-4,optimizer='SGD',epochs=20)
	m.train(X,y,batch_size=64,TestSet=(Xte,yte))

	n = np.arange(len(m.TrainLosses))
	plt.plot(n,m.TrainLosses)
	plt.show()
	
	
	# d = load_digits() 

	# X,y = d['data'], d['target']
	# inds = np.random.permutation(len(X))
	# X,y = X[inds],y[inds]
	# cutOff = round(.8*len(X))

	# ny = np.zeros((len(X),10))
	# ny[np.arange(len(X)), y] = 1
	# Xt,yt,xte,yte = X[inds[:cutOff]],ny[inds[:cutOff]],X[inds[cutOff:]],ny[inds[cutOff:]]

	# X= X.T
	# ny = ny.T

	# Xt = Xt.T
	# yt = yt.T
	# xte = xte.T
	# yte = yte.T

	# print(Xt.shape,yt.shape,xte.shape,yte.shape)

	# l1 = Layer(64,64,activation='relu')
	# l2 = Layer(64,10,activation='softmax')
	# m = Model([l1,l2])
	# # m.addLayer(l2)
	# # m.addLayer(l3)
	# m.compile(alpha=1e-5,epochs=400,Metric=CELoss,optimizer='Momentum')
	# m.train(Xt,yt,TestSet=(xte,yte),batch_size=0)

	# m2 = Model([l1.reset_(), l2.reset_()])
	# m2.compile(alpha=1e-5,epochs=400,Metric=CELoss,optimizer='SGD')
	# m2.train(Xt,yt,TestSet=(xte,yte),batch_size=0)

	# m3 = Model([l1.reset_(), l2.reset_()])
	# m3.compile(alpha=1e-3,epochs=400,Metric=CELoss,optimizer='RMSprop')
	# m3.train(Xt,yt,TestSet=(xte,yte),batch_size=0)

	# m4 = Model([l1.reset_(), l2.reset_()])
	# m4.compile(alpha=1e-3,epochs=400,Metric=CELoss,optimizer='ADAM')
	# m4.train(Xt,yt,TestSet=(xte,yte),batch_size=0)



	# arr = np.arange(len(m.TrainLosses))
	# plt.plot(arr,m.TrainLosses,label='Momentum')
	# plt.plot(arr,m2.TrainLosses,label='SGD')
	# plt.plot(arr,m3.TrainLosses,label='RMSprop')
	# plt.plot(arr,m4.TrainLosses,label='ADAM')

	# plt.legend()
	# plt.show()



# print(m(X).argmax(axis=0),ny.argmax(axis=0))
# labels = m(X).argmax(axis=0)
# for i in range(10):
# 	plt.subplots()
# 	plt.imshow(X.T[i].reshape(8,8))
# 	print(labels[i])
# 	plt.show()
