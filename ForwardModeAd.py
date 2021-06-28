## forward mode automatic differentiation

from functools import reduce
import math
import numpy as np
from scipy import special
from sklearn.datasets import load_breast_cancer

class Dual():
	def __init__(self,re,eps=0,requires_grad=False):
		self.re = np.array(re)
		self.eps = np.array(eps)
		self.requires_grad=requires_grad

	@property
	def __name__(self):
		return 'dual'

	@property
	def shape(self):
		return self.re.shape

	@property
	def T(self):
		return Dual(self.re.T,self.eps.T)

	def __add__(self,num):
		reqgrad = (self.requires_grad or num.requires_grad)
		return Dual(self.re+num.re, self.eps+num.eps,reqgrad) if isinstance(num,Dual) else Dual(self.re+num,self.eps,reqgrad)

	def __rmul__(self,num):
		try:
			reqgrad = (self.requires_grad or num.requires_grad)
		except:
			reqgrad = self.requires_grad
		return Dual(self.re*num.re,self.re*num.eps+num.re*self.eps,reqgrad) if isinstance(num,Dual) else Dual(num*self.re,num*self.eps,reqgrad)


	__mul__ = __rmul__

	def __sub__(self,num):
		return self.__add__(-1*num)

	def __truediv__(self,scl):
		return Dual(self.re/scl,self.eps/scl)

	def __repr__(self):
		s = '+' + str(self.eps) + 'e' if self.requires_grad else ''
		return f'{self.re}+{self.eps}e'

	def __pow__(self,power):
		if type(power) != int:
			raise Exception(f"POWER TYPE '{type(power).__name__}' NO CAN DO AMIGO.")
		return reduce(lambda x,y : x*y, [self for i in range(power)])

	def sqrt(self):
		return Dual(np.sqrt(self.re),self.eps*.5/(np.sqrt(self.re)),self.requires_grad)

	def __matmul__(self,mat):
		reqgrad = (self.requires_grad or mat.requires_grad)             
		return Dual(self.re@mat.re,self.eps@mat.re + self.re@mat.eps,reqgrad)

	def __rmatmul__(self,mat):
		reqgrad = (self.requires_grad or mat.requires_grad)
		return Dual(mat.re@self.re,mat.re@self.eps + mat.eps@self.re,reqgrad)

	def __neg__(self):
		return Dual(-self.re,-self.eps)

	def __eq__(self,other):
		return self.re == other.re

	def __iadd__(self,other):
		self.re += other.re
		self.eps += other.eps
		return self
	def __isub__(self,other):
		return self.__iadd__(-other)

	@staticmethod
	def randn(*args, requires_grad=False):
		X = np.random.randn(*args)
		E = np.zeros_like(X) if not requires_grad else np.eye(X.shape[0])
		return Dual(X,E,requires_grad)

	@staticmethod
	def array(l, requires_grad=False):
		X = np.array(l)
		E = np.zeros_like(X) if not requires_grad else np.eye(X.shape[0])
		return Dual(X,E,requires_grad)

	@staticmethod
	def log(arr):
		X = np.log(arr.re)

		if arr.requires_grad:
			D = (1/arr.re)
			E = np.zeros([D.shape[0]]*2)
			np.fill_diagonal(E,D)
		else:
			E = np.zeros_like(X)
		return Dual(X,E@arr.eps,arr.requires_grad)

	@staticmethod
	def sigmoid(arr):
		X = special.expit(arr.re)

		if arr.requires_grad:
			D = X*(1-X)
			E = np.zeros([D.shape[0]]*2)
			np.fill_diagonal(E,D)
		else:
			E = np.zeros_like(X)
		return Dual(X,E@arr.eps,arr.requires_grad)

def f(x,y):
        return ((x**2) * y + x + y)
arr = np.array
sigmoid = Dual.sigmoid
log = Dual.log
d = load_breast_cancer()
X = d['data']
y = d['target'].reshape(-1,1)
X = Dual.array(X)
y = Dual.array(y)

W = Dual.randn(30,1,requires_grad=True)*.001


def accuracy(pred,y):
        pred.re[pred.re>.5]=1
        pred.re[pred.re<.5]=0
        return (pred.re==y.re).mean()


for i in range(3000):
    W.eps = np.eye(30)
    Z = (X@W)
    A = sigmoid(Z)
    TERR = y*log(A)+(Dual(1)-y)*log(Dual(1) - A)
    ER = (-(Dual(np.ones((1,569)),np.zeros((1,569))))@TERR)
    print(ER.re,accuracy(A,y),end='\r')
    W -= (Dual(ER.eps.T)*1e-8)
