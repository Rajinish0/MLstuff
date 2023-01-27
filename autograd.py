import numpy as np
from scipy.signal import fftconvolve

'''
a pytorch like mini automatic diff. library. 
usage : 
var = Grad.Variable

a = var(1)
b = var(2)

c = (2*(a+b))**2 ## or c = f(a, b) any function
c.backward()

a.grad, b.grad 

works with numpy too
a = var(np.random.randn(32, 4))
b = var(np.random.randn(32, 4))
c = (0.5*(a-b)**2).sum()

c.backward()

a.grad, b.grad

currently has the ability to create simple neural nets. 

the conv2d node is experimental, although it is working (the grads are flowing, loss is going down) I haven't checked if my partial derivatives are actually correct
need to tie the weights with torch or tf and make the checks.

W for conv2d is of shape (nK, kH, kW, nC) where nK = num of kernels, kH/W= kernel height/width, nC is number of input channels
'''

def convFunc(x, kernel):
    ## i could do this by getting submatrices of the x matrix of size kernel each, but that'd require me store all of them. So I went the easy way and just looped through
    ## the batch size and number of kernels which should be considerably smaller than image size (on average), so this should be on average faster than looping through an image
    bs, imgh, imgw, _ = x.shape
    nK, kh, kw, _ = kernel.shape
    o = np.empty((bs, imgh-kh+1, imgw-kw+1, nK))
    for b in range(bs):
        for k in range(nK):
            o[b, :, :, k] = fftconvolve(x[b], kernel[k], 'valid')[..., 0] ## I'm not yet sure how to extend my code for simple 1d fft to 2d fft, but this should be faster than the convolve function
    return o



class Grad:

	'''
	TO DO:
	-> functionality to enable disable graph making
	-> add axis functionality to sum
	-> functionality of padding, strides for conv2d.
	-> gradient checks for conv2d.
	'''
	# ## vars
	# _makeGraph = True


	# ##funcs
	# def disableGraph():
	# 	_makeGraph = False

	# def enableGraph():
	# 	_makeGraph = True

	@staticmethod
	def undobroadCast(a, output):
		## a was added to something and was broadcasted to get output
		'''
		NEED TO FIX : 
		i'm not called out.squeeze() because for instance if 
		a.shape = (1, 4, 1, 3)
		then it should stay like that, out.squeeze would make it (4, 3)
		'''
		s1 = a.shape
		s2 = output.shape
		out = output

		for i, each in enumerate(s2[:(len(s1))]):
			if s1[ -(i+1) ] != s2[ -(i+1) ]:
				out = out.sum(axis=-(i+1), keepdims=True)

		for axis in range(len(s2) - len(s1)):
			out = out.sum(axis=axis, keepdims=True)

		return out


	class Node:
		def __init__(self):
			pass

		def __call__(self):
			##store values
			pass

		def backward(self, grad=1):
			pass


	class Add(Node):

		def __call__(self, a, b):
			## need to fix if the shapes are not equal
			self.a = a
			self.b = b
			return Grad.Variable(self.a._val + self.b._val, self)

		def backward(self, grad=1):
			self.a.backward( Grad.undobroadCast(self.a._val ,grad ) )
			self.b.backward( Grad.undobroadCast(self.b._val, grad ) )

	class Sub(Node):


		def __call__(self, a, b):
			self.a = a 
			self.b = b
			return Grad.Variable(self.a._val - self.b._val, self)

		def backward(self, grad=1):
			self.a.backward(grad)
			self.b.backward(-grad)


	class Mul(Node):


		def __call__(self, a, b):
			self.a = a 
			self.b = b
			return Grad.Variable(self.a._val*self.b._val, self)

		def backward(self, grad=1):
			self.a.backward( Grad.undobroadCast(self.a._val, grad*self.b._val))
			self.b.backward( Grad.undobroadCast(self.b._val, grad*self.a._val))

	class Div(Node):
		def __call__(self, a, b):
			self.a = a 
			self.b = b 
			return Grad.Variable(self.a._val/self.b._val, self)

		def backward(self, grad=1):
			self.a.backward(grad/self.b._val)
			self.b.backward( (-grad*self.a._val)/(self.b._val*self.b._val) )

	class Pow(Node):

		def __call__(self, a, b):
			self.a = a 
			self.b = b 
			return Grad.Variable(pow(self.a._val, self.b._val), self)

		def backward(self, grad):
			self.a.backward(grad*self.b._val*(self.a._val**(self.b._val-1)))
			self.b.backward(grad*pow(self.a._val, self.b._val)*np.log(self.a._val))

	class Matmul(Node):
		def __call__(self, a, b):
			self.a = a 
			self.b = b 
			return Grad.Variable(self.a._val@self.b._val, self)

		def backward(self, grad):
			# print(self.a._val.shape, self.b._val.shape, grad.shape)
			self.a.backward( (grad)@(self.b._val.T)  )
			self.b.backward( (self.a._val.T)@grad  )

	class Sum(Node):
		def __call__(self, a):
			self.a = a
			return Grad.Variable(self.a._val.sum(), self)

		def backward(self, grad):
			self.a.backward(np.ones_like(self.a._val)*grad)

	class sin(Node):
		def __call__(self, a):
			self.a = a
			return Grad.Variable(np.sin(a._val), self)

		def backward(self, grad):
			self.a.backward(grad*np.cos(self.a._val))


	class cos(Node):
		def __call__(self, a):
			self.a = a
			return Grad.Variable(np.cos(a._val), self)

		def backward(self, grad):
			self.a.backward(-grad*np.sin(self.a._val))

	class sigmoid(Node):
		def __call__(self, a):
			self.a = a 
			self.sig = 1/(1+np.exp(-self.a._val) )
			return Grad.Variable(
				self.sig ,
				self
			 )

		def backward(self, grad):
			self.a.backward(grad*self.sig*(1-self.sig))

	class relu(Node):
		def __call__(self, a):
			self.a = a 
			return Grad.Variable(
				np.where(self.a._val > 0, self.a._val, 0),
				self
				)

		def backward(self, grad):
			self.a.backward(grad*np.where(self.a._val>=0, 1, 0))

	class neg(Node):
		def __call__(self, a):
			self.a = a
			return Grad.Variable(
				-self.a._val, 
				self 
				)

		def backward(self, grad):
			self.a.backward(-grad)

	class Log(Node):
		def __call__(self, a):
			self.a = a 
			return Grad.Variable(
				np.log(self.a._val),
				self 
				)

		def backward(self, grad):
			self.a.backward(grad/self.a._val)

	class Flatten(Node):
		def __call__(self, a):
			self.a = a
			bs = a._val.shape[0]
			return Grad.Variable(
				self.a._val.reshape(bs, -1),
				self
				)

		def backward(self, grad):
			self.a.backward(grad.reshape(self.a._val.shape))


	class Conv2D(Node):
		def __call__(self, X, kernel):
			self.X = X 
			self.kernel = kernel
			return Grad.Variable( convFunc(X._val, kernel._val),
								  self)


		def backward(self, grad):
			_, kh, kw, _ = self.kernel.shape 
			_, imh, imw, _ = self.X.shape 
			_, gradh, gradw, _ = grad.shape

			## gradh-kh+1+2p = imh -> 2p = imh+kh-gradh-1
			padHeight = (imh + kh -1 -gradh)//2
			padWidth = (imw+kw-1-gradw)//2
			self.kernel.backward(  
			np.transpose(convFunc( np.transpose(X._val, (3, 1, 2, 0) ), 
							  	  np.transpose(grad,   (3, 1, 2, 0) ) ),
							  	  (3, 1, 2, 0) ))

			self.X.backward(
				convFunc(np.pad(grad, ((0,0), (padHeight,padHeight), (padWidth,padWidth), (0,0))),
				   		 np.transpose(self.kernel._val, (3, 1,2,0))[:, ::-1, ::-1, :] )
				)
	class Variable:
		def __init__(self, val, node = None, requires_grad = True):
			self._val = np.array(val)
			self._node = node
			self.grad = None
			self.requires_grad = requires_grad

		@property
		def shape(self):
			return self._val.shape
		

		def __repr__(self):
			return str(self._val)

		def zero(self, b):
			return Grad.Variable(self._val*0, self._node, self.requires_grad)

		def zero_(self, b):
			self._val *= 0
			return self

		def grad_zero(self):
			self.grad *= 0
			# del self._node 
			return self 

		def sum(self):
			return Grad.Sum()(self)


		def __add__(self, b):
			return Grad.Add()(self, self.adjustB(b))

		def __sub__(self, b):
			return Grad.Add()(self, self.adjustB(-b))

		def __mul__(self, b):
			return Grad.Mul()(self, self.adjustB(b))

		__rmul__ = __mul__

		def __matmul__(self, b):
			return Grad.Matmul()(self, self.adjustB(b))

		def __rmatmul__(self, b):
			return Grad.Matmul()(self.adjustB(b), self)

		def __rtruediv__(self, b):
			return Grad.Div()(self.adjustB(b), self)

		def __truediv__(self, b):
			return Grad.Mul()(self, self.adjustB(1/b))

		def __pow__(self, b):
			return Grad.Pow()(self, self.adjustB(b))

		def __neg__(self):
			return Grad.neg()(self)


		def adjustB(self, b):
			return Grad.Variable(b, requires_grad=False) if not isinstance(b, Grad.Variable) else b

		def backward(self, grad=np.array(1)):
			if self.requires_grad:
				self.grad = grad if not self.grad else self.grad + grad
			if self._node: self._node.backward(grad)
			


'''	
TESTS
import torch
from torch import nn

var = Grad.Variable

# a = var(2)
# b = var(3)
# c = var(7)
# # g = (a.add(b)).mul(9).mul(c)

# g = (a + b)*9*c

h = np.random.randn(32, 50)
h -= h.mean(axis=0, keepdims=True)
h /= h.std(axis=0, keepdims=True)
X = var(h, requires_grad=False)
y= np.random.randn(32, 10)
W = var(np.random.randn(50, 64)/50)
b1 = var(np.random.randn(64))
W2 = var(np.random.randn(64, 128)/64)
b2 = var(np.random.randn(128))
W3 = var(np.random.randn(128, 10)/128)
b3 = var(np.random.randn(10))

w1 = nn.Linear(50, 64, bias=True)
w2 = nn.Linear(64, 128, bias=True)
w3 = nn.Linear(128, 10, bias=True)
w1.weight = nn.Parameter(torch.tensor(W._val.T))
w1.bias = nn.Parameter(torch.tensor(b1._val))
w2.weight = nn.Parameter(torch.tensor(W2._val.T))
w2.bias = nn.Parameter(torch.tensor(b2._val))
w3.weight = nn.Parameter(torch.tensor(W3._val.T))
w3.bias = nn.Parameter(torch.tensor(b3._val))



act = torch.sigmoid(w3(w2(w1(torch.tensor(X._val)))))
# v = (0.5*((A-Y)**2)).sum()
# v.backward()

# print(A.grad, Y.grad)
# print(A-Y)


# d = Grad.sigmoid()(X.matmul(W).matmul(W2).matmul(W3))
d = Grad.sigmoid()(((X@W+b1)@W2 + b2)@W3 + b3)
print(d._val)
print(act)

# print(np.where(d._val>0, d._val, 0))
# print(torch.relu(act.T))
# print(d, d._node)
val = np.array(0.2).astype(np.float32)
print(type(val))
v = ( 0.5*((d - y)**2) ).sum()
# print(v, v._node)
# v.backward()

print(W.grad)
v2=  (0.5*(act - torch.tensor(y))**2).sum()
v.backward()
v2.backward()

print(W.grad.shape, w1.weight.grad.shape)
print(W.grad)
# print(b1.grad.shape, "HEREagain")
print(w1.weight.grad.T)
print(W2.grad)
print(w2.weight.grad.T)
print(W3.grad)
print(w3.weight.grad.T)


print(b1.grad)
print(w1.bias.grad)

print(b2.grad)
print(w2.bias.grad)

print(b3.grad)
print(w3.bias.grad)

# x = var(1)
# y = var(2)
# g = y/2 
# g.backward()
# # print(x.grad)
# print(y.grad)


x = var( np.random.randn(1, 4) )
y = var(np.random.randn(5, 4))
g = (x * y).sum()
g.backward()

print(y._val.sum(axis=0))
print(x.grad)
print(y.grad)
print(x)



X = var(np.random.randn(32, 28, 28, 3)/100, requires_grad=False)
y = np.random.randn(32, 10)
W1 = var(np.random.randn(2, 3, 3, 3))
W3 = var(np.random.randn(1352, 10))


for i in range(25):
	o = Grad.Conv2D()(X, W1)
	o = Grad.Flatten()(o)
	o = o@W3

	L = (0.5*(o-y)**2).sum()
	print(L)
	L.backward()
	W1 = W1 - (1e-2)*W1.grad
	W3 = W3 - (1e-2)*(W3.grad)
	W1._node = None 
	W3._node = None
# W1.grad_zero()
# W3.grad_zero()

	# print(W1.grad.shape)
	# print(W3.grad.shape)


# X = var (np.random.randn(32, 28, 38, 3))


# e = b**(2)
# print(e)
# e.backward()
# print(b.grad)

# g = b.mul(b)
# print(g)

# g.backward()
# print(a.grad)
# print(b.grad)
# print(c.grad)
'''
