import math
import numpy as np

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

a.backward()
b.backward()

currently has the ability to create neural nets without the bias terms. 
'''


class Grad:

	'''
	to do:
	functionality to enable disable graph making
	'''
	# ## vars
	# _makeGraph = True


	# ##funcs
	# def disableGraph():
	# 	_makeGraph = False

	# def enableGraph():
	# 	_makeGraph = True


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
			self.a.backward(grad)
			self.b.backward(grad)

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
			self.a.backward(grad*self.b._val)
			self.b.backward(grad*self.a._val)

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
			return Grad.Variable(math.sin(a._val), self)

		def backward(self, grad):
			self.a.backward(grad*math.cos(self.a._val))


	class cos(Node):
		def __call__(self, a):
			self.a = a
			return Grad.Variable(math.cos(a._val), self)

		def backward(self, grad):
			self.a.backward(-grad*math.sin(self.a._val))

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

	class Variable:
		def __init__(self, val, node = None, requires_grad = True):
			self._val = val
			self._node = node
			self.grad = None
			self.requires_grad = requires_grad

		def __repr__(self):
			return str(self._val)

		def add(self, b):
			return Grad.Add()(self, self.adjustB(b))

		def sub(self, b):
			return Grad.Sub()(self, self.adjustB(b))

		def mul(self, b):
			return Grad.Mul()(self, self.adjustB(b))

		def div(self, b):
			return Grad.Mul(self, self.adjustB(1/b))

		def matmul(self, b):
			return Grad.Matmul()(self, self.adjustB(b))

		def pow(self, b):
			return Grad.Pow()(self, self.adjustB(b))

		def zero(self, b):
			return Grad.Variable(self._val*0, self._node, self.requires_grad)

		def zero_(self, b):
			self._val *= 0
			return self

		def grad_zero(self):
			self.grad *= 0
			return self 

		def sum(self):
			return Grad.Sum()(self)


		def __add__(self, b):
			return self.add(b)

		def __sub__(self, b):
			return self.sub(b)

		def __mul__(self, b):
			return self.mul(b)

		def __matmul__(self, b):
			return self.matmul(b)

		def __rmatmul(self, b):
			return Grad.Matmul()(self.adjustB(b), self)

		def __div__(self, b):
			return self.div(b)

		__rmul__ = __mul__

		def __pow__(self, b):
			return self.pow(b)


		def adjustB(self, b):
			return Grad.Variable(b, requires_grad=False) if not isinstance(b, Grad.Variable) else b

		def backward(self, grad=1):
			# if not isinstance(grad, int): print(grad.shape)
			if self.requires_grad:
				self.grad = grad if not self.grad else self.grad + grad
			if self._node: self._node.backward(grad)
