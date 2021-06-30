import torch,copy
from torch import nn
class Input(nn.Module):
	def __init__(self,input_shape):
		super().__init__()
		self.units = input_shape
		self.store_output = False
		self.output = False
		# self.layer =nn.Linear(1,2)
		# self._backward_hooks,self._forward_hooks,self._forward_pre_hooks= torch.tensor([]),torch.tensor([]),torch.tensor([])

	@property
	def built(self):
		return True
	
	def forward(self,X):
		out = X*1
		self.output = out if self.store_output else None
		return out

	def copy(self):
		return self
		
	def __repr__(self):
		return 'Input({})'.format(self.units)

	def storeOutput(self):
		self.store_output = True

class Dense(nn.Module):
	def __init__(self,units,activation=nn.ReLU,input_shape=None):
		super().__init__()
		self.units = units
		self.activation = activation
		self.rawactivation = copy.deepcopy(activation)
		self.W = nn.Linear(input_shape,units) if input_shape is not None else None
		self.store_output = False
		self.output = None

	@property
	def built(self):
		return True if self.W is not None else False

	def forward(self,X):
		try:
			act = self.activation(self.W(X))
			self.output = act if self.store_output else None
			return act
		except:
			raise Exception('Layer not built')

	def storeOutput(self):
		self.store_output = True

	def copy(self):
		d = Dense(copy.copy(self.units),copy.copy(self.rawactivation))
		if self.W is not None:
				wCopy = self.W.weight.detach().clone()
				bCopy = self.W.bias.detach().clone()
				n = nn.Linear(self.W.weight.shape[1],self.W.weight.shape[0])
				n.weight = nn.Parameter(wCopy)
				n.bias = nn.Parameter(bCopy)
				d.W = n
		else:
				d.W = None
		return d

class Concat(nn.Module):
	def __init__(self,index):
		super().__init__()
		self.index = index
		self.built = False
		self.store_output = False
		self.ouput = None

	def built(self):
		return self.built

	def build(self,prevLayer,model):
		self.layer = model.layers[self.index]
		self.layer.storeOutput()
		self.units = self.layer.units+prevLayer.units
		self.built = True

	def forward(self,X):
		try:
			out = torch.cat((self.layer.output,X),axis=1)
			self.output = out if self.store_output else None
			return out
		except:
			raise Exception('LAYER NOT BUILT')
      
  
	def storeOutput(self):
		self.store_output = True

	def copy(self):
		return Concat(copy.copy(self.index))

class Add(Concat):
	def __init__(self,index):
		# nn.Module.__init__(self)
		Concat.__init__(self,index)

	def build(self,prevLayer,model):
		self.layer = model.layers[self.index]
		self.layer.storeOutput()
		self.units = copy.copy(self.layer.units)
		self.built = True
		Add.Check(self.layer,prevLayer)

	def forward(self,X):
		out = self.layer.output + X
		self.output = out if self.store_output else None
		return out

	def copy(self,X):
		return Add(copy.copy(self.index))

	@staticmethod
	def Check(layer,layer2):
		if not layer.units == layer2.units:
			raise Exception('layer1 {} layer2 {} units not equal'.format(layer,layer2))
class Sequential():
  def __init__(self,layers=[]):
    self.layers = layers
    self.isCompiled= False
    self.Model = self.build()
    self.TrainLosses = []

  def build(self):
    layers = []
    try:
      if not self.layers[0].built:
        return 
    except:
      return

    for ind, layer in enumerate(self.layers):
      if isinstance(layer,Input):
        layers.append(layer)
        continue

      if isinstance(layer,Concat):
        layer.build(layers[-1], self)        
        layers.append(layer)
        continue

      kwargs = {} if layer.rawactivation is not nn.Softmax else {'dim':1}
      if layer.W is None:
        W = nn.Linear(self.layers[ind-1].units,layer.units)
        layer.W = W
        layers.append(layer)
        layer.activation = layer.activation(**kwargs)
      else:
        layers.append(layer)
        layer.activation = copy.deepcopy(layer.rawactivation(**kwargs))

    return nn.Sequential(*layers)

  def add(self,layer):
    self.layers.append(layer)
    self.Model = self.build()

  def forward(self,X):
    return self.Model(X)

  def parameters(self):
    return (weight for layer in self.Model
				if isinstance(layer,Dense) 
				for weight in [layer.W.weight,layer.W.bias])
  
  def fit(self,X,y,epochs,verbose=0):
    assert self.isCompiled
    for _ in range(epochs):
      pred = self.forward(X)
      l = self.loss(pred,y)
      l.backward()
      self.optim.step()
      self.optim.zero_grad()
      if verbose:
        print(_+1, l.detach(),end='\r')
      self.TrainLosses.append(l)
    print()  
    return self

  def __call__(self,X):
    return self.forward(X)

  def compile(self,optim,loss):
    self.optim = optim
    self.loss = loss
    self.isCompiled=True

  def copy(self):
    copiedLayers = [each.copy() for each in self.layers]
    s = Sequential(copiedLayers)
    s.isCompiled = copy.copy(self.isCompiled)
    return s
      
  def __repr__(self):
    return self.Model.__repr__()
  
if __name__ == '__main__':
  from sklearn.datasets import load_digits
  d = load_digits()
  X,y = d['data'],d['target'].reshape(-1)
  X,y = torch.tensor(X).float(),torch.tensor(y).long()
  model = Sequential([Input(64),
                    Dense(64,activation=nn.ReLU),
                    Dense(64,activation=nn.ReLU),
                    Dense(10,activation=nn.Softmax)])

  model2 = Sequential([Input(64),
                     Dense(128,activation=nn.ReLU),
                     Dense(64,activation=nn.ReLU),
                     Concat(1), ## (concat with Dense(128)-- layer 1)
                     Dense(10,activation=nn.Softmax)])

  model.compile(optim=torch.optim.Adam(model.parameters(),lr=1e-3),loss=nn.CrossEntropyLoss())
  model.fit(X,y,epochs=400,verbose=1)
  
  model2.compile(optim=torch.optim.Adam(model2.parameters(),lr=1e-3),loss=nn.CrossEntropyLoss())
  model2.fit(X,y,epochs=400,verbose=1)
    
