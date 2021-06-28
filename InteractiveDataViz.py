import pygame,random,math,time,copy,sys, numpy as np
from pygame.locals import *
from Linalg import Vector, Matrix
from sklearn.datasets import load_breast_cancer,load_iris,load_diabetes

	


run = True
w,h = 1200,600
screen = pygame.display.set_mode((w,h))
clock = pygame.time.Clock()
frameRate = 120
deltaTime = 1/frameRate
tw,th = w//2,h//2
font = pygame.font.Font('freesansbold.ttf', 5) 


class PlotEngine():
	def __init__(self,cx,cz,cy,c=None,interactive=True):
		self.xlim = 200
		self.ylim = 200
		self.zlim = 200
		self.scale = 20
		self.lims = [self.xlim,-self.ylim,self.zlim]
		self.scaledPoints,self.alteredLims = self.scalePoints(cx,cy,cz)
		self.Interactive = Interactive() if interactive else None
		self.isInteractive = True if interactive else False
		self.y = c
		self.Colors = self.setColors(c)
		self.lines = self.GetLines()
		self.datapoints = list(zip(*self.scaledPoints))
		self.dataVecs = self.GetDataPoints()
		self.digits = self.GetDigits()

	@property
	def angX(self):
		return self.Interactive.angX
	@property
	def angY(self):
		return self.Interactive.angY

		

	def GetLines(self):
		lines = []
		for i,each in enumerate(self.lims):
			v = Vector(0,0,0)
			v.elems[i] = each
			proj(v)
			v = self.rotate(v)
			lines.append(v)
		return lines

	def rotate(self,v):
		if self.isInteractive:
			rotatedV = Vector(*v.elems).RotationX(self.angX,Vector(0,0,0))
			rotatedV = rotatedV.RotationY(self.angY,Vector(0,0,0))
			return rotatedV
		return v

	def scalePoints(self,*args):
		scaled = []
		alteredLims = []
		for each in [*args]:
			min_, max_ = each.min(), each.max()
			max_ = max_ if max_ != 0 else 1
			scaled.append((each/(max_)) * self.scale)
			alteredLims.append((min_,max_))

		return (scaled,alteredLims)

	def setColors(self,c):
		if c is None:
			return None

		cassignments = {}	
		uniques = np.unique(c)
		for each in uniques:
			cassignments[each] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		return cassignments



	def plot(self):
		if self.isInteractive:
			if self.Interactive.Update():
				self.lines = self.GetLines()
				self.digits = self.GetDigits()
				self.dataVecs = self.GetDataPoints()
# 		lines = self.GetLines()
		self.CheckEvent()
		self.drawLines()
		self.drawDigits()
		self.drawDataPoints()

	def GetDataPoints(self):
		dpoints = []
		for ind,each in enumerate(self.datapoints):
			v = Vector(*each)
			v.y *= -1
			v.z *= -1
			proj(v)
			v = self.rotate(v)
			color = (255,0,0) if self.Colors is None else self.Colors[self.y[ind]]
			dpoints.append((v,color))
		return dpoints

	def drawDataPoints(self):
		for each,color in self.dataVecs:
			pygame.draw.circle(screen,color,(tw+each.x*10,th+each.y*10),2)

			# try:
			# 	cur = each
			# 	nex = self.dataVecs[i+1]
			# 	pygame.draw.line(screen,(51,51,51),(tw+each.x*10,th+each.y*10),(tw+nex.x*10,th+nex.y*10))
			# except Exception as err:
			# 	pass



	def GetDigits(self):
		digs = []
		for i,(min_,max_) in enumerate(self.alteredLims):
			lim = 200
			sign = 1 if i == 0 else -1
			vals = np.linspace(-max_,max_,2*lim)[::self.scale]


			for ind,j in enumerate(range(-lim,lim,self.scale)):
				v = Vector(0,0,0)
				v.elems[i] = j*sign
				proj(v)
				v = self.rotate(v)
				digs.append((v,round((vals[ind]),1),i))

		return digs


	def drawDigits(self):
		for (v,j,i) in self.digits:
			text = font.render(str(j), True, (0,121,0)) 
			textRect = text.get_rect()  
			textRect.center = (tw+v.x+10*(i),th+v.y+10*(not i))
			screen.blit(text,textRect)




	def drawLines(self):
		for i, v in enumerate(self.lines):
			c = [0,0,0]
			c[i] = 121
			pygame.draw.line(screen,c,(tw-v.x,th-v.y),(tw+v.x,th+v.y),5)


	def CheckEvent(self):
		global run
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.QUIT
				run = False
				sys.exit()

			if self.isInteractive:
				self.Interactive.GetEvent(event)

		keys = pygame.key.get_pressed()
		return (keys,run)	



class Interactive():
	fl = 5
	def __init__(self):
		self.mouseIsDown = False
		self.origXPos,self.origYPos = (None,None)
		self.angX,self.angY = (0,0)
		self.NeedUpdate = False

	def GetEvent(self,event):
		if event.type == pygame.MOUSEBUTTONDOWN:
			if event.button == 4:
				Interactive.fl -= 6
				self.NeedUpdate = True

			elif event.button == 5:
				Interactive.fl += 6
				self.NeedUpdate = True

			else:
				self.origXPos,self.origYPos = pygame.mouse.get_pos()
				self.mouseIsDown = True

		elif event.type == pygame.MOUSEBUTTONUP:
			self.mouseIsDown = False

	def Update(self):
		if self.mouseIsDown:
			self.process()
			return True

		if self.NeedUpdate:
			self.NeedUpdate = False		
			return True

	def process(self):
		curx, cury = pygame.mouse.get_pos()

		self.angY += (curx - self.origXPos)/5
		self.angX += (cury - self.origYPos)/5

		self.origXPos, self.origYPos = pygame.mouse.get_pos()


def proj(vert):
	# fl = Interactive.fl
	# persp = fl/(fl+vert.z)
	# vert.x *= persp
	# vert.y *= persp
	disT = Interactive.fl
	vert.x *= (1-(disT-vert.z/25)/100)
	vert.y *= (1-(disT-vert.z/25)/100)
	vert.z *= (1-(disT-vert.z/25)/100)


def func():
	clock.tick(frameRate)
	screen.fill((0,0,0))
	plot.plot()
	pygame.display.update()

d = load_iris()
X = d['data']

X= X[:,:3]

# x = np.arange(-10,11)
# y = np.ones(21)
# c = np.round(np.random.randn(21)).astype(np.int32)
# print(c)
# z = (x)**4
y = d['target'].flatten()
plot = PlotEngine(X[:,0],X[:,1],X[:,2],c=y)


while run:
	func()
