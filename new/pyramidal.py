import numpy as np
import cv2
from new.PA2_2 import DownSample
from new.PA2_2 import UpSample
from new.PA2_2 import LucasKanade
from new.PA2_2 import LucasKanadeIterative
from PIL import Image

def LK_pyramidal(img1, img2, level):
	I1 = np.array(img1)
	I2 = np.array(img2)
	S = np.shape(I1)
	# finding the good features
	I1_smooth = cv2.GaussianBlur(I1, (3,3), 0)
	features = cv2.goodFeaturesToTrack(I1_smooth # Input image
	,10000 # max corners
	,0.01 # lambda 1 (quality)
	,10 # lambda 2 (quality)
	)
	q1 =[]
	feature = np.int0(features)
	for i in range(len(feature)):
		q1.append(([feature[i,0,0], feature[i,0,1]]))
	q2 = np.array(q1)
	pyramid1 = np.empty((S[0],S[1],level)) 
	pyramid2 = np.empty((S[0],S[1],level)) 
	pyramid1[:,:,0] = I1 			#since the lowest level is the original imae
	pyramid2[:,:,0] = I2 
	for j in range(1, level):
		I1 = DownSample(I1)
		I2 = DownSample(I2)
		pyramid1[0:np.shape(I1)[0], 0:np.shape(I1)[1], j] = I1
		pyramid2[0:np.shape(I2)[0], 0:np.shape(I2)[1], j] = I2
	(u,v) = LucasKanade(img1, img2)
	for i in range(level):
		level_I1 = pyramid1[0:int((len(pyramid1[:,0])/(2**(level-1-i)))),0:int((len(pyramid1[0,:])/(2**(level-1-i)))),level-1-i]
		level_I2 = pyramid2[0:int((len(pyramid2[:,0])/(2**(level-1-i)))),0:int((len(pyramid2[0,:])/(2**(level-1-i)))),level-1-i]
		if i == 0:
			for k in range(0, level):
				(u,v,r) = LucasKanadeIterative(level_I1, level_I2, u, v)
		else:
			(u,v,r) = LucasKanadeIterative(level_I1, level_I2, u, v)
		if i < level - 1:
			u = UpSample(u)
			v = UpSample(v)
		
	p = []
	for i,j in q2:
		#print(i,j)
		p.append([i+u[j,i],j+v[j,i]])
	p1 = np.array(p)
	return q2,p1
#Image1 = Image.open('basketball1.png').convert('L')
#Image2 = Image.open('basketball2.png').convert('L')
#LK_pyramidal(Image1, Image2, 3)
		