import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import cv2
import numpy as np


def conv_block(input_size, output_size):
	block = nn.Sequential(
		nn.Conv2d(input_size, output_size, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(output_size), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
		nn.Conv2d(output_size, output_size, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(output_size), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
	)

	return block



class ConcatNetwork(nn.Module):
	def __init__(self):
		super(ConcatNetwork, self).__init__()
		# test convolution
		self.conv1 = conv_block(1, 16)
		self.conv2 = conv_block(16, 32)
		# self.conv3 = conv_block(32, 64)
		#self.ln1 = nn.Linear(64 * 58 * 58, 16)
		self.ln1 = nn.Linear(32 * 14 * 14, 16)
		self.relu = nn.ReLU()
		self.batchnorm = nn.BatchNorm1d(16)
		self.dropout = nn.Dropout2d(0.5)
		self.ln2 = nn.Linear(16, 1)

		self.p1 = nn.Linear(3, 2)
		self.p2 = nn.Linear(2, 2)
		self.p3 = nn.Linear(2, 1)

		self.f1 = nn.Linear(2, 4)
		self.f2 = nn.Linear(4, 4)
		self.f3 = nn.Linear(16, 1)

		self.last = nn.Linear(3,3)

	def forward(self, x):

		# conv image
		x = self.conv1(x)
		# x = self.conv2(x)
		# x = x.reshape(x.shape[0], -1)
		# x = self.ln1(x)
		# x = self.relu(x)
		# x = self.dropout(x)
		# x = self.ln2(x)
		# x = self.relu(x)


		# x = torch.cat((x[0], x[1], x[2]), dim=1)


		# x = self.last(x)

		return x





img = cv2.imread('map1.pgm', cv2.IMREAD_GRAYSCALE)
height, width = img.shape[:2]
# print("height = %d" % height)
# print("width = %d" % width)
img2 = cv2.resize(img, (64, 64))
img2 = img2/255.0
img2 = img2.astype('float32')

# Input to CNN
x = np.asarray([img2])

# Set to CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# convert to torch
x = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)

# CNN
c1 = conv_block(1,4).to(DEVICE)
cnn1 = c1(x)
print(cnn1.shape)
cnn1 = cnn1.view(cnn1.size(0), -1)
print(cnn1.shape)
lnn = nn.Sequential(nn.Linear(4 * 16 * 16, 32)).to(DEVICE)
lnn1 = lnn(cnn1)
print(lnn1.shape)


# # cnn2 = cnn2.reshape(cnn2.shape[0], -1)
# # print(cnn2.shape)
# ln1 = nn.Linear(13, 64).to(DEVICE)
# s1 = ln1(cnn1)
# relu1 = nn.ReLU(s1)
# print(s1.shape)
# # state2 = concat_network(state)







# cv2.imshow('image',img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# cv2.imshow('image2',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()