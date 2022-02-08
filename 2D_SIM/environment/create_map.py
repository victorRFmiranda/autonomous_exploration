import cv2
import numpy as np


def create_map(img, path):
# 	width, height = img.shape

	x = -100000
	y = -100000
	aux1 = []
	aux2 = []
	points = []
	count = 0
	for px in points[0]:
		for py in points[1]:
			if(py != y + 1):
				aux2 = y
				count += 1
			y = py

			


# 	f.close()









if __name__ == "__main__":
	# image_name = input("Image name with extension (example.png):")
	image_name = "ufmg_test.png"
	world_path = "./world/name.world"
	img = cv2.imread(image_name, 0)

	points = np.where(img == 0)


	# print(points.shape)
	print(points[0])

	print("comparar os indices atuais com os anteriores, se for diferrente pode ser o limite das retas")


	# print(img)
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	