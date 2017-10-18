from time import time
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pdb
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import MiniBatchDictionaryLearning
from matplotlib import pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

def noisy_patches(image):
	image = image / 255.

	# downsample for higher speed
	image = image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]
	image /= 4.0
	height, width = image.shape

	# Distort the right half of the image
	print('Distorting image...')
	distorted = image.copy()
	#distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)
	distorted += 0.075 * np.random.randn(height, width)
	print(distorted.shape)
	cv2.imshow('noisy', distorted)
	# Extract all reference patches from the left half of the image
	print('Extracting reference patches...')
	t0 = time()
	patch_size = (7, 7)
	data = extract_patches_2d(distorted, patch_size)
	data = data.reshape(data.shape[0], -1)
	data -= np.mean(data, axis=0)
	data /= np.std(data, axis=0)
	print('done in %.2fs.' % (time() - t0))
	
	return (data)


# creating a sparse coded signal
# for documentaion 
# visit http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_coded_signal.html#sklearn.datasets.make_sparse_coded_signal

def sparse_signal(noisy_data, n_coeffs, learning_ratio):
	
	n_samples, n_features = noisy_data.shape # p X n
	n_components = int(learning_ratio*noisy_data.shape[0]) #k
	# x = D*alpha, D = initial_dict
	              # alpha = sparse_matrix
	x, init_dict, sparse_matrix = make_sparse_coded_signal(n_samples=n_samples,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_coeffs,
                                   random_state=0)
	indexes = np.random.random_integers(0, noisy_data.shape[1]-1, n_components)
	#pdb.set_trace()
	init_dict = noisy_data[indexes, :]
	

	return init_dict, sparse_matrix

def visualize_dict(V):
	patch_size = (7, 7)
	plt.figure(figsize=(4.2, 4))
	for i, comp in enumerate(V[:100]):
		plt.subplot(10, 10, i + 1)
		plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
		plt.xticks(())
		plt.yticks(())
	
	plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
	plt.show()

def ksvd(noisy_data, D, alpha):
	#def sparse_update():
		#omp = OrthogonalMatchingPursuit()
		
		#omp.fit(D.T, noisy_data.T)
		#return omp.coef_
	#sparse_rep = sparse_update()
	print(D.shape)
	print('Updating Dictionary')
	t0 = time()
	dico = MiniBatchDictionaryLearning(n_components=D.shape[0], 
										alpha=2, 
										n_iter=100, 
										dict_init=D, 
										transform_algorithm='omp')
	V = dico.fit(noisy_data)
	dt = time() - t0
	print('done in %.2fs.' % dt)

	return V.components_, dico.transform(noisy_data)

image = cv2.imread(args['image'], 0)
data = noisy_patches(image)

D, alpha = sparse_signal(data, 3, 0.1)
dict_final, sparse_rep = ksvd(data, D, alpha)
patches = np.dot(sparse_rep, dict_final)
patches += np.mean(patches, axis=0)
patches = patches.reshape(data.shape[0], *(7,7))
reconstruction = reconstruct_from_patches_2d(patches, (image.shape[0]//2, image.shape[1]//2))
cv2.imshow('reconstructed', reconstruction)



cv2.imshow('orignal', image)

cv2.waitKey(0)