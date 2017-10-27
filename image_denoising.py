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
from skimage.exposure import rescale_intensity


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-n_comp", "--n_components", type=int, default=100, help="number of componets in the dictionary")
ap.add_argument("-iter", "--n_iter", type=int, default=500, help="number of iterations")
ap.add_argument("-coeff", "--non_zero_coeff", type=int,default=1, help="number of non zero coefficients")
args = vars(ap.parse_args())

n_comp = args['n_components']
n_iter = args['n_iter']
non_zero_coeff = args['non_zero_coeff']
def noisy_patches(image, dict_learning=False, channel=None):
	image = image / 255.
	if dict_learning:
		#downsample for higher speed
		image = image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]
		image /= 4.0
	
	print('Distorting image...')
	distorted = image.copy()

	if channel :
		height, width, channel = image.shape
		distorted += 0.1 * np.random.randn(height, width, channel)
	else:
		height, width = image.shape
		distorted += 0.075 * np.random.randn(height, width)
	cv2.imwrite('noisy.jpg', (distorted*255))
	print(distorted.shape)

	print('Extracting reference patches...')
	t0 = time()
	patch_size = (7, 7)
	data = extract_patches_2d(distorted, patch_size)
	data = data.reshape(data.shape[0], -1)
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	data -= mean
	data /= std
	print('done in %.2fs.' % (time() - t0))
	
	return (data, mean, std)

def ksvd(noisy_data):
	
	print('Updating Dictionary')
	t0 = time()
	dico = MiniBatchDictionaryLearning(n_components=n_comp, 
										alpha=2, 
										n_iter=n_iter,)
										#dict_init=D)
	print('done in %.2fs.' % (time() - t0))
	V = dico.fit(noisy_data).components_
	return V, dico


if __name__ == '__main__':

	image = cv2.imread(args['image'])
	channel=None
	if len(image.shape) >2:
		channel = image.shape[2]
	data, _, _ = noisy_patches(image, dict_learning=True, channel=channel)
	dict_final, dico = ksvd(data)
	n0_data, mean, std= noisy_patches(image,channel=channel)
	dico.set_params(transform_algorithm='omp',transform_n_nonzero_coefs = non_zero_coeff )
	code = dico.transform(n0_data)
	patches = np.dot(code,dict_final)
	patches*= std
	patches += mean
	patches = (patches.reshape(n0_data.shape[0], 7, 7, channel))
	print('Reconstructing...')
	reconstruction = reconstruct_from_patches_2d(patches, (image.shape[0], image.shape[1], channel))
	reconstruction*=255
	difference = image - reconstruction
	error = np.sqrt(np.sum(difference ** 2))
	print('Difference (norm: %.2f)' %error)
	print('Finished reconstruction..')
	cv2.imshow('reconstructed', reconstruction.astype('uint8'))
	cv2.imshow('orignal', image)
	cv2.imwrite('orignal.jpg', image)
	cv2.imwrite('reconstructed.jpg', reconstruction.astype('uint8'))
	cv2.waitKey(0)