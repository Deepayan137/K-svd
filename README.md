# K-svd
Image Denoising via Sparse and Redundant Representations over Learned Dictionaries


## Steps to follow for K- svd implementation
- [ ] add a fucnction which will take an image and add gaussian noise to it.
	* your function should take an clean image and a parameter say sigma and it should return a noisy image. the parameter sigma 
	will be used to vary the amount of noise to be added.
- [x] extact patches from the image
	* function should take a noisy image and dimension of patches say n and it should return a numpy matrix of dimension say(p, n*n)
- [ ] do sparse coding
	* function should initialize a dictionary and do a sparse coding. the function should return a matrix which contains sparse vectors of of the image patches. The dimension of the sparse matrix should be p X N
- [ ] update dictionary
	* this function should implement the dictionary update using the K- svd algorithm and return an updated dictionary of size 
	(N X p)

## Note 
Since this is a collaborative work, please follow the guidelines so that there is no library dependecy issues in the future.
 
* We will be using python 3.6 for our work
* Please follow pep-8 style guide. Google it and follow the instructions
* Use 4 spaces for indentation
* Add comments wherever necessary. Make your code as redable as possible.
* pull this github repo before making any commit otherwise there can be a version error.
