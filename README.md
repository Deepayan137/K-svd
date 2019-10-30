## Runnnig the code
``` bash
python image_denoising.py -i imagename -iter 500 -coeff 2 -n_comp 100
```
## optional parameters

1. -iter: number of iterations required to learn the dictionary
2. -n_comp: number of components that the dictionary should contain
3. -coeff: number of non zero coefficients in the sparse representation of the image

## output

The output will give you 3 images, Orignal, Noisy and Reconstructed

## Results: Deep

![The above picture shows the distortion of the image Lena and its subsequent reconstruction using dictionary learning](Images/denoising.png)

The above picture shows the distortion of the image Lena and its subsequent reconstruction using dictionary learning
(a) shows the orignal image (b) shows the image after adding gaussian noise (c) Reconstructed Image



