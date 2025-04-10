# Leaf Color Patterns Highlighted with Spectral Components Analysis
This repository contains the official implementation for Krishnamoorthi S et al. (under revision) in the STAR* Protocols.

The Python code is available in [krishnamoorthi_2024_star_protocol.py](./krishnamoorthi_2024_star_protocol.py) or in Jupyter notebook format as [krishnamoorthi_2024_star_protocol.ipynb](./krishnamoorthi_2024_star_protocol.ipynb). One sample hyperspectral image can be downloaded from [SpecimIQ_images/6725](./SpecimIQ_images), and other images are available in Figshare (https://figshare.com/s/83e7f0fef20bdb82169f). After downloading, unzip the file to access the images.

```bash
# To run the notebook, download krishnamoorthi_2024_star_protocol.ipynb and open it in Jupyter Notebook using:
jupyter notebook krishnamoorthi_2024_star_protocol.ipynb
```

## Summary
Leaf color patterns in nature exhibit remarkable diversity related to chemical properties and structural leaf features. Hyperspectral imaging captures such diverse color patterns with high spectral resolution. Hyperspectral image data are stored as 3D cubes with spatial (x, y) and spectral (λ) dimensions. Spectral component analysis is a powerful technique for extracting complex spectral patterns from leaf reflectance. By projecting hyperspectral images onto decomposed components, this method can reveal distinct color patterns and, in some cases, identify previously undetectable features on leaves. 
</br>
</br>
This protocol outlines the steps for correcting uneven lighting, identifying spectral components, and projecting hyperspectral cubes onto these components to highlight specific spectral features. Originally developed to analyze foliar color changes of model plants under nutrient stress (Krishnamoorthi S et al. (2024) Cell Reports [https://doi.org/10.1016/j.celrep.2024.114463]), this GitHub repository utilizes ornamental plants as alternative applications.
</br>
</br>
## Project Workflow
<img src="https://github.com/dr-daisuke-urano/Plant-Hyperspectral/blob/main/Figure2.jpg" alt="Alt text" width="35%">
Figure 1. The workflow of protocol: Please find Krishnamoorthi S (under review) STAR*Protocol for more details.
</br>
</br>

## Dependencies
To create a Conda environment with the dependencies used in Krishmoorthi S (under review) STAR\*Protocol, download environment.yml file and use the following command:

```bash
# Download environment.yml file and create new environment 
conda env create --name plant_hyperspectral --file environment.yml

# Open the Spyder IDE from the created environment
conda activate plant_hyperspectral
spyder
```

- python 3.12.3
- matplotlib 3.8.4
- numpy 1.26.4
- opencv-python 4.9.0.80
- pandas 2.2.1
- seaborn 0.11.2
- scikit-learn 1.5.0
- scipy 1.13.1
- spectral 0.23.1
- joblib 1.4.2

## Usage
## Data loading and Background masking
We used a white background to capture hyperspectral images of ornamental plants. In these images, the dominant spectral patterns typically come from green leaf pixels and white background pixels, which often appear as the top features in spectral component analysis. To improve the detection of spectral features relevant to leaf color, background pixels may be masked with the following steps. 

1.	Import the necessary libraries into the python environment 

```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
```

2.	Load hyperspectral data cube from the SPECIM IQ image directory in this GitHub repository. This protocol provides specim_loading function. Note: Download sample hyperspec images from https://figshare.com/s/83e7f0fef20bdb82169f and unzip the file. 
```python
folder = r'.\SpecimIQ_images'  # Specify folder location
img_ID = 6275   # Four sample images are provided; 6271, 6275, 6716 and 6788
hyperspectral_cube = specimIQ_loading(rf'{folder}\{img_ID}')

# Make a new folder to store result files
try:
    os.mkdir(rf'{folder}\{img_ID}_results')
except FileExistsError:
    print(f"The {img_ID}_results directory already exists.")
```

3.	For visualization, reconstruct RGB image from hyperspectral_cube with specimIQ_RGB function 
```python
original_RGB = specimIQ_RGB(hyperspectral_cube, gamma=1.5) 
cv2.imwrite(rf'{folder}\{img_ID}_results\original_RGB_{img_ID}.jpg', original_RGB[:,:,::-1])
plt.imshow(original_RGB)
plt.axis('off')
plt.show()
```

4.	Isolate and select leaf specific region. We provide SpecimIQ_background_masking function that masks white background, crops the masked region, and returns the cropped hyperspectral cube and the refined mask. 

The function requires the following parameters.
- hyperspectal_cube: 3D hyperspectral data cube of shape (x, y, λ).
- threshold_val: A float value that sets the threshold for background masking

```python
cropped_masked_cube, refined_mask = SpecimIQ_background_masking(hyperspectral_cube, threshold_val=1.0)
```

## Normalization of reflectance:
Hyperspectral images of leaves could show inconsistent reflectance, due to factors like leaf tilting, bending, or camera design issues (e.g., lens vignetting). These non-biological variations can be minimized through the following normalization step.

1.	To improve computational efficiency, reduce the number of pixels. Adjust pix_num as required.
```python
pix_num = 200
resized_masked_cube = resize(gradated_cube, (pix_num, pix_num, 204), mode='reflect', anti_aliasing=True)
```

2.	To normalize the pixel intensity across all wavelength channels, divide the reflectance value by the mean reflectance from the 875-925 nm bands, which typically exhibit high reflectance intensities regardless of species and environmental stresses.

```python
normalized_cube = resized_masked_cube / resized_masked_cube[:,:,170:175].mean(axis =2)[:,:, np.newaxis]
```

<img src="https://github.com/dr-daisuke-urano/Plant-Hyperspectral/blob/main/Figure3.jpg" alt="Alt text" width="60%">
Figure 2: Brightness adjustment of leaf reflectance spectra of Goeppertia makoyana hyperspectral images. 
(A) RGB images of the ventral and dorsal sides of G. makoyana leaves, with a 1 cm scale bar. (B, C) Reflectance patterns and pixel clustering of light-green and dark-green pixels. The graphs display the mean reflectance (solid line) and standard deviation (shaded regions). Data are shown from (top) the original hyperspectral image under uniform lighting, (middle) the same image with a manually applied brightness gradient (90% to 111%), and (bottom) the brightness-normalized image. 
</br>

## Data processing
After brightness normalization, the hyperspectral cube is ready for visualizing leaf spectral patterns, pixel clustering, and spectral component analysis.

### Pixel clustering 
1.	Import the hsi_pixel_clustering function to cluster leaf pixels based on reflectance spectrum. 
The function also visualizes leaf areas assigned to individual clusters and their spectral patterns. It returns a 2D array of cluster membership information and mean reflectance patterns of individual clusters. Clustering techniques available are KMeans, Gaussian Mixture Models (GMM), and Fuzzy C-Means (CMeans).

The function requires the following parameters.</br>
- cube: 3D hyperspectral data cube of shape (x, y, wavelength).</br>
- bands: 1D array the wavelengths corresponding to the hyperspectral bands.</br>
- method: Clustering algorithm to use. Choose from "KMeans", "GMM", or "CMeans".</br>
- num_clusters: Number of clusters.</br>
- path: Absolute path to the directory to save results.</br>

```python
membership, cluster_reflectance = hsi_pixel_clustering(cube, bands, num_clusters=3, method=’GMM’, path='output_directory')
```

</br>
<img src="https://github.com/dr-daisuke-urano/Plant-Hyperspectral/blob/main/Figure5.jpg" alt="Alt text" width="60%">
Figure 3: Leaf spectral patterns of clustered pixels in various plant species. Begonia aconitifolia, Aglaonema symphony, and Caladium thousand. (A-C) RGB images and corresponding pixel clusters (top) selected using the GMM clustering method. GMM was applied to hyperspectral cubes after background masking and brightness normalization. The graphs (bottom) display the mean reflectance (solid line) with standard deviations (shaded bands) for the pixel clusters identified by GMM.

## Spectral component analysis

1.	Import hsi_spec_comp_analysis function, which decomposes original hyperspectral images into a reduced number of components using methods such as NMF, SVD, FastICA, PCA, or SparsePCA. It returns both the fitted model and the projected data cube. 

The function requires the following parameters.
- cube: 3D hyperspectral data cube of shape (x, y, wavelength).
- bands: 1D array the wavelengths corresponding to the hyperspectral bands.
- dim: Number of components to retain.
- method: Decomposition method to use. Choose from ‘SVD', 'NMF', 'ICA', 'PCA', 'SparsePCA'.
- path: Absolute path to the directory to save results.

```python
model, projected_cube = hsi_spec_comp_analysis(cube, bands, dim=10, method='SVD', path='output_directory')
```

</br>
<img src="https://github.com/dr-daisuke-urano/Plant-Hyperspectral/blob/main/Figure4.jpg" alt="Alt text" width="60%">
Figure 4: Spectral component analysis of Hyperspectral Imaging Data. (A) Pseudo-coloured images of Aglaonema symphony obtained from the hyperspectral imaging. Colors represent pixel intensity values projected on ICA, SparcePCA, SVD and NMF components. (B) Line graphs show spectral features identified in top component, whichdescribe how the original data at each wavelength channel contribute to the top components. 

## Citation
Shalini Krishnamoorthi, Grace Zi Hao Tan, Yating Dong, Richalynn Leong, Ting-Ying Wu, Daisuke Urano (2024) [https://doi.org/10.1016/j.celrep.2024.114463].<br>
Shalini Krishnamoorthi, Daisuke Urano (under revision) STAR\*Protocols 
