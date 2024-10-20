# Leaf Color Patterns Highlighted with Spectral Components Analysis
This is the official implementation for Krishnamoorthi S et al. (under review) STAR\*Protocol. 

## Summary
Leaf color patterns in nature exhibit remarkable diversity related to chemical properties and structural leaf features. Hyperspectral imaging captures such diverse color patterns with high spectral resolution. Hyperspectral image data are stored as 3D cubes with spatial (x, y) and spectral (λ) dimensions. Spectral component analysis is a powerful technique for extracting complex spectral patterns from leaf reflectance. By projecting hyperspectral images onto decomposed components, this method can reveal distinct color patterns and, in some cases, identify previously undetectable features on leaves. 
</br>
</br>
This protocol outlines the steps for correcting uneven lighting, identifying spectral components, and projecting hyperspectral cubes onto these components to highlight specific spectral features. Originally developed to analyze foliar color changes in Marchantia polymorpha under nutrient stress (Krishnamoorthi S et al. (2024) Cell Reports [https://doi.org/10.1016/j.celrep.2024.114463]), this GitHub repository utilizes ornamental plants as alternative applications.
</br>
</br>
## Project Workflow
<img src="https://github.com/dr-daisuke-urano/Plant-Hyperspectral/blob/main/Figure2.jpg" alt="Alt text" width="35%">
Workflow of the Protocol. Please find Krishnamoorthi S (under review) STAR*Protocol for image acquisition and details of the method.
</br>
</br>

## Dependencies
To create a Conda environment with the dependencies used in Krishmoorthi S (under review) STAR\*Protocol, download environment.yml file and use the following command:

```bash
conda env create --name Plant-Hyperspectral --file environment.yml
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
We used a white background to capture hyperspectral images of ornamental plants. In these images, the dominant spectral patterns typically come from green leaf pixels and white background pixels, which often appear as the top features in spectral component analysis. To improve the detection of spectral features relevant to leaf color patterns, background pixels can be masked with the following steps. 

1.	Import the necessary libraries into the python environment 

Note: Version information for Python and its packages is provided in the Resource Table.
```python
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
```

2.	Load hyperspectral data cube from the SPECIM IQ image directory. This protocol provides specim_loading function. 
```python
folder = rf'path-to-directory ’
img_ID = 6766 #Replace with hyperspec data folder
hyperspectral_cube=specim_loading(rf'path-to-directory/{folder}’)
```

3.	For visualization, reconstruct RGB image from hyperspectral_cube with specimIQ_RGB function 
```python
original_RGB = specimIQ_RGB(hyperspectral_cube, gamma=1.5) 
cv2.imwrite('file_to_write.jpg', original_RGB[:,:,::-1])
plt.imshow(original_RGB)
plt.axis('off')
plt.show()
```

4.	To isolate and select leaf specific region, apply masking technique:

&nbsp; &nbsp; a.	Highlight the plant pixel by multiplying pre-defined plant reference spectrum with hyperspectral data. 
Note: This reference spectrum that effectively distinguishes leaf pixels from the white background. Any commonly used masking methods can be used instead.
```python
reference_pic = np.dot(hyperspectral_cube[:, :,10:200], plant_reference_spectrum)
```

&nbsp; &nbsp; b.	To create a binary mask for easy segmentation, threshold the resulting image that highlights leaf and non-leaf regions, and refine the mask by applying erosion to smooth the boundaries
```python
_, mask = cv2.threshold(reference_pic, 1, 1, cv2.THRESH_BINARY_INV)
mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
```

### Visualize the masked area
```python
plt.imshow(mask, cmap='gray')
plt.axis('OFF')
plt.show()
```

Critical: The threshold value for masking the white background can be adjusted as needed for individual cases. It significantly affects the quality of the segmentation.

&nbsp; &nbsp; c.	Identify contours and refine the masked area that encloses the leaf pixels
```python
contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)
refined_mask = np.zeros_like(mask)
cv2.drawContours(refined_mask, [contour], -1, (1), thickness=cv2.FILLED)
```

### Visualize the refined masked area
```python
plt.imshow(refined_mask, cmap='gray')
plt.axis('OFF')
plt.show()
```
 
&nbsp; &nbsp; d.	Crop the image to focus on the region of interest using the contour 
```python
x, y, w, h = cv2.boundingRect(contour)
l = max(w, h)
masked_cube = hyperspectral_cube * refined_mask[:,:,np.newaxis]
masked_cube[masked_cube == 0] = 'nan'
cropped_masked_cube = np.full((l, l, masked_cube.shape[2]), np.nan)
cropped_masked_cube[(l-h)//2:h+(l-h)//2, (l-w)//2:w+(l-w)//2, :] = masked_cube[y:y+h,x:x+w,:]
print(f'height:{h}, width:{w}')
```

## Normalization of reflectance:
Hyperspectral images of leaves could show inconsistent reflectance, due to factors like leaf tilting, bending, or camera design issues (e.g., lens vignetting). These non-biological variations can be minimized through the following normalization step.

1.	To improve computational efficiency, reduce the number of pixels by resizing the data cube. Adjust pix_num as required.
```python
pix_num = 200
resized_masked_cube = resize(cropped_masked_cube, (pix_num, pix_num, 204), mode='reflect', anti_aliasing=True)
```

2.	To normalize the pixel intensity across all wavelength channels, divide the reflectance value by the mean reflectance from the 875-925 nm bands, which typically exhibit high reflectance intensities regardless of species and environmental stresses1 .

Note: Figures 3B illustrates the spectral patterns before and after normalization. The normalization process reduced variability, minimizing non-biological influences and enhancing the detection of leaf color patterns, as shown by the pixel clustering results in Figures 3C.
```python
normalized_cube = resized_masked_cube / resized_masked_cube[:,:,170:175].mean(axis =2)[:,:, np.newaxis]
```

## Data processing
After background masking and normalization, the hyperspectral cube is prepared for visualizing leaf spectral patterns, extracting pixel groups with similar patterns using clustering algorithms (Figure 3C), and identifying spectral features through spectral component analysis (Figure 4B). 

### Pixel clustering 
1.	Import the hsi_pixel_clustering function to cluster leaf pixels based on reflectance spectrum. 
The function also visualizes leaf areas assigned to individual clusters and their spectral patterns. It returns a 2D array of cluster membership information and mean reflectance patterns of individual clusters. Clustering techniques available are KMeans, Gaussian Mixture Models (GMM), and Fuzzy C-Means (CMeans).

The function requires the following parameters.
cube: 3D hyperspectral data cube of shape (x, y, wavelength).
bands: 1D array the wavelengths corresponding to the hyperspectral bands.
method: Clustering algorithm to use. Choose from "KMeans", "GMM", or "CMeans".
num_clusters: Number of clusters.
path: Absolute path to the directory to save results. 

```python
membership, cluster_reflectance = hsi_pixel_clustering(cube, bands, num_clusters=3, method=’GMM’, path='output_directory')
```

2.	The below step by step process explains the hsi_pixel_clustering function workflow
a.	Reshape the 3D hyperspectral cube into a 2D array where each row corresponds to a pixel and each column corresponds to a wavelength.
```python
x, y, wl = cube.shape
reshaped_cube = cube.reshape(x * y, wl)
```

b.	Select the clustering method and perform pixel clusterin:
i.	Kmeans: sklearn.cluster.KMeans
K-Means is a hard-clustering algorithm that defines cluster centroids and assign each pixel to exactly one cluster. It partitions pixels into distinct groups based on their spectral characteristics by minimizing the variance within each cluster4. The algorithm classifies pixels into a predefined number of clusters (K) set by the user and assigns each pixel to the nearest cluster centroid, which presents the average spectral signature of the pixels within the cluster.

ii.	CMeans: skfuzzy.cluster.skfuzzy
The Fuzzy C-Means algorithm is a soft-clustering method that assigns pixels to multiple clusters with different degrees of membership. Rather than placing each pixel exclusively in a single cluster, C-Means evaluates the pixel's spectral features and computes how closely it aligns with each cluster centroid. The proximity of a pixel's spectral profile to a centroid determines its membership strength in that cluster7. In Figure 3B, C-Means method is utilized to distinguish light- and dark-green pixels based on their spectral characteristics.

iii.	GMM: sklearn.mixture.GuassianMixture
GMM (Gaussian Mixture Model) is a soft-clustering method that classifies each pixel in hyperspectral images into several clusters based on probabilities, rather than assigning each pixel to exactly one cluster5. This approach is particularly useful when spectral signatures overlap or when there is uncertainty about which cluster a pixel belongs to. GMM provides a probabilistic framework, allowing for a more detailed understanding of the spectral data by reflecting the likelihood of each pixel belonging to multiple clusters. Below shows an example usage when ‘GMM’ is selected.  

```python
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=num_clusters, max_iter=1000, covariance_type='full').fit(non_nan_reshaped_cube)
        non_nan_cluster_membership = model.predict(non_nan_reshaped_cube)
```

c.	Recreate cluster_membership by assigning np.nan to masked pixels
```python
cluster_membership = np.full(x * y, np.nan)
cluster_membership[non_nan_pixels] = non_nan_cluster_membership
```

d.	Calculate the mean reflectance and standard deviation for each cluster and all non-NaN pixels
```python
mean_reflectance = np.zeros((num_clusters + 1, wl))
std_reflectance = np.zeros((num_clusters + 1, wl))

for n in range(num_clusters):
        cluster_pixels = reshaped_cube[cluster_membership == n]
        mean_reflectance[n, :] = np.nanmean(cluster_pixels, axis=0)
        std_reflectance[n, :] = np.nanstd(cluster_pixels, axis=0)

# For the "all non-NaN pixels" group
mean_reflectance[num_clusters,:] = np.nanmean(non_nan_reshaped_cube, axis=0)
std_reflectance[num_clusters,:] = np.nanstd(non_nan_reshaped_cube, axis=0)
```

e.	Generate plots to visualize the results, displaying the mean reflectance with standard deviation bands and the RGB image showing spatial projections for each cluster.
```python
for n in range(num_clusters + 1):
    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

     # Plotting the mean reflectance with std bands for each cluster
     if n < num_clusters:
         ax1.set_title(f'Mean Reflectance for Cluster {n + 1}')
         ax1.plot(bands, mean_reflectance[n, :], label=f'Cluster {n + 1}')
         mask = cluster_membership.reshape(x, y) == n
         ax2.set_title(f'{method} Cluster {n + 1}')
         ax2.imshow(RGB * mask[:, :, np.newaxis])

     # Plotting them for all non-Nan pixels
     else:
         ax1.set_title('Mean Reflectance for All Pixels')
         ax1.plot(bands, mean_reflectance[n, :], label='All Pixels')
         ax2.set_title(f'{method} All Pixels')
         ax2.imshow(RGB)

    # Fill std deviation band and add the x and y labels
    ax1.fill_between(bands, mean_reflectance[n, :] - std_reflectance[n, :], 
                   mean_reflectance[n, :] + std_reflectance[n, :], alpha=0.3, label='St. Dev.')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.legend()
    ax2.axis('off')

    # Save figure if path is provided
    if path is not None:
        if n < num_clusters:
            fig.savefig(fr'{path}/{method}_Cluster_{n + 1}.pdf')
        else:
            fig.savefig(fr'{path}/{method}_All_Pixels.pdf')
        
    plt.show()
```

f.	Return the cluster_membership and the mean_reflectance
```python
return cluster_membership, mean_reflectance
```
## Citation
Shalini Krishnamoorthi, Grace Zi Hao Tan, Yating Dong, Richalynn Leong, Ting-Ying Wu, Daisuke Urano (2024) [https://doi.org/10.1016/j.celrep.2024.114463].<br>
Shalini Krishnamoorthi, Daisuke Urano (under review) STAR\*Protocol 
