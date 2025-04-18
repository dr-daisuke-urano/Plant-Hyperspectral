# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:50:05 2024

@author: daisuke
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

'''
specify the font and fontsize for saving in PDF format
'''
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10
#%matplotlib inline

# Specim IQ camera stores hyperspectral images at 204 wavelength channels, as below 
specimIQ_wavelength = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56,
446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80,
501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62,
548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70,
595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04,
643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53,
738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67,
786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07,
835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74,
883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88,
981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58
]

def specimIQ_loading(path):
    """
    Load a hyperspectral data cube from the SPECIM IQ image directory.

    Parameters:
    path (str): The absolute path to the directory containing .hdr and .raw files.

    Returns:
    np.ndarray: The calibrated hyperspectral data cube.
    """
    
    import spectral as spy
    import numpy as np
    
    # Extract the directory name as the ID
    ID = path.split('\\')[-1]
 
    try:
        # Open the hyperspectral data files
        data_ref = spy.io.envi.open(f'{path}\\capture\\{ID}.hdr', f'{path}\\capture\\{ID}.raw')
        white_ref = spy.io.envi.open(f'{path}\\capture\\WHITEREF_{ID}.hdr', f'{path}\\capture\\WHITEREF_{ID}.raw')
        dark_ref = spy.io.envi.open(f'{path}\\capture\\DARKREF_{ID}.hdr', f'{path}\\capture\\DARKREF_{ID}.raw')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")

    # Perform calibration by subtracting dark reference and dividing by the white reference
    cube = (np.array(data_ref.load()) - np.array(dark_ref.load())) / (np.array(white_ref.load()) - np.array(dark_ref.load()))
    
    return cube

def specimIQ_RGB(cube, gamma=1.0):    
    import numpy as np

    # R, G, and B reference reflectance values from the Specim IQ wavelength channels 6 to 98
    # These reference values were generated using the ColorChecker Classic chart image.
    B_reference = np.array([0.28462822, 0.28685858, 0.28758628, 0.29093164, 0.29571122,
           0.30220778, 0.30079957, 0.27963749, 0.29555491, 0.31906699,
           0.32516116, 0.32451853, 0.32559147, 0.32121282, 0.31593104,
           0.31127376, 0.30159306, 0.29403532, 0.28518392, 0.27803837,
           0.26438833, 0.25220742, 0.23785448, 0.22080147, 0.20234738,
           0.18885382, 0.1813013 , 0.17122129, 0.16100587, 0.1516296 ,
           0.14161461, 0.13246509, 0.12350839, 0.11398222, 0.1053238 ,
           0.09502631, 0.08695949, 0.0798408 , 0.07264553, 0.06786152,
           0.06351456, 0.0600972 , 0.05584277, 0.04966747, 0.04722196,
           0.04584859, 0.04574296, 0.04914031, 0.04966885, 0.0486881 ,
           0.0484957 , 0.04782599, 0.04706948, 0.04643554, 0.04711054,
           0.0459301 , 0.04562978, 0.04483982, 0.04478453, 0.04492691,
           0.04528737, 0.04501297, 0.044905  , 0.04501652, 0.0449133 ,
           0.04555969, 0.04434773, 0.03968953, 0.03880938, 0.0427795 ,
           0.04565767, 0.04564508, 0.04515549, 0.04575521, 0.04612792,
           0.04700006, 0.04855985, 0.04939982, 0.04975982, 0.05118658,
           0.05134406, 0.05192206, 0.0528893 , 0.05424882, 0.05466969,
           0.05568517, 0.05646897, 0.05624799, 0.05528999, 0.05502877,
           0.05431273, 0.0536918 , 0.05313713])
    
    G_reference = np.array([0.21133903, 0.19229854, 0.18078605, 0.16417557, 0.15436052,
           0.14840434, 0.12556966, 0.09313943, 0.0970141 , 0.12014251,
           0.11981628, 0.11819124, 0.12000915, 0.11835837, 0.1193578 ,
           0.12030675, 0.1218589 , 0.12349002, 0.12709269, 0.13230053,
           0.13579399, 0.14191656, 0.14937976, 0.15476981, 0.15940244,
           0.16737199, 0.18270389, 0.2020068 , 0.22317632, 0.24598215,
           0.27072405, 0.29505249, 0.31961973, 0.34010537, 0.35724534,
           0.36693443, 0.3737123 , 0.37546906, 0.37373522, 0.37074009,
           0.3664101 , 0.35928211, 0.3499074 , 0.33562034, 0.32473127,
           0.31567165, 0.31262132, 0.30839412, 0.29890602, 0.28856173,
           0.27746994, 0.26626604, 0.25537294, 0.24514435, 0.2363445 ,
           0.22501424, 0.21382301, 0.20501528, 0.19609266, 0.18763979,
           0.17928106, 0.17153051, 0.16436592, 0.15626209, 0.1498779 ,
           0.1446759 , 0.13768314, 0.12677298, 0.12379434, 0.12577324,
           0.12516931, 0.12377166, 0.12184466, 0.12027606, 0.11963645,
           0.11984991, 0.11964515, 0.11898388, 0.11754042, 0.11713352,
           0.11657382, 0.11545898, 0.11547345, 0.11564586, 0.11568949,
           0.11692526, 0.11748922, 0.11804433, 0.11938791, 0.12119119,
           0.12227899, 0.1234448 , 0.12521574])
    
    R_reference = np.array([0.19379772, 0.18053199, 0.16449151, 0.14728372, 0.13217046,
           0.12471709, 0.09912372, 0.06362731, 0.06501805, 0.08884007,
           0.0872441 , 0.0855771 , 0.0840513 , 0.0828051 , 0.08234076,
           0.08084493, 0.08133215, 0.08087183, 0.08254146, 0.08292572,
           0.08340879, 0.08386889, 0.08278485, 0.08019788, 0.07456487,
           0.06964235, 0.07022374, 0.0718186 , 0.07329241, 0.07471453,
           0.07527669, 0.07545765, 0.07577908, 0.07413419, 0.07420735,
           0.07298076, 0.07303222, 0.07371355, 0.07331879, 0.07373113,
           0.07325049, 0.07174468, 0.06877615, 0.06185071, 0.05729419,
           0.05601269, 0.05861221, 0.06474009, 0.06926641, 0.07156813,
           0.07453253, 0.07746038, 0.07975167, 0.08327926, 0.08758171,
           0.08938982, 0.09140255, 0.09479274, 0.09979963, 0.10750622,
           0.11896181, 0.13528612, 0.15544309, 0.18053507, 0.20842546,
           0.24427374, 0.28137826, 0.31276189, 0.34233328, 0.4002486 ,
           0.45385175, 0.49514783, 0.53280182, 0.56352838, 0.58944713,
           0.61350888, 0.63375058, 0.64685355, 0.65512809, 0.66278691,
           0.66776043, 0.67205826, 0.67632037, 0.68149331, 0.68275553,
           0.68573297, 0.68618982, 0.68956697, 0.68951365, 0.69225473,
           0.69300172, 0.69386459, 0.69621948])
        
    R = (cube[:,:,6:99] * R_reference[np.newaxis, np.newaxis, :]).sum(axis = 2)
    G = (cube[:,:,6:99] * G_reference[np.newaxis, np.newaxis, :]).sum(axis = 2)
    B = (cube[:,:,6:99] * B_reference[np.newaxis, np.newaxis, :]).sum(axis = 2)

    # Normalize by 90th percentile to enhance contrast, and convert RGB image to Uint8 format 
    RGB = np.stack([R/np.percentile(R,90), G/np.percentile(G, 90), B/np.percentile(B, 90)], axis=2)
    RGB = np.clip(RGB, 0, 1)
    RGB = (235*RGB).astype(np.uint8)

    # Gamma correction
    gamma_table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(RGB, gamma_table)
    
    return gamma_corrected

def SpecimIQ_background_masking(hyperspectral_cube, threshold_val = 1.0):
    # A representative leaf reflectance generated with SVD. SPECIM IQ wavelength channels [10:200]
    plant_reference_spectrum = np.array([0.04069669,  0.04079603,  0.04126037,  0.04213693,  0.04273207,
            0.04379434,  0.04518137,  0.04713085,  0.04874695,  0.05029235,
            0.0516203 ,  0.05295029,  0.05294916,  0.05332283,  0.05327264,
            0.05344182,  0.05390424,  0.05425218,  0.0543171 ,  0.05522408,
            0.05585662,  0.05727984,  0.05921757,  0.06182781,  0.06514108,
            0.0688416 ,  0.07287744,  0.07792781,  0.08274431,  0.08712468,
            0.09143544,  0.09475504,  0.09722056,  0.09856757,  0.0991911 ,
            0.09913175,  0.09923179,  0.09917054,  0.09974652,  0.10164138,
            0.10276858,  0.1031924 ,  0.1027126 ,  0.10187102,  0.10124572,
            0.10113701,  0.10179041,  0.10246365,  0.10329962,  0.10449796,
            0.10565817,  0.10722926,  0.10852164,  0.10937907,  0.11002037,
            0.11090264,  0.11098943,  0.11108233,  0.11116369,  0.11111724,
            0.11098415,  0.11097883,  0.11096041,  0.11322217,  0.11394354,
            0.11286716,  0.11229089,  0.11243878,  0.11269937,  0.11351002,
            0.11409797,  0.11430819,  0.1129232 ,  0.11037677,  0.10717013,
            0.10334962,  0.10013911,  0.09814503,  0.09737767,  0.09646666,
            0.09434346,  0.09063417,  0.08613345,  0.08249818,  0.07965832,
            0.07827442,  0.0788685 ,  0.08293296,  0.09276   ,  0.10782823,
            0.12418226,  0.13724419,  0.14330781,  0.14223865,  0.13543469,
            0.12344085,  0.10910331,  0.09239325,  0.07515585,  0.05786373,
            0.04189477,  0.02756335,  0.01521532,  0.00479685, -0.00368386,
           -0.01059016, -0.01556718, -0.01969922, -0.02287875, -0.02538634,
           -0.02725541, -0.02868451, -0.0298151 , -0.03078676, -0.03144858,
           -0.03190495, -0.03239354, -0.03274344, -0.03315384, -0.0335644 ,
           -0.03404963, -0.03421362, -0.03422848, -0.03380797, -0.03416794,
           -0.03555574, -0.03529606, -0.03537379, -0.03648132, -0.03648571,
           -0.03692753, -0.03719202, -0.03735861, -0.037777  , -0.03793817,
           -0.03801186, -0.0384836 , -0.03849779, -0.03870448, -0.03879738,
           -0.03916825, -0.03950635, -0.039951  , -0.04001093, -0.04007773,
           -0.04053165, -0.04100606, -0.04129533, -0.04136556, -0.0415791 ,
           -0.04163111, -0.04166479, -0.04189004, -0.04211083, -0.04238162,
           -0.04291235, -0.04348892, -0.04388314, -0.04394482, -0.04384172,
           -0.04397907, -0.0437822 , -0.04379905, -0.04419531, -0.04406442,
           -0.04378627, -0.04406204, -0.04387016, -0.04338229, -0.04324095,
           -0.04314747, -0.04305496, -0.04254075, -0.04202947, -0.04179743,
           -0.04191031, -0.04088917, -0.0412918 , -0.04073383, -0.04114633,
           -0.04026839, -0.0399152 , -0.04082155, -0.04049295, -0.04036159,
           -0.03976451, -0.0397357 , -0.04025013, -0.0402305 , -0.03912869])

    #Apply the representative leaf reflectance to hyperspectral cube for masking background
    reference_pic = np.dot(hyperspectral_cube[:, :, 10:200], plant_reference_spectrum)

    # Threshold the SVD image to create a mask
    _, mask = cv2.threshold(reference_pic, threshold_val, 1, cv2.THRESH_BINARY_INV)

    # Morphological operations are applied to make the mask smooth and eroded slightly. 
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) 

    # Visualize the original masked area
    plt.imshow(mask, cmap='gray')
    plt.axis('OFF')
    plt.show()

    #Identify contours and refine the masked area that encloses the leaf pixels.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    refined_mask = np.zeros_like(mask)
    cv2.drawContours(refined_mask, [contour], -1, (1), thickness=cv2.FILLED)

    # Visualize the refined masked area
    plt.imshow(refined_mask, cmap='gray')
    plt.axis('OFF')
    plt.show()

    #Crop the masked region using the largest contour
    x, y, w, h = cv2.boundingRect(contour)
    l = max(w, h)
    masked_cube = hyperspectral_cube * refined_mask[:,:,np.newaxis]
    masked_cube[masked_cube == 0] = 'nan'
    cropped_masked_cube = np.full((l, l, masked_cube.shape[2]), np.nan)
    cropped_masked_cube[(l-h)//2:h+(l-h)//2, (l-w)//2:w+(l-w)//2, :] = masked_cube[y:y+h,x:x+w,:]
    print(f'height:{h}, width:{w}')
    
    return cropped_masked_cube, refined_mask

def hsi_pixel_clustering(cube, bands=specimIQ_wavelength, num_clusters=10, path=None, method='GMM'):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    # Create and save RGB image
    RGB = specimIQ_RGB(np.nan_to_num(cube, nan=0), gamma=0.7)
    if path is not None:
        cv2.imwrite(rf'{path}/{method}_originalRGB.jpg', RGB[:,:,::-1])

    # Reshape the hyperspectral cube to 2 dims (pixels x wavelength)
    x, y, wl = cube.shape
    reshaped_cube = cube.reshape(x * y, wl)

    # Remove NaN pixels
    non_nan_pixels = ~np.all(np.isnan(reshaped_cube), axis=1)
    non_nan_reshaped_cube = reshaped_cube[non_nan_pixels]

    # Perform pixel clustering based on the chosen method
    if method == 'KMeans':
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=num_clusters, random_state=0).fit(non_nan_reshaped_cube)
        non_nan_cluster_membership = model.labels_
    elif method == 'CMeans':
        from skfuzzy.cluster import cmeans
        cntr, u, u0, d, jm, p, fpc = cmeans(non_nan_reshaped_cube.T, num_clusters, 2, error=0.005, maxiter=1000, init=None)
        non_nan_cluster_membership = np.argmax(u, axis=0)
    elif method == 'GMM':
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=num_clusters, max_iter=1000, covariance_type='full').fit(non_nan_reshaped_cube)
        non_nan_cluster_membership = model.predict(non_nan_reshaped_cube)
    else:
        raise ValueError("Unsupported method. Choose from 'KMeans', 'CMeans', or 'GMM'")

    # Recreate cluster_membership by assigning np.nan to masked pixels
    cluster_membership = np.full(x * y, np.nan)
    cluster_membership[non_nan_pixels] = non_nan_cluster_membership

    # Calculate the mean reflectance and standard deviation for each cluster and all pixels
    mean_reflectance = np.zeros((num_clusters + 1, wl))
    std_reflectance = np.zeros((num_clusters + 1, wl))

    for n in range(num_clusters):
        cluster_pixels = reshaped_cube[cluster_membership == n]
        mean_reflectance[n, :] = np.nanmean(cluster_pixels, axis=0)
        std_reflectance[n, :] = np.nanstd(cluster_pixels, axis=0)

    # For the "all non-Nan pixels" group
    mean_reflectance[num_clusters, :] = np.nanmean(non_nan_reshaped_cube, axis=0)
    std_reflectance[num_clusters, :] = np.nanstd(non_nan_reshaped_cube, axis=0)

    # Visualize each cluster and "All pixels"
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
        else:
            # For "All pixels"
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

    return cluster_membership, mean_reflectance

def hsi_spec_comp_analysis(cube, bands=specimIQ_wavelength, dim=10, path=None, method='SVD'):
    # import necessary libraries
    import numpy as np
    import sklearn.decomposition as skd
    import matplotlib.pyplot as plt
    
    # Reshape the hyperspectral cube to 2 dims (pixels x wavelength)
    x, y, wl = cube.shape
    reshaped_cube = cube.reshape(x * y, wl)
    
    # Remove rows where all the values across wavelengths are NaN
    non_nan_pixels = ~np.all(np.isnan(reshaped_cube), axis=1)
    non_nan_reshaped_cube = reshaped_cube[non_nan_pixels]
    
    # Print the shapes before and after NaN removal
    print(f"Original shape: {reshaped_cube.shape}, Shape after removing NaNs: {non_nan_reshaped_cube.shape}")
    
    # Select, initialize, fit the decomposition model
    if method == 'SVD':
        model = skd.TruncatedSVD(n_components=dim, random_state=0, n_iter=100)
    elif method == 'NMF':
        model = skd.NMF(n_components=dim, random_state=0, max_iter=5000)
    elif method == 'ICA':
        model = skd.FastICA(n_components=dim, random_state=0, max_iter=5000)        
    elif method == 'PCA':
        model = skd.PCA(n_components=dim, random_state=0)
    elif method == 'SparsePCA':
        model = skd.SparsePCA(n_components=dim, random_state=0, max_iter=5000)
    else:
        raise ValueError("Unsupported method. Choose from 'SVD', 'NMF', 'ICA', 'PCA' or 'SparsePCA'.")

    hsi_model = model.fit_transform(non_nan_reshaped_cube)

    # Create an empty array to restore the transformed data
    hsi_model_full = np.full((x * y, dim), np.nan)
    hsi_model_full[non_nan_pixels] = hsi_model
    
    # Reshape the 2 dim data back into 3 dim images (x, y, components)
    projected_cube = hsi_model_full.reshape(x, y, dim)
    
    # Visualize and save the results        
    for n in range (dim):
        fig = plt.figure(figsize=(8,4))
        if method in ['SVD', 'PCA']:
            fig.suptitle(f'{method} component {n}, Explained Variance Ratio: {model.explained_variance_ratio_[n]:.3f}')
        else:
            fig.suptitle(f'{method} component {n}')
            
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(bands, model.components_[n])
        ax1.set_xlabel('Wavelength[nm]')
        ax1.set_ylabel('Reflectance')
        cmap = matplotlib.colormaps.get_cmap('gist_earth')
        cmap.set_bad(color='black')
        ax2.imshow(projected_cube[:,:,n], cmap=cmap)
        ax2.axis('off') 
        if not path is None:
            fig.savefig(fr'{path}/{method}_component_{n}.pdf')
        plt.show()
    
    # Return the model and the projected hyperspectral cube
    return model, projected_cube

#%%
'''
Load sample hyperspectral image
'''
folder = r'.\SpecimIQ_images' # Modify the path to the folder containing unzipped hyperspectral image folders (e.g., C:\Users\Admin\Downloads)
img_ID = 6275   # Four sample images are provided in Figshare (https://figshare.com/s/83e7f0fef20bdb82169f); 6271, 6275, 6716 and 6788
hyperspectral_cube = specimIQ_loading(rf'{folder}\{img_ID}')
try:
    os.mkdir(rf'{folder}\{img_ID}_results')
except FileExistsError:
    print(f"The {img_ID}_results directory already exists.")

'''
Show RGB image of the cube
'''
original_RGB = specimIQ_RGB(hyperspectral_cube, gamma=1.5) 
cv2.imwrite(rf'{folder}\{img_ID}_results\original_RGB_{img_ID}.jpg', original_RGB[:,:,::-1])
plt.imshow(original_RGB)
plt.axis('off')
plt.show()

#%%
'''
Background masking using a reference plant sepctrum
'''
cropped_masked_cube, _ = SpecimIQ_background_masking(hyperspectral_cube, threshold_val=1.0)

'''
Resize the masked cube to 200 x 200 pixels, and normalize 
'''
pix_num = 200
resized_masked_cube = resize(cropped_masked_cube, (pix_num, pix_num, 204), mode='reflect', anti_aliasing=True)
normalized_cube = resized_masked_cube / resized_masked_cube[:,:,170:175].mean(axis =2)[:,:, np.newaxis]

#%%
'''
Projection and CMeans clustering to define green and purple pixels 
'''
for f, cube in [['Original', resized_masked_cube], ['Normalized', normalized_cube]]:

    try:
        os.mkdir(rf'{folder}\{img_ID}_results\{f}')
    except FileExistsError:
        print(f"The {img_ID}_results\{f} directory already exists.")
    
    for method in ['PCA', 'NMF', 'ICA', 'SparsePCA', 'SVD']:
        model, projected_cube = hsi_spec_comp_analysis(cube, dim =5, method=method, path=rf'{folder}\{img_ID}_results\{f}')
        np.savetxt(rf'{folder}\{img_ID}_results\{f}\{method}_components.csv', model.components_, fmt="%.4f", delimiter=',') 
        [plt.plot(specimIQ_wavelength[10:], model.components_[i, 10:], label=f'{method} Comp. {i+1}') for i in np.arange(5)]; plt.xlabel('Wavelength [nm]'); plt.ylabel('Component Value'); plt.legend()
        plt.savefig(rf'{folder}\{img_ID}_results\{f}\{method}_components.pdf'); plt.show()

    for cluster_method in ['CMeans', 'GMM']: 
        membership, reflectance = hsi_pixel_clustering(cube, num_clusters=3, method=cluster_method, path=rf'{folder}\{img_ID}_results\{f}')
        
#%%
'''
Visualize the effect of normalization using 875-925nm nm bands
Manually select Cluster Color
'''
for f, cube in [['Original', resized_masked_cube], ['Normalized', normalized_cube]]:
    x, y, l = cube.shape
    plt.figure(figsize=(6, 4))

    for cluster, color in [[0, 'Darkred'], [1, 'Lightgreen'], [2, 'Olive']]:
        # Get cluster pixels
        c = membership == cluster
        c_values = (cube.reshape(x * y, l) * c[:, np.newaxis])
        
        # Filter out rows with all zeros
        c_values = c_values[~np.all(c_values == 0, axis=1)]
        
        # Calculate the mean and standard deviation across the pixel data for each wavelength
        c_value_mean = np.nanmean(c_values, axis=0)
        c_value_std = np.nanstd(c_values, axis=0)   
        
        if c_value_mean.sum() > 0:
            # Plot the mean reflectance with standard deviation bands for individual clusters
            plt.plot(specimIQ_wavelength[10:], c_value_mean[10:], c=color, label=color)
            plt.fill_between(specimIQ_wavelength[10:], c_value_mean[10:] - c_value_std[10:], c_value_mean[10:] + c_value_std[10:], color=color, alpha=0.3)
            print(f'Cluster {cluster} pixel number: {np.sum(c)}')
    
    # Add labels and title
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.title('Mean Reflectance for Individual Cluster Masks')
    plt.legend()
    plt.savefig(rf'{folder}\{img_ID}_results\{f}\{cluster_method}_Cluster_Pixel_Reflectance.pdf')
    plt.show()
    
