import numpy as np
from sklearn.decomposition import PCA

def load_data(filename):
    data = np.load(filename)
    X=data[:,0:-1]
    y=data[:,-1]
    return X,y

def standardize(image_data,axis=1):
    # Compute mean and std along the picked axis
    mean = np.mean(image_data, axis=axis, keepdims=True)
    std = np.std(image_data, axis=axis, keepdims=True)
    # Subtract the mean and divide by standard deviation (so mean=0, and std = 1)
    scaled_images = (image_data - mean) / std
    return scaled_images

def preprocess(X,pca=None,principal_components=34):
    X_standardized = standardize(image_data=X,axis=1) # standardize by rows
    X_standardized_rows_centered = X_standardized - np.mean(X_standardized, axis=0) # center each feature around 0 to remove mean differences so PCA is not biased
    if pca == None:
        pca = PCA(n_components=principal_components) # make PCA, with specified principal components (by default 34 from previous findings. See eda.ipynb)
        return pca.fit_transform(X_standardized_rows_centered),pca # fit then project and return the resulting projection and PCA object
    else:
        return pca.transform(X_standardized_rows_centered)