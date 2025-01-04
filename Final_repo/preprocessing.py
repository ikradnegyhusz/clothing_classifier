import pickle
import numpy as np

def preprocess(data):
    '''
    Function to preprocess the data by scaling and applying PCA
    imput: numpy array
    '''
    df_scaled = (data[:, :-1] - data[:, :-1].mean(axis=1, keepdims=True)) / data[:, :-1].std(axis=1, keepdims=True)
    pca = pickle.load(open('./models/pca_model.pkl', 'rb'))
    df_scaled_pcs = pca.transform(df_scaled)
    y = data[:,-1] 
    y = y.reshape(-1, 1) 
    df_with_y = np.hstack((df_scaled_pcs, y)) 
    return df_with_y