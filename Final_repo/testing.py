import preprocessing as pre
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def bootstrapping_test(model_filepath,n=1000,prnt = True):
    X, y = pre.load_data('data/fashion_test.npy')
    pca = pickle.load(open("models/pca_65.pkl","rb"))
    X_processed = pre.preprocess(X, pca=pca)
    model = pickle.load(open(model_filepath, 'rb'))

    df = pd.DataFrame(X_processed)
    df['y'] = y

    acc_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    matrix = np.zeros((5, 5))

    for i in range(0, n):
        if prnt:
            print(f"{round(100*i/n,2)}% ",end="")
            print("|"*int(100*(i/n)),end="")
            print(" "*(99-int(100*(i/n))),end="")
            print("|",end="\r")
        sample = df.sample(len(df), replace=True)
        X = sample.iloc[:,:-1].to_numpy()
        y = sample.iloc[:,-1].to_numpy()
        preds = model.predict(X)
        if (type(preds[0]) is np.array) or (type(preds[0]) is not np.int64):
            preds = np.argmax(preds,axis=1)
        accuracy = accuracy_score(y,preds)
        recall_score_list.append(recall_score(y, preds, average='weighted'))
        precision_score_list.append(precision_score(y, preds, average='weighted'))
        f1_score_list.append(f1_score(y, preds, average='weighted'))
        acc_list.append(accuracy)
        matrix += confusion_matrix(y, preds)
    
    results={}
    results["accuracy"]=round(np.mean(acc_list)*100, 2)
    results["accuracy_std"]=round(np.std(acc_list)*100, 2)
    results["recall"]=round(np.mean(recall_score_list)*100, 2)
    results["recall_std"]=round(np.std(recall_score_list)*100, 2)
    results["precision"]=round(np.mean(precision_score_list)*100, 2)
    results["precision_std"]=round(np.std(precision_score_list)*100, 2)
    results["f1"]=round(np.mean(f1_score_list)*100, 2)
    results["f1_std"]=round(np.std(f1_score_list)*100, 2)
    results["cm"]=(matrix/n)/np.sum(matrix/n, axis=1)[:, np.newaxis]
    if prnt:
        print(f'accuracy: {results["accuracy"]}% ± {results["accuracy_std"]}%')
        print(f'recall: {results["recall"]}% ± {results["recall_std"]}%')
        print(f'precision: {results["precision"]}% ± {results["precision_std"]}%')
        print(f'f1: {results["f1"]}% ± {results["f1_std"]}%')
        print('confusion matrix:\n', results["cm"])
    return results

def plot_confusion_matrix(matrix,title="Confusion Matrix"):
    row_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Shirt']
    column_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Shirt']
    df = pd.DataFrame(matrix, index=row_labels, columns=column_labels)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df, annot=True, cmap='viridis', cbar=True)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.tick_top() 

    # Add title and labels
    plt.title(title, y=1.1)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")