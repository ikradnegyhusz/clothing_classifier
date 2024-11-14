import numpy as np
from sklearn.model_selection import train_test_split
import pickle
print("Imports done.")

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1 - np.sum(prob**2)

def information_gain(y, y_left, y_right):
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    
    return gini_impurity(y) - (p_left * gini_impurity(y_left) + p_right * gini_impurity(y_right))

def split_data(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        print(depth)
        n_samples, n_features = X.shape
        #stopping conditions
        if n_samples < self.min_samples_split or depth == self.max_depth or len(np.unique(y)) == 1:
            if len(y)==len(set(y)):
                cls_return = y[np.random.randint(0,len(y))]
            else:
                cls_return=np.argmax(np.bincount(y))

            return {'type': 'leaf', 'class': cls_return}
        
        # splitting, finding best split with highest information gain
        best_feature, best_threshold, best_gain = None, None, -np.inf
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_data(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain == -np.inf:
            return {'type': 'leaf', 'class': np.argmax(np.bincount(y))}
        
        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, tree):
        if tree['type'] == 'leaf':
            return tree['class']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])


train = np.load("fashion_train.npy")
print("Loaded data.")
X = train[:,:-1]
y = train[:,-1]

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
print("Split data.")

tree = DecisionTree(max_depth=3)
print("Made decision tree.")
tree.fit(X_train, y_train)
print("Fitted decision tree.")

predictions = tree.predict(X_val)

pickle.dump(tree,open("model.pkl","wb"))
pickle.dump(X_train,open("X_train.pkl","wb"))
pickle.dump(y_train,open("y_train.pkl","wb"))
pickle.dump(X_val,open("X_val.pkl","wb"))
pickle.dump(y_val,open("y_val.pkl","wb"))
pickle.dump(predictions,open("preds_on_y_val.pkl","wb"))
