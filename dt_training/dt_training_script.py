import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import time

time1=time.time()

def information_gain(y, y_left, y_right):
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    
    return gini_impurity(y) - (p_left * gini_impurity(y_left) + p_right * gini_impurity(y_right))

def split_data(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

def gini_impurity(counts):
    total = counts.sum()
    prob = counts / total
    return 1 - np.sum(prob**2)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth == self.max_depth or np.unique(y).size == 1:
            return {'type': 'leaf', 'class': np.bincount(y).argmax()}

        best_feature, best_threshold, best_gain = None, None, -np.inf
        parent_impurity = gini_impurity(np.bincount(y))

        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            sorted_y = y[sorted_idx]
            sorted_x = X[sorted_idx, feature]

            # Computing thresholds
            unique_values, first_indices = np.unique(sorted_x, return_index=True)
            if len(first_indices) < 2:
                continue  # No valid split possible

            # Calculating left sizes and right sizes dynamically
            sizes_left = first_indices[1:]  # sizes of left splits for each potential threshold
            sizes_right = n_samples - sizes_left

            # Ensure we only consider valid splits
            valid_splits = np.where((sizes_left >= self.min_samples_split) & 
                                    (sizes_right >= self.min_samples_split))[0]
            if valid_splits.size == 0:
                continue

            # Calculate impurities for these splits
            cumulative_y_counts = np.array([np.bincount(sorted_y[:idx], minlength=np.max(y)+1) 
                                            for idx in sizes_left[valid_splits]])
            left_impurities = np.array([gini_impurity(counts) for counts in cumulative_y_counts])
            right_y_counts = np.bincount(sorted_y, minlength=np.max(y)+1) - cumulative_y_counts
            right_impurities = np.array([gini_impurity(counts) for counts in right_y_counts])

            # Calculate gains
            left_sizes = sizes_left[valid_splits]
            right_sizes = sizes_right[valid_splits]
            gains = parent_impurity - (left_sizes / n_samples * left_impurities + 
                                       right_sizes / n_samples * right_impurities)

            max_gain_idx = np.argmax(gains)
            if gains[max_gain_idx] > best_gain:
                best_gain = gains[max_gain_idx]
                best_feature = feature
                best_threshold = unique_values[valid_splits[max_gain_idx]]

        if best_gain == -np.inf:
            return {'type': 'leaf', 'class': np.bincount(y).argmax()}

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[right_idx], y[right_idx]
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
        predictions = np.array([self._predict_one(x, self.tree) for x in X])
        return predictions

    def _predict_one(self, x, tree):
        while tree['type'] != 'leaf':
            if x[tree['feature']] <= tree['threshold']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree['class']



train = np.load("fashion_train.npy")
X = train[:,:-1]
y = train[:,-1]

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

predictions = tree.predict(X_val)

pickle.dump(tree,open("model.pkl","wb"))
pickle.dump(X_train,open("X_train.pkl","wb"))
pickle.dump(y_train,open("y_train.pkl","wb"))
pickle.dump(X_val,open("X_val.pkl","wb"))
pickle.dump(y_val,open("y_val.pkl","wb"))
pickle.dump(predictions,open("preds_on_y_val.pkl","wb"))

time2=time.time()
print(time2-time1)