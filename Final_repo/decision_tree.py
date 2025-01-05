import numpy as np

def split_data(X, y, feature, threshold):
    # split data based on specified threshold
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

def gini_impurity(counts):
    # compute gini impurity based on the formula
    total = counts.sum()
    prob = counts / total
    return 1 - np.sum(prob**2)

class DecisionTree:
    # initialize hyperparameters
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # fit the model (start recursive function with specified data)
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    # main function 
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        # stopping condition based on hyperparameters
        if n_samples < self.min_samples_split or depth == self.max_depth or np.unique(y).size == 1:
            return {'type': 'leaf', 'class': np.bincount(y).argmax()}

        best_feature, best_threshold, best_gain = None, None, -np.inf

        # gini impurity of the total classes (used for calculation of information gain)
        parent_impurity = gini_impurity(np.bincount(y))

        # loop through features and get the best split at each feature based on information gain
        for feature in range(n_features):
            # sort data based on current feature values
            sorted_idx = np.argsort(X[:, feature])
            sorted_y = y[sorted_idx]
            sorted_x = X[sorted_idx, feature]

            # computing thresholds based on unique feature values
            unique_values, first_indices = np.unique(sorted_x, return_index=True)
            if len(first_indices) < 2:
                continue  # No valid split possible

            # calculating left sizes and right sizes dynamically
            sizes_left = first_indices[1:]  # sizes of left splits for each potential threshold
            sizes_right = n_samples - sizes_left # deduct right sizes

            # ensure we only consider valid splits
            # if the minimum sample of the left and right split is above or equal to
            # the min_samples_split parameter then the calculation should keep going
            valid_splits = np.where((sizes_left >= self.min_samples_split) & 
                                    (sizes_right >= self.min_samples_split))[0]
            if valid_splits.size == 0:
                continue

            # calculate impurities for these splits
            # cumulative counts for left splits to calculate impurities
            cumulative_y_counts = np.array([np.bincount(sorted_y[:idx], minlength=np.max(y)+1) 
                                            for idx in sizes_left[valid_splits]])
            left_impurities = np.array([gini_impurity(counts) for counts in cumulative_y_counts])
            # compute right split impurities by subtracting left counts from total counts
            right_y_counts = np.bincount(sorted_y, minlength=np.max(y)+1) - cumulative_y_counts
            right_impurities = np.array([gini_impurity(counts) for counts in right_y_counts])

            # calculate information gains
            left_sizes = sizes_left[valid_splits]
            right_sizes = sizes_right[valid_splits]
            gains = parent_impurity - (left_sizes / n_samples * left_impurities + 
                                       right_sizes / n_samples * right_impurities)

            # track the best gain and corresponding split
            max_gain_idx = np.argmax(gains)
            if gains[max_gain_idx] > best_gain:
                best_gain = gains[max_gain_idx]
                best_feature = feature
                best_threshold = unique_values[valid_splits[max_gain_idx]]

        # if no valid split is found, create a leaf node
        if best_gain == -np.inf:
            return {'type': 'leaf', 'class': np.bincount(y).argmax()}

        # split data based on the best feature and threshold
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[right_idx], y[right_idx]
        # recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        # return a decision node with split details
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
            # if tree is not a leaf then make the split based on the threshold
            if x[tree['feature']] <= tree['threshold']:
                tree = tree['left']
            else:
                tree = tree['right']
        # if tree is a leaf, then the class from the tree is returned.
        return tree['class']