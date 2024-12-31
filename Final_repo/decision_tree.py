import math
import numpy as np


def gini_impurity(y, K_classes):
    if len(y) == 0:
        return 0
    s = 0
    for i in K_classes:
        p = np.sum(y == i) / len(y)
        s += p**2
    return 1-s

def PL_weight(N_L_or_R, N_u):
    '''
    N_L - number of elements in the left split
    N_u - number of elements in the node split
    N_R - number of elements in the right split
    '''
    return N_L_or_R / N_u

def calculate_depth(tree):
    # If the tree list is empty, depth is 0
    if not tree:
        return 0
    # Calculate depth using the formula
    return math.floor(math.log2(len(tree))) + 1

def max_depth(tree):
    if not tree:
        return 0
    return 2**tree - 1

class tree:
    def __init__(self, df, K_classes):
        higest_info_gain = [0, 0, 0]

        for row in range(1,len(df)):
            for col in range(1,len(df[0])-1):
                right = df[df[:, col] >= df[row, col]][:, -1]
                left = df[df[:, col] < df[row, col]][:, -1]

                gini_right = gini_impurity(right, K_classes) 
                gini_left = gini_impurity(left, K_classes)

                w_right = PL_weight(len(right), len(df))
                w_left = PL_weight(len(left), len(df))

                gini_combined_weigthed = w_right * gini_right + w_left * gini_left

                info_gain = gini_impurity(df[:, -1], K_classes) - gini_combined_weigthed

                if info_gain > higest_info_gain[0]:
                    higest_info_gain = [info_gain, row, col, gini_combined_weigthed]
            print(f'{row/len(df)*100}%')

        best_row, best_col = higest_info_gain[1], higest_info_gain[2]
        threshold = df[best_row, best_col]


        right_split = df[df[:, best_col] >= threshold]
        left_split = df[df[:, best_col] < threshold]


        self.matrix = df
        self.threshold = threshold
        self.right_split = right_split
        self.left_split = left_split
        self.info_gain = higest_info_gain[0]
        self.gini_combined_weigthed = higest_info_gain[2]

class DT:
    def __init__(self, df, K_classes, depth):
        self.df = df
        self.K_classes = K_classes
        self.dt = []
        i = 0
        while len(self.dt) <= max_depth(depth):
            if len(self.dt) < 3:
                root = tree(df, K_classes)
                right = tree(root.right_split, K_classes)
                left = tree(root.left_split, K_classes)
                if (root.info_gain == 0):
                    break
                self.dt.append(root)
                if (right.info_gain == 0):
                    break
                self.dt.append(right)
                if (left.info_gain == 0):
                    break
                self.dt.append(left)
            else:
                root = tree(self.dt[i].right_split, K_classes)
                right = tree(root.right_split, K_classes)
                left = tree(root.left_split, K_classes)
                if (root.info_gain == 0):
                    break
                self.dt.append(root)
                if (right.info_gain == 0):
                    break
                self.dt.append(right)
                
                if (left.info_gain == 0):
                    break
                self.dt.append(left)
            i += 1