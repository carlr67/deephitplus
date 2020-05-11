# DeepHitPlus
#

import pandas as pd
import numpy as np

def apply_features(full_data, full_feat_list, feat_list):
    # Takes as input feat_list, an array of feature lists (length: num_Event + 1)
    # First element = feature list for shared subnetwork
    # Second, third, ... elements = feature lists for cause-specific subnetworks

    # Get concatenated list of features
    c_feat_list = [inner for outer in feat_list for inner in outer]

    data_df = pd.DataFrame(full_data, columns=full_feat_list)

    newdata = np.asarray(data_df[c_feat_list])
    x_dim = [len(x) for x in feat_list]

    DIM = (x_dim)

    return DIM, newdata

def get_feat_list(features, num_Event, data, full_feat_list):
    # Returns the feature list for the dataset depending on a chosen feature selection mode
    if features == "all":
        feat_list = [full_feat_list for i in range(num_Event + 1)]

    return feat_list
