# DeepHitPlus
#

def apply_features(data, feat_list):
    # Takes as input feat_list, an array of feature lists (length: num_Event + 1)
    # First element = feature list for shared subnetwork
    # Second, third, ... elements = feature lists for cause-specific subnetworks

    # Get concatenated list of features
    c_feat_list = [inner for outer in feat_list for inner in outer]

    newdata = data[c_feat_list]
    x_dim = [len(x) for x in feat_list]

    DIM = (x_dim)

    return DIM, newdata

def get_feat_list(feature_mode, num_Event, data, original_feat_list):
    # Returns the feature list for the dataset depending on a chosen feature selection mode
    if feature_mode == "all":
        feat_list = [original_feat_list for i in range(num_Event + 1)]

    return feat_list
