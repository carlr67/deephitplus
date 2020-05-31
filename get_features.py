'''
Functions to select features based on various feature selection settings

Code by Carl Rietschel
https://github.com/carlr67/deephitplus
'''

import pandas as pd
import numpy as np

def apply_features(full_feat_list, feat_list, tr_data, va_data, te_data):
    # Takes as input feat_list, an array of feature lists (length: num_Event + 1)
    # First element = feature list for shared subnetwork
    # Second, third, ... elements = feature lists for cause-specific subnetworks

    # Get concatenated list of features
    c_feat_list = [inner for outer in feat_list for inner in outer]

    x_dim = [len(x) for x in feat_list]

    new_tr_data = np.asarray(pd.DataFrame(tr_data, columns=full_feat_list)[c_feat_list])
    new_va_data = np.asarray(pd.DataFrame(va_data, columns=full_feat_list)[c_feat_list])
    new_te_data = np.asarray(pd.DataFrame(te_data, columns=full_feat_list)[c_feat_list])

    DIM = (x_dim)

    return DIM, new_tr_data, new_va_data, new_te_data

def get_feat_list(features, num_Event, eval_time, data, full_feat_list, times, labels, param_dict, cv_iter, path_to_immportances):
    # Returns the feature list for the dataset depending on a chosen feature selection mode

    feature_mode = features.split("_")[0]

    if feature_mode in ["all", "sparse"]:
        feat_list = [full_feat_list for i in range(num_Event + 1)]

    elif feature_mode == "filter":
        method = features.split("_")[1]

        # Import packages for the chosen filter method
        if method == "anova":
            print("Using ANOVA p-value (in ascending order) for feature selection")
            from sklearn.feature_selection import f_classif
        elif method == "svm":
            print("Using SVM absolute coeffs (in descending order) for feature selection")
            from sklearn import svm
        elif method == "relieff":
            print("Using ReliefF feature importances (in descending order) for feature selection")
            from skrebate import ReliefF

        output = pd.DataFrame(full_feat_list, columns=["Feature"])
        output.set_index("Feature", inplace=True)

        dfindex = ['Horizon', 'Event', 'Feature']
        dfcols = ['Score']
        result = pd.DataFrame(columns = dfindex + dfcols)
        result.set_index(keys=dfindex, inplace=True)

        event_feat_list = []

        for event in range(1, num_Event + 1):
            print("  Now computing: Event", event)

            for ti in eval_time:
                print("                   time", ti)

                data_for_calc = data
                label_for_calc = ((times.flatten() < ti) & (labels.flatten() == event)) * 1

                if method == "anova":
                    # Get the ANOVA p-value
                    F, pval = f_classif(data_for_calc, label_for_calc)
                    feature_score = pval

                elif method == "svm":
                    clf = svm.SVC(kernel='linear')
                    clf.fit(data_for_calc, label_for_calc)
                    feature_score = np.absolute(clf.coef_).flatten()

                elif method == "relieff":
                    fs = ReliefF(discrete_threshold=10, n_features_to_select=100, n_jobs=-1, n_neighbors=100, verbose=True)
                    fs.fit(data_for_calc, label_for_calc)
                    feature_score = fs.feature_importances_

                tmp_result = pd.DataFrame({
                    'Horizon': ti,
                    'Event': event,
                    'Feature': full_feat_list,
                    'Score': feature_score})
                tmp_result.set_index(keys=dfindex, inplace=True)

                result = result.append(tmp_result)

            output["Event " + str(event)] = result.groupby(["Event", "Feature"]).mean().loc[event]

            eventdf = output["Event " + str(event)]
            top = param_dict['top'][event-1]

            if method in ["relieff", "svm"]:
                ascending = False
            elif method in["anova"]:
                ascending = True

            eventdf = eventdf.sort_values(ascending=ascending)

            print("Using top {} features from:".format(top))
            print(eventdf)
            print()

            filteredeventdf = eventdf.iloc[0:top].copy()

            event_feat_list.append(filteredeventdf.index.values)


        shared_feat_list = list(set.intersection(*[set(x) for x in event_feat_list]))

        feat_list = event_feat_list.copy()
        feat_list.insert(0, shared_feat_list)

    elif feature_mode == "hybrid":
        metric = features.split("_")[1]
        cutofftype = features.split("_")[2]

        # Load feature importance and get the most important features
        result = pd.read_csv(path_to_immportances + '/result_all_VAL-IMPORTANCES_cv' + str(cv_iter) + '.csv')

        # For using the p-value that a feature importance > 0
        if metric == 'p':
            avgdf0 = result.groupby(["Event", "Feature", "Permutation"]).mean()

            dim1 = len(avgdf0.index.get_level_values(0).unique())
            dim2 = len(avgdf0.index.get_level_values(1).unique())
            a = avgdf0.values.reshape((dim1, dim2, -1))

            from scipy.stats import ttest_1samp
            r = ttest_1samp(a=a, popmean=0, axis=2)

            analysisdf = avgdf0.groupby(["Event", "Feature"]).mean()
            analysisdf["statistic"] = r[0].flatten()
            analysisdf["Importance"] = r[1].flatten()/2 # Divide by 2 to get one-tailed test

            avgdf = analysisdf.query("statistic>0") # Only take ones with positive importance
            # print(avgdf)

            sortascending = True

        # For using 'raw' feature importances
        elif metric == 'm':
            avgdf = result.groupby(["Event", "Feature"]).mean()
            sortascending = False

        event_feat_list = []
        for e in range(num_Event):
            event = e + 1
            eventdf = avgdf.loc[event]

            print("Retrieving feature importances for event {}".format(event))

            eventdf.index.names = ["Feature (Event " + str(event) + ")"]
            eventdf.sort_values(by="Importance", ascending=sortascending, inplace=True)

            if metric == 'm' and cutofftype == 'cut':
                cutoff = param_dict['importancecutoff'][e]
                filteredeventdf = eventdf.query("Importance >= " + str(cutoff))
                print("Using features with raw importance (M) above {}:".format(cutoff))
            elif metric == 'p' and cutofftype == 'cut':
                cutoff = param_dict['importancecutoff'][e]
                filteredeventdf = eventdf.query("Importance <= " + str(cutoff))
                print("Using features with importance p-value (P) below {}:".format(cutoff))
            elif cutofftype == 'top':
                n_top = param_dict['top'][e]
                filteredeventdf = eventdf.iloc[0:n_top,:].copy()
                print("Using top {} features using metric({}) from:".format(n_top, metric))

            print(eventdf)
            print()

            event_feat_list.append(filteredeventdf.index.values)

        shared_feat_list = list(set.intersection(*[set(x) for x in event_feat_list]))

        feat_list = event_feat_list.copy()
        feat_list.insert(0, shared_feat_list)

    return feat_list
