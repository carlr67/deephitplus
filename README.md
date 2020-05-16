# DeepHit+
This repository contains code for *Automatic Feature Selection for Survival Analysis with Competing Risks using Deep Learning*
- Read the quick [introduction](#introduction) to survival analysis and the purpose of my models/code
- Jump to [references](#references) to get up to speed with the theory. Recommended read before use of the models.
- Get started using these [instructions](#instructions) to build your own models using the code


## Introduction
Survival analysis is an important branch of statistics and machine learning, and is applied in many fields, including medicine and finance. It requires different models to traditional classification or regression problems due to the notion of time, and incomplete observations due to *censoring* (e.g. a medical trial limited to 10 years can only record observations up to that maximum time horizon). Examples for survival analysis questions in the medical realm include
- What is the likelihood of developing a cancer in 1, 3 or 5 years?
- What is the impact of a treatment on a patient's probability of survival?

Deep learning approaches to survival analysis were the topic of my research as a master's student, using data from a large medical trial. This data included a large number of features, many of which potentially irrelevant to the prediction problem, and which, as I observed, led to a loss of performance of deep learning models.

### Models
My research developed extensions to the model [DeepHit](#original-aaai-deephit-paper) with feature selection techniques specifically adapted for both survival analysis and deep learning models, with the objective of improving model performance. Please see my [MSc Thesis](#msc-thesis) and [NeurIPS Workshop Paper](#neurips-workshop-paper) for detailed description of the new methods:
- **DeepHitPlus**: This model implements DeepHit, with slight modifications to use C-index as early stopping criterion, and random search on the subnetwork layer sizes.
- **FilterDeepHitPlus**: This model uses one of three filter feature selection methods (ANOVA, SVM, ReliefF) to pre-process the data. Benefits of the method are its simple approach and improved performance, albeit limited by the drawbacks of each filter method used.
- **HybridDeepHitPlus**: This model selects features using feature importances extracted from the deep learning model itself, thus being able to select more tailored features that can contribute towards performance in a deep learning setting. Drawbacks are the increased complexity and having to train the deep learning model twice.
- **SparseDeepHitPlus**: This model uses a sparse and regularized initial layer to implement soft feature selection. This has the charm of using the network architecture itself for inbuilt feature selection, though performance is very dependent on hyperparameters.
- **AttentiveDeepHitPlus**: This model was an experimental part of my MSc thesis, using a separate network to generate attention masks for soft feature selection. Despite underperforming in my first experiments, potential benefits include allowing example-specific feature selection and improved interpretability.

### Uploaded code
The currently uploaded code base has full functionality to run *DeepHitPlus*, *FilterDeepHitPlus* and *HybridDeepHitPlus*.

It does not include *SparseDeepHitPlus* and *AttentiveDeepHitPlus*, as these are separate implementations (please contact me if you are interested in these too). I encourage further research on these ideas with the charm of 'inbuilt' feature selection and model interpretability).

### Acknowledgements
I would like to thank my MSc thesis supervisor [Professor Mihaela van der Schaar](https://www.turing.ac.uk/people/researchers/mihaela-van-der-schaar) and her inspiring research group, in particular [Jinsung Yoon](https://sites.google.com/view/jinsungyoon) and [Changhee Lee](http://www.vanderschaar-lab.com/team/), for their invaluable support, advice and insights in conducting this research.


## References

### MSc Thesis
My MSc Thesis gives a broad background of survival analysis with deep learning, introducing the relevant concepts from the ground up. Good as an introduction for a reader outside the field, or for covering new models and additional details not included in the short paper below.

Carl Rietschel. Automated Feature Selection for Survival Analysis with Deep Learning. University of Oxford, 2018. [Thesis](https://ora.ox.ac.uk/objects/uuid:e63f1610-11bd-46f0-af14-b310b4bea048).

### NeurIPS Workshop Paper
**Machine Learning for Health (ML4H) Workshop at NeurIPS 2018**
The paper gives a short summary of the highlights of my research. Useful for a quick glance of the results, but due to the page limit very abbreviated.

Carl Rietschel, Jinsung Yoon, and Mihaela van der Schaar. Feature selection for survival analysis with competing risks using deep learning. arXiv preprint arXiv:1811.09317, 2018. [Paper](https://arxiv.org/abs/1811.09317).

### Original AAAI DeepHit Paper
**AAAI Conference on Artificial Intelligence (AAAI), 2018**
This paper describes the original DeepHit model, on which my above extensions are based.

Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018. [Paper](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit). [Appendix](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit_Appendix). [Code](https://github.com/chl8856/DeepHit)


## Instructions
### Step 1: Connect data source
1. Write a function in `import_data.py` analogous to `import_dataset_SYNTHETIC` to import a new dataset
2. Edit the line `(x_dim), (full_data, time, label), (mask1, mask2), full_feat_list = impt.import_dataset_SYNTHETIC(norm_mode = 'standard')` in `run_deephitplus.py` to use the new function to load the new dataset
3. Edit the array `eval_time` to choose at which time horizons to conduct model performance evaluations (for testing and validation)

### Step 2: Choose DeepHitPlus version and related settings
Edit the settings in the second part of the `SETTINGS` section in `run_deephitplus.py` to select which model to run.

#### DeepHitPlus (no feature selection) ####
1. Set `features = 'all'` to run the model without feature selection
2. If running as a preparation for *HybridDeepHitPlus*, set `calculate_importances = 'ON'`, and choose a `NUM_PERMUTATIONS` (20 should work fine)

#### FilterDeepHitPlus ####
1. Set `features = ...` to run the model with feature selection using a filter method, with the following three options
   * `'filter-anova'` for feature selection using the ANOVA p-value
   * `'filter-svm'` for feature selection using SVM weights
   * `'filter-relieff'` for feature selection using ReliefF feature importances

#### HybridDeepHitPlus ####
1. First run the model as DeepHitPlus (with no feature selection) setting `calculate_importances = 'ON'`. This outputs feature importances in a results folder that HybridDeepHitPlus will use as its feature selection
2. Set `features = ...` to run the model as HybridDeepHitPlus, with the following four options:
   * `'hybrid-m-top'` for feature selection using the raw importance values, treating the number of features (top N) chosen per event as a hyperparameter
   * `'hybrid-m-cut'` for feature selection using the raw importance values, treating a cutoff value for feature importance per event as a hyperparameter
   * `'hybrid-p-top'` for feature selection using the p-value that the raw feature importance > 0, treating the number of features (top N) chosen per event as a hyperparameter
   * `'hybrid-p-cut'` for feature selection using the p-value that the raw feature importance > 0, treating a cutoff value for feature importance per event as a hyperparameter
3. Set `path_to_immportances` to the path to the results output folder containing the importances (see the first step in this list)

### Step 3: Edit the run settings (if desired)
1. Make selections in the section `run settings`
   * `random_search_mode` toggles whether to use random search, and `RS_ITERATION` the number of iterations (more usually better, dependent on computation time)
   * `CV_ITERATION` chooses the number for K-fold cross-validation, and `cv_to_search` which cross-validations iteration to conduct random search on (usually the first will suffice)
   * `valid_mode` toggles whether to use validation for early stopping during training (recommended)
2. Further selections of the random number generating seed are possible in the section below

### Step 4: Choose hyperparameters (or search space)
See [references](#references) for more details on each parameter
1. If conducting no random search, edit the hyperparameters in `DEFAULT_param_dict` for the search
2. If conducting random search, edit the hyperparameter search sets in `SET_param_dict` to define the search space

### Step 5: Install dependencies
To run the code, the following python packages are required:
1. `numpy`
2. `pandas`
3. `tensorflow`
4. `sklearn`
5. `skrebate` (only required for FilterDeepHitPlus with ReliefF)

### Step 6: Run the model
1. Run using `python run_deephitplus.py` in your terminal
2. Output and trained models are saved in the `./output/` folder for use, using the model type (feature selection method and random search mode), as well as the hyperparameter string as folder reference
