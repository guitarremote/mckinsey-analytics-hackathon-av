
# McKinsey Analytics Online Hackathon - Healthcare Analytics-Analytics Vidhya

My take on the McKnsey Analytics Hackathon hosted by Analytics Vidhya. The problem statement was to predict the probability of stroke happening to a given set of patients. The dataset was simple with only 10 features such as gender, age, hypertension, maritial status etc., and the evaluation metric was AUC ROC score. The dataset was imbalanced with majority of observations not having a stroke. I tried multiple approaches such as:

* Class weights: Impose heavier costs when errors are made on minority class
* Down-samplng: Randomly remove instances in the majority class
* Up-sampling: Randomly replicate instances in the minority class
* SMOTE(synthetic minority oversampling technique): Synthesize new minority instances by interpolating between existng ones and down sample majoty class

Since AUC ROC is a threshold independent metric, we can only expect some minor improvement in scores using these methods. Using class weights fetched me the best result. I could not spend much time on feature engineering or coming up with better imputing strategies for missing data. Some things I should have spent more time and tried out are knn-immputation, coarse classification of some continous variables, extracting new features, random forests and a manual grid search
