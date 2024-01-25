# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Author: Emily Travinsky
- Date: 01/2024
- Version: 1.0.0
- Type: Random Forest Classifier with GridSearch and Cross Validation
- Implementation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_
- Questions: eberkson@email.arizona.edu

The trained model is a Random Forest Classifier from the Python scikit-learn
package. The Classifier is trained with a grid search for the best
parameter combinations of n_estimators and max_depth. The grid search is run
with a 10-fold cross validation. The optimal performance was found for a 
Random Forest with a max depth of 15 and 60 estimators.

## Intended Use
This model is intended to be used for predicting salary classifications using
US Census Data. The model is trained to predict whether individuals earn above
or below $50k per year. This model is not intended to be used with alternate
US Census Data formats or inputs, or to classify salaries with a decision 
boundary other than $50k.

## Training Data
There were 32,561 instances in the provided US Census Data, and 80% of that 
was used for model training. Categorical data was one-hot encoded and labels
were binarized. The training and testing data was stratified based on sex.

## Evaluation Data
20% of the 32,561 Census Data instances were used for model testing.

## Metrics
Model performance was gauged based on precision, recall, and fbeta score. The
highest performing model, with a max depth of 15 and 60 estimators, achieved:
Precision: 0.806
Recall: 0.553
fbeta: 0.656

## Ethical Considerations
The dataset is heavily imbalanced in representation of native US citizens and 
various race categories. The sliced_metrics.csv file can be used to compare
overall model performance to the model performance on individual groups.

For instance, the overall model achieved an average precision of 0.788, but the
Handlers-Cleaners occupation only achieved an average precision of 0.57.

The average model recall was 0.576, but all marital-status categories aside
from Married-Civ-Spouse achieved recall scores below 0.37.

## Caveats and Recommendations
Downstream decisions should not be made based on this model's salary 
predictions, given the Ethical Considerations listed above. Before utilizing
this model for any policy-making decisions, the Ethical Considerations should
be explored further, specifically with model performance on individual classes.