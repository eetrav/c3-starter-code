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
Random Forest with a max depth of 15 and 45 estimators.

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
highest performing model, with a max depth of 15 and 45 estimators, achieved:
Precision: 0.788496
Recall: 0.575953
fbeta: 0.665671

## Ethical Considerations

## Caveats and Recommendations
