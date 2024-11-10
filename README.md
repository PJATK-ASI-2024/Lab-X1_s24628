# Documentation

## Correlation Matrix 

The correlation matrix provides an overview of relationships between numerical variables in the dataset. 
Strong correlations (positive or negative) are highlighted in darker colors, with values ranging from -1 to +1.

## Feature Importance

The feature importance plot shows the relative importance of each feature in predicting the target variable within a Random Forest model. 
The higher the bar, the more significant the feature is for the model's decision-making process.

## Comparison of Actual vs Predicted Categories

This bar plot compares the actual and predicted frequency distributions across five score categories: high, low, medium, very high, 
and very low. Each category displays the frequency count for both actual (blue) and predicted (orange) values, 
allowing for an assessment of how well the model’s predictions align with the real data distribution.

## Confusion Matrix of Predicted vs Actual

The plot is a confusion matrix, showing the performance of a model in classifying data into five categories: very low, low, medium, high, and very high. 
The rows represent the true labels, while the columns represent the predicted labels. 
The numbers in each cell indicate the number of instances that were classified into a particular predicted category, given their true category. 
For example, the cell in the second row and third column shows that 45 instances were correctly classified as "low" (true label) but were predicted as "medium" (predicted label).