# Documentation

## Correlation Matrix 

<img width="485" alt="Zrzut ekranu 2024-11-7 o 21 45 53" src="https://github.com/user-attachments/assets/d417d2f1-d97a-4ec3-b298-1796d96994a6">

The correlation matrix provides an overview of relationships between numerical variables in the dataset. 
Strong correlations (positive or negative) are highlighted in darker colors, with values ranging from -1 to +1.

## Feature Importance

<img width="567" alt="Zrzut ekranu 2024-11-7 o 21 46 04" src="https://github.com/user-attachments/assets/c880eb36-3a0e-4a9e-bb09-a525b37b32fc">

The feature importance plot shows the relative importance of each feature in predicting the target variable within a Random Forest model. 
The higher the bar, the more significant the feature is for the model's decision-making process.

## Comparison of Actual vs Predicted Categories

<img width="565" alt="Zrzut ekranu 2024-11-7 o 21 46 10" src="https://github.com/user-attachments/assets/c45a90d7-0669-4b26-8403-ba5db2bd985d">

This bar plot compares the actual and predicted frequency distributions across five score categories: high, low, medium, very high, 
and very low. Each category displays the frequency count for both actual (blue) and predicted (orange) values, 
allowing for an assessment of how well the model’s predictions align with the real data distribution.

## Confusion Matrix of Predicted vs Actual

<img width="567" alt="Zrzut ekranu 2024-11-7 o 21 46 16" src="https://github.com/user-attachments/assets/11a04bdc-bd63-42aa-98bb-cf8c90fd4c19">

The plot is a confusion matrix, showing the performance of a model in classifying data into five categories: very low, low, medium, high, and very high. 
The rows represent the true labels, while the columns represent the predicted labels. 
The numbers in each cell indicate the number of instances that were classified into a particular predicted category, given their true category. 
For example, the cell in the second row and third column shows that 45 instances were correctly classified as "low" (true label) but were predicted as "medium" (predicted label).

