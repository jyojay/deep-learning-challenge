# deep-learning-challenge
ML Neural Networks challenge 

## Table of Contents

- [Repository Folders and Contents](#Repository-Folders-and-Contents)
- [Libraries Imported](#Libraries-Imported)
- [Background and Details](#Background-and-Details)
  
## Repository Folders and Contents
- AlphabetSoupCharity.ipynb  --> Jupyter notebook code
- AlphabetSoupCharity_Optimisation.ipynb  --> Jupyter notebook code with multiple optimisation methods
- Models
    - AlphabetSoupCharity.h5
    - AlphabetSoupCharity_Optimisation.h5
- Weights --> THis consists of multiple weights files from callback that saved the initial model's weights every five epochs. Format: `weights{epoch:04d}.h5`
  - checkweightmethod2.csv --> Weights saved in csv format from first layer of method 3 at the end of model training
  - checkweightmethod4.csv --> Weights saved in csv format from first layer of method 4 at the end of model training
- Report.pdf --> A report on the DL models

## Libraries Imported
- train_test_split from sklearn.model_selection
- StandardScaler from sklearn.preprocessing
- pandas
- tensorflow

## Background and Details

In this Challenge we have used `LogisticRegression` to train and evaluate a model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company is used to build a model that can identify the creditworthiness of borrowers. </br>
The dataset consists of `loan_size`,	`interest_rate`,	`borrower_income`,	`debt_to_income`,	`num_of_accounts`,	`derogatory_marks`,	 `total_debt` and `loan_status` columns. A value of 0 in the `loan_status` column meant that the loan was healthy and a value of 1 meant that the loan had a high risk of defaulting. Based on this, `loan_status` column was used as label or feature to be predicted, and all the remaining columns from the dataset as known features. </br>

### Splitting the data into training and testing datasets by using train_test_split
Once the two dataframes for labels and features was created, `train_test_split` from `sklearn.model_selection` was used to split the data into training and testing datasets `X_train, X_test, y_train, y_test`
### Fitting a logistic regression model by using the training data
`LogisticRegression` from `sklearn.linear_model` was used to instantiate LogisticRegression Model and fit the training dataset `(X_train, y_train)`. Solver used was `lbfgs`, tried max_iter = 200 and got the same result as default iteration hence decided to allow code to use default iterations. Also random_state of 1 was used in this step.
### Saving the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model
Predictions of testing data labels were then made using testing feature data `X_test` using the fitted model and results compared against actuals: </br> </br>


### Evaluating the model’s performance



