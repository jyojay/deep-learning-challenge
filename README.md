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
In this Challenge we have created a model that can help nonprofit foundation Alphabet Soup select the applicants for funding with the best chance of success in their ventures. With our knowledge of machine learning and neural networks, we have used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
CSV containing more than 34,000 organisations that have received funding over the years from Alphabet Soup is provided in CSV format. Within this dataset are a number of columns that capture metadata about each organisation as below: </br> 
- **EIN** and **NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organisation classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organisation type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special considerations for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

### Step 1: Preprocessing the Data
- After the CSV is loaded into a Pandas Dataframe, ID columns, `EIN` and `NAME` are dropped as in instructions for the initial model.
- **IS_SUCCESSFUL** is identified as the **Target** of our model and the rest of the variables as the features.
- Number of unique values for each column is determined.
- Binning of **APPLICATION_TYPE** and **CLASSIFICATION** columns is performed as there are more than 10 unique values in each. For doing so, cutoff points are identified and "rare" categorical variables are binned together in a new value, Other.</br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/ac0aca93-8897-40a0-8297-e6e46039259d)
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/14c4c338-985e-4419-8ef6-3ca6d497f159)
- Categorical data are converted to numeric with `pd.get_dummies`.
- Preprocessed data is split into target `X` and features `y` arrays.
- `train_test_split` from `sklearn.model_selection` was used to split the data into training and testing datasets `X_train, X_test, y_train, y_test`.
- Training and testing features datasets are scaled by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.
  
### Step 2: Compile, Train, and Evaluate the Model
- A neural network model is created by assigning the number of input features (44) and nodes for each layer using TensorFlow and Keras.
- Tthe first hidden layer has an activation function `relu`and 80 neurons.
- A second hidden layer with an activation function `relu` is created with 30 neurons.
- Output layer is added with 1 neuron and activation function `sigmoid` since we were creating a binary classification model </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/4529cbb3-4e4f-4377-bf62-bb3f26d7c4a9)
- Structure of the model created is as below: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/40b321c9-6bc6-40d3-afc4-de446b409c2b)
- The model is compiled.
- A callback that saves the model's weights every five epochs is created </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/0265cdcb-710c-4707-9c75-d6d087218d31)
- The model is evaluated using the test data to determine the loss and accuracy.</br>
  ![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/8a0684a0-0ef8-4ede-aae2-eac737ce46b1)
  - Weights are saved every 5 epochs in `Weights` folder as file name format `weights{epoch:04d}.h5`
#### Result:
This model with a total of 3 layers including input, output and a hidden layer with 44 input features and activation methods relu, relu and sigmoid had an accuracy of 72.5% and loss of 56%.
- The model is saved and export to an HDF5 file: `AlphabetSoupCharity.h5`in `Models' folder

