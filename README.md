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
- Binning of **APPLICATION_TYPE** and **CLASSIFICATION** columns is performed as there are more than 10 unique values in each. For doing so, cutoff points are identified and "rare" categorical variables are binned together in a new value, **Other**.</br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/ac0aca93-8897-40a0-8297-e6e46039259d)
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/14c4c338-985e-4419-8ef6-3ca6d497f159)
- Categorical data are converted to numeric with `pd.get_dummies`.
- Preprocessed data is split into target `X` and features `y` arrays.
- `train_test_split` from `sklearn.model_selection` was used to split the data into training and testing datasets `X_train, X_test, y_train, y_test`.
- Training and testing features datasets are scaled by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.
  
### Step 2: Compile, Train, and Evaluate the Model
- A neural network model is created by assigning the number of input features (44) and nodes for each layer using TensorFlow and Keras.
- The first hidden layer has an activation function `relu`and 80 neurons.
- A second hidden layer with an activation function `relu` is created with 30 neurons.
- Output layer is added with 1 neuron and activation function `sigmoid` since we were creating a binary classification model </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/552cf35c-a16c-4e50-9005-7fe125c5ee65)

- Structure of the model created is as below: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/40b321c9-6bc6-40d3-afc4-de446b409c2b)
- The model is compiled.
- A callback that saves the model's weights every five epochs is created </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/0265cdcb-710c-4707-9c75-d6d087218d31)
- The model is evaluated using the test data to determine the loss and accuracy.</br>
  ![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/8a0684a0-0ef8-4ede-aae2-eac737ce46b1)
  - Weights are saved every 5 epochs in `Weights` folder as file name format `weights{epoch:04d}.h5`
##### Model evaluation:
This model with a total of 3 layers including input, output and a hidden layer with 44 input features and activation methods relu, relu and sigmoid had an accuracy of 72.5% and loss of 56%.
- The model is saved and export to an HDF5 file: `AlphabetSoupCharity.h5`in `Models' folder

### Step 3: Optimising the Model
- Using our knowledge of TensorFlow, an effort to optimise the model to achieve a target predictive accuracy higher than 75% is made.
- A new Jupyter Notebook file `AlphabetSoupCharity_Optimisation.ipynb` is created.
- Dependencies are imported and the charity_data.csv is read to a Pandas DataFrame.
- The dataset is preprocessed as in Step 1.
- Based on results from the initial model, four attempts are made on designing a neural network model with target predictive accuracy higher than 75%:</br>

**Image 1** </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/04d787ba-3763-4668-85c7-345905de8f1b)


### Model optimisation method 1
#### Increasing number of layers and neurons 
- Without changing the inputs from the previous model, number of neurons and number of layers are increased. Note: Tried many permutaitons and combinations by increasing neurons, increasing number of layers, increasing number of layers and neurons and kept one of the iterations in the step in the notebook. For details ref **Image 1** above.
- Model structure: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/b19faf54-6b1a-46e4-b307-953f717990f6)
- Model training: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/cf244b5d-3eb9-4c5b-b410-1fb6a9576571)
- Model evaluation: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/3b40d290-c203-4edb-a4f7-69d174df46eb)
- **Result**:
  - No change from initial model is observed. 
  - Note: Tried many permutaitons and combinations by increasing neurons, increasing number of layers, increasing number of layers and neurons and kept one of the iterations in the notebook. No observation had better accuracy.  

### Model optimisation method 2
#### Changing activation method and neurons
- Without changing the inputs from the previous model, number of neurons increased. `tanh` is used instead of `relu` in one of the layers. Note: Tried many permutaitons and combinations of activation functions and neurons/layers and kept one of the iterations in the step in the notebook. For details ref **Image 1** above.
- Model structure: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/76831016-435b-4bfc-8c89-0c016e68a9a9)
- Model training: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/7347a71e-869b-44c3-866a-f01da996dc6e)
- Model evaluation: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/8ce42547-17ff-4346-bf27-e6a1a3d71535)

- **Result**:
    - No change from previous model is observed at **72.3%** accuracy. 

### Model optimisation method 3
#### Dropping status and special considerations related columns
- Since on analysing the unique values of each columns it was found that status and special considerations had two unique values each with very few records on one of the two, it should not have too much contribution to the model. I planned to check this by dropping the two columns.
- No changes to the inputs from the initial model for number of layers, neurons and activation methods . For details ref **Image 1** above.
- Model structure: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/100ce13e-75c1-4d2d-984d-b932eca4812e)
- Model training: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/01c75503-630e-42c1-b613-eeba4841f490)
- Model evaluation: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/423eca86-92d1-4ee0-b3b5-bfb82c015dc4)

- **Result**:
  - As expected no change from previous model is observed at **72.4%** accuracy.

#### Method 3b - Increasing number of epochs to 200 
- Model training: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/16eca264-8a32-4bcd-a9d4-f1d6ca787f3f)
- Model evaluation: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/8819822c-789b-490b-b8a4-2564ff3871eb)
- **Result**:
   - We achieve **72.3%** accuracy which is again similar to the last models. Also to be noted that loss is **around 53%** in all above trials

### Model optimisation method 4
#### Checking Name/EID column values for duplicates (reappearance over time) and if one of them needs to be included for creating our model

- We could see that names of organisation are not unique like EIN which is the sole unique identifier for this data. By dropping only EIN column we generated a model to see if it has any impact on accuracy of model when tested </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/affaa732-b273-4351-87f1-b7d1b3cde3df) </br>
- `NAME` was retained in the original dataset while only `EIN` was dropped.
- Binning was performed on `NAME` to reduce number of features. For doing so, cutoff points were identified and "rare" categorical variables were binned together in a new value, **Other** </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/5d7e63d2-c191-468c-828e-04825177a687)
- Categorical data was converted using pd.get_dummies after all binning activities
- No changes to the inputs from the initial model for number of layers, neurons and activation methods . For details ref **Image 1** above.
- Model structure: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/e269c0b0-39dd-4524-971e-e84a107f5741)

- Model training: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/b539a52c-37a4-47d8-81a5-904f6e47ad05)

- Model evaluation: </br>
![image](https://github.com/jyojay/deep-learning-challenge/assets/132628129/e941a6b9-b38a-449e-9352-694022ab5ddc)

- **Result**:
  - A model accuracy of **79.4%** is observed with model trained on as less as 20 epochs and 30 neurons between 3 layers with activation functions relu, relu and sigmoid. Loss has also reduced to **44.2%** 

- Results are Saved and exported as an HDF5 file AlphabetSoupCharity_Optimisation.h5`.
- Weights are also saved in csv format from first layer of method 3 and 4 at the end of model training for our evaluation.
**Please refer to the report for more details**
