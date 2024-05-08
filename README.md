# Predicting Student Exam Scores

### Introduction

The goal of this machine learning project is to predict the `math score` for a high school student based on social and economic factors, as well the test scores on other exams.

#### Independent variables

The numerical variables are:
* `reading score`
* `writing score`

The categorical variables are:
* `gender`
* `ethnicity`
* `parental level of education`
* `lunch`
* `test preparation course`
  
#### Target variable:
* `math score`

Dataset Source:
[http://roycekimmons.com/tools/generated_data/exams](http://roycekimmons.com/tools/generated_data/exams)


## AWS Deployment
AWS EC2 deployment link: [http://3.144.212.169:8080/predictions](http://3.144.212.169:8080/predictions)

The  application was deployed in an AWS EC2 instance using a Docker container and GitHub Actions for a Continuous Integration, Continuous Development, Continuous Deployment Environment.

An AWS ECR registry was used to store and deploy the docker container for this web application.

The application was built using Flask.

## User Interface

![User Interface](./interface.jpg)

Select the values for the independent variables and press "Predict the Math Score" button to get the predicted value.

## Project Approach

### 1. Data Ingestion: 
  * The raw data is read from a csv file and converted to a pandas DataFrame. 
  * The data is then split into training and testing sets and saved as csv files.

### 2. Data Transformation: 
  * In this phase a ColumnTransformer Pipeline is created that includes the numerical and categorical pipelines.
  * For numeric variables, a SimpleImputer to fill in missing variables with a median strategy, and then StandardScaler is applied to standardize features by removing the mean and scaling to unit variance.
  * For categorical variables, a SimpleImputer is applied with the most frequent strategy, then One Hot Encoding, and finally a StandardScaler.
  * This data transformation pipeline is saved as a pickle file called preprocessor.pkl

### 3. Model Training: 
 * In this phase the base model is train and evaluated against the dataset.
 * After using a grid search and hyperparameter tuning the best performing model was linear regression.
 * This model is saved as pickle file called trained_model.pkl.

### 4. Prediction Pipeline: 
 * This pipeline takes data given to it as a post request.
 * Converts this data into a pandas DataFrames.
 * Loads the preprocessor and trained models pickle files.
 * Uses the data and the pickle files to predict and return the final result.

### 5. Flask Application: 
 * The flask application [app.py](./app.py) is created to link the user interface to the Python backend used to predict the math scores. It is the core of the web application.

## Exploratory Data Analysis Notebook

Link: [EDA Notebook](./notebooks/EDA\-\Student\Scores.ipynb)

## Model Training  Notebook

Link: [Model Training Notebook](./notebooks/Model\Training.ipynb)