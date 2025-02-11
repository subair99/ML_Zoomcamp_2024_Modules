
# Using Bagging And Xgboost To Train Large Datasets
I was recently confronted by a gigantic problem of having to train and score a dataset with 7,432,685 rows and 16 columns. The dataset is from the Kaggle [ML Zoomcamp 2024 Competition]( https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition) which is aimed at rewording the model with the best root mean squared error. Before I continue I will define and explain some concepts.


## What is Bagging?
Bagging, which is also known as Bootstrap Aggregating, is an ensemble technique that combines multiple instances of a base model trained on different subsets of the training data, selected randomly with replacement. The power of ensemble methods in the ability to combine multiple models thereby improving overall prediction accuracy and model stability. Bagging stands out as a popular and widely implemented ensemble method.


## What is Ensemble Modeling?
Ensemble modeling is a process by which multiple diverse base models are used to predict an outcome, with the motivation of reducing the generalization error of the prediction. If the base models are diverse and independent, the prediction error decreases when the ensemble approach is used. The steps involved in creating an ensemble model are multiple machine learning models are trained independently, then their predictions are aggregated by voting, averaging, or weighting. This ensemble is then used to make the overall prediction.


## Advantages of Bagging
1. The chances of overfitting can be reduced which results in improved model accuracy on testing data.
2. It makes it possible to conveniently train a very large dataset with one’s desired algorithm.
3. It averages out the predictions of multiple models trained on different subsets of data leading to lower variance than a single model.
4.  There is less impact on bagged models when there is changes in the training dataset leading to a more stable overall model
5. This will be a great help for algorithms that tend to have high variance, such as decision trees.
6. It allows for parallel processing and efficient use of computational resources as each model in the ensemble can be trained independently.
7. It removes the need for complex modifications to the learning algorithm since the concept behind bagging is straightforward.
8. Noise is reduced in the final prediction because of the averaging process.
9. The performance in a scenario where the dataset is imbalanced can be increased by bagging.


## Practical Demonstration
An example of bagging will be demonstrated using the dataset of the competition mentioned at the beginning of this article. We start by importing the required modules.

> # Filter warnings
> import warnings 
> warnings.filterwarnings('ignore')

> # Import required modules
> import numpy as np
> import pandas as pd
> import seaborn as sns
> import matplotlib.pyplot as plt
> from xgboost import XGBRegressor
> from joblib import Parallel, delayed
> from sklearn.impute import SimpleImputer
> from catboost import CatBoostRegressor, Pool
> from sklearn.ensemble import BaggingRegressor
> from sklearn.metrics import mean_squared_error
> from sklearn.model_selection import train_test_split
> from sklearn.preprocessing import LabelEncoder, MinMaxScaler
> %matplotlib inline

Then we load the data

# Define the path
path = '/kaggle/input/ml-zoomcamp-2024-competition/'

# Load the data
sales = pd.read_csv(f"{path}/sales.csv")
sales.drop(columns=['Unnamed: 0'], inplace=True)
stores = pd.read_csv(f"{path}/stores.csv")
stores.drop(columns=['Unnamed: 0'], inplace=True)
catalog = pd.read_csv(f"{path}/catalog.csv")
catalog.drop(columns=['Unnamed: 0'], inplace=True)

# Merge sales with store and catalog info for feature enrichment
sales = sales.merge(stores, on="store_id", how="left")
sales = sales.merge(catalog, on="item_id", how="left")

# Add time-based features
sales["date"] = pd.to_datetime(sales["date"])
sales["year"] = sales["date"].dt.year
sales["month"] = sales["date"].dt.month
sales["day"] = sales["date"].dt.day
sales["day_of_week"] = sales["date"].dt.dayofweek

# Select features and target
features = [
    'division', 'format', 'city', 'area', 'dept_name', 'class_name', 'subclass_name', 'item_type', 
    'weight_volume', 'weight_netto', 'fatness', 'year', 'month', 'day', 'day_of_week', 'quantity'
]

sales = sales[features]













<p align="center">
  <img src="./images/breast_cancer_classification.jpg">
</p>



## Problem Statement
Breast cancer is one of the most common cancers that affects women and people assigned female at birth (AFAB). It happens when cancerous cells in your breasts multiply and become tumors. About 80% of breast cancer cases are invasive, meaning a tumor may spread from your breast to other areas of your body. Breast cancer typically affects women age 50 and older, but it can also affect women and people AFAB who are younger than 50. Men and people assigned male at birth (AMAB) may also develop breast cancer. There are 3 main of breast cancer types and this helps to tailor treatment to be as effective as possible with the fewest possible side effects:

- Invasive (infiltrating) ductal carcinoma (IDC): This cancer starts in the milk ducts and spreads to nearby breast tissue.
- Lobular breast cancer: This breast cancer starts in the milk-producing glands (lobules) in the breast and often spreads to nearby breast tissue.
- Ductal carcinoma in situ (DCIS): Like IDC, this breast cancer starts in the milk ducts but the difference is DCIS doesn’t spread beyond the milk ducts.

This project aims to develop a robust breast cancer detection model using Convolutional Neural Networks (CNNs) to automate the analysis of mammogram images by identifying and classifying breast tumors accurately thereby potentially reducing the burden on radiologists and assisting in early detection efforts, which is crucial for successful treatment and improved survival rates.

This project was implemented as a requirement for the completion of [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) - a free course about Machine Learning.



## Exploratory Data Analysis
The dataset used in this project is the [Breast Cancer Dataset](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection) from kaggle. 

It consists of 3,383 mammogram images focused on breast tumours, annotated in a folder structure of 336 test, 2,372 train and 675 valid. The structure was modified by adding the valid folder to the train folder so that the data for the project consists of 336 test and 3,047 train.

The dataset consists of images with and without tumour which are labelled 1 and 0 respectively with the distribution shown below:

- No of images in  train:  0 - 2017  
- No of images in  train:  1 - 1030  

- No of images in test:  0 - 208  
- No of images in test:  1 - 128  

The diagrams below show the histogram of the class distribution, number of images per dataset and size distribution.

<p align="center">
  <img src="./images/dataset_images_distribution.jpg">
</p>

<p align="center">
  <img src="./images/dataset_images_count.jpg">
</p>

<p align="center">
  <img src="./images/dataset_size_distribution.jpg">
</p>

Finally, some samples of the dataset were drawn as shown below:

<p align="center">
  <img src="./images/sample_images.jpg">
</p>



## First trainings
After conducting the exploratory data analysis, 3 pre-trained deep convolutional neural network models were used to train the data for 10 epochs each with the aim of finding the one that will return the best mean test accuracy which will then be used to create the final model.

The results obtained with BATCH_SIZE = 64, DROP_RATE = 0.5, and LEARNING_RATE = 0.0001 are shown below:

The Results for InceptionV3
- Mean Train Accuracy: 0.704
- Mean Test Accuracy: 0.6503
- Mean Train Loss: 0.5663
- Mean Test Loss: 0.7455

The Results for ResNet101V2
- Mean Train Accuracy: 0.703
- Mean Test Accuracy: 0.6509
- Mean Train Loss: 0.5776
- Mean Test Loss: 0.6993

The Results for Xception
- Mean Train Accuracy: 0.7251
- Mean Test Accuracy: 0.6473
- Mean Train Loss: 0.5335
- Mean Test Loss: 0.7709

The plot of the run of each model are shown below:

### InceptionV3
<p align="center">
  <img src="./images/InceptionV3_result.jpg">
</p>

### ResNet101V2
<p align="center">
  <img src="./images/ResNet101V2_result.jpg">
</p>

### Xception
<p align="center">
  <img src="./images/Xception_result.jpg">
</p>

InceptionV3 was selected because its mean test accuracy is just 0.0006 less than ResNet101V2 and its runtime, which is important for sustainability, is half that of ResNet101V2. The diagrams below show the runtime of each model: 

<p align="center">
  <img src="./images/InceptionV3_runtime.jpg">
</p>

<p align="center">
  <img src="./images/ResNet101V2_runtime.jpg">
</p>

<p align="center">
  <img src="./images/Xception_runtime.jpg">
</p>



## Second trainings
After selecting the pre-trained model for the project, its parameters were tuned to get better results which are listed below:

InceptionV3 with BATCH_SIZE = 32
- Mean Train Accuracy: 0.706
- Mean Test Accuracy: 0.6542
- Mean Train Loss: 0.5677
- Mean Test Loss: 0.6335

InceptionV3 with DROP_RATE = 0.5
- Mean Train Accuracy: 0.7089
- Mean Test Accuracy: 0.6622
- Mean Train Loss: 0.5609
- Mean Test Loss: 0.6304

InceptionV3 with LEARNING_RATE = 0.00001
- Mean Train Accuracy: 0.701
- Mean Test Accuracy: 0.6506
- Mean Train Loss: 0.5706
- Mean Test Loss: 0.6987

All results were higher than the original and their plots are shown below respectively:

<p align="center">
  <img src="./images/InceptionV3_result_batch.jpg">
</p>

<p align="center">
  <img src="./images/InceptionV3_result_drop.jpg">
</p>

<p align="center">
  <img src="./images/InceptionV3_result_learing.jpg">
</p>



## Final training
The final training was conducted using all the tuning hyper-parameters and the results obtained are listed below:

InceptionV3 with BATCH_SIZE = 32, DROP_RATE = 0.5, LEARNING_RATE = 0.00001
Mean Train Accuracy: 0.715
Mean Test Accuracy: 0.6725
Mean Train Loss: 0.5475
Mean Test Loss: 0.6174

As expected, the result was much higher, and this can be attributed to the reasons below:

Firstly, smaller batch sizes can lead to more frequent updates, which may improve convergence speed thereby providing better accuracy, but can be computationally expensive and time-consuming. It requires less memory and can lead to better generalization due to less noise during the optimization process.

Secondly, dropout prevents the network from becoming too dependent on specific neurons, which improves the model's ability to generalize to new data, it forces the network to learn more generalized representations by preventing it from relying on specific features, it can be seen as training an ensemble of smaller networks, which helps the model learn more robust features.

Finally, the right learning rate will enable the model to converge on something useful while still training in a reasonable amount of time, this can be achieved by cyclical learning rate that involves letting the learning rate oscillate up and down during training, exponential decay that decreases the learning rate at an exponential rate, which can lead to faster convergence, and ReduceLROnPlateau that adjusts the learning rate when a plateau in model performance is detected. The plot of the run is shown below:

<p align="center">
  <img src="./images/InceptionV3_final.jpg">
</p>



## The model
The resultant model obtained from the final model was saved in keras format so that it can be hosted online for inferencing. The size of the model on disk is 274 MB which makes it impossible to host directly online and the solution to this problem is to enlist the use of tools that will help in solving this problem. Docker and kubernetes are the defacto tools used to solve this problem. 

Docker is a software platform that makes it possible to build, test, and deploy applications quickly. Docker packages software into standardized units called containers that provides all a software needs to run including libraries, system tools, code, and runtime. Using Docker enables quick deployment and scaling of applications into any environment and with a guarantee of the code running.

Kubernetes is an open-source software that allows the deployment and management of containerized applications at scale. Kubernetes manages clusters of cloud compute instances and runs containers on those instances with processes for deployment, maintenance, and scaling. Using Kubernetes, enables running any type of containerized applications using the same toolset on-premises and in the cloud.

The advantages of using docker and kubernetes together are scalability, high availability, portability, security, ease of use, reduced costs, improved agility. The image below shows the aggregated summary of the model.

 <p align="center">
  <img src="./images/InceptionV3_params.jpg">
</p>



## The project
After saving the model and deciciding on the tools for inferencing, the next task is to built the project by 
starting with the virtual environment followed by the required files in the sequence shown below:

> pip install pipenv
> pipenv install tensorflow==2.16.2
> pipenv install numpy==1.26.4
> pipenv install pillow flask gunicorn requests keras_image_helper
> pipenv install --dev notebook==7.2.1 ipywidgets jupyter pandas matplotlib seaborn

Then 