## I. Introduction


This study was conducted to predict the "mpg" (miles per gallon) value, which determines the performance of automobiles, using the mpg dataset from the Seaborn library. The study utilized K-Nearest Neighbors (KNN), Random Forest, and Artificial Neural Network (ANN) models. The performance of these models was tested, and the best model was identified. Additionally, the selected best model was used to predict the mpg value for a specific car.

## II. Application Process

Data Processing and Cleaning

Dataset: The mpg dataset from the Seaborn library was used. Missing data was cleaned, and horsepower, acceleration, and weight were selected as independent variables, while mpg was selected as the dependent variable.

Training and Test Sets: The data was split into two groups, 90% for training and 10% for testing.

Model Training

KNN Model: The number of neighbors was set to 5, and the model was trained.

Random Forest Model: A model consisting of 100 trees was used.

ANN Model: An artificial neural network model with two hidden layers, each containing 50 neurons, was trained.

Model Testing

The performance of the K-Nearest Neighbors (KNN), Random Forest, and Artificial Neural Network (ANN) models was compared on the test set. The performance criteria used in this comparison are:

- Mean Squared Error (MSE): Measures the amount of error between predicted and actual values.

- R2 Score: Measures the model’s ability to explain the variations in the dependent variable (ranges from 0 to 1, with 1 being the best).
  
  ![image](https://github.com/user-attachments/assets/e5f687c8-2fe3-4942-87c3-6a6773effa24)

  According to the table above, the KNN model, which has the lowest MSE and the highest R2 Score values, has been identified as the most successful model for this dataset.


Predictions and Comparisons

The most successful model, KNN, was used to predict the fuel consumption (mpg) for a specific car. The car’s predicted features were:

horsepower = 130

acceleration = 13

weight = 3500

The KNN model predicted the mpg value for this car to be 17.84.

## III. Evaluation

The following conclusions were drawn from the study:

Model Performances:

The KNN model emerged as the most successful algorithm for this dataset and problem. The Random Forest model showed a close performance. The ANN model, however, needs more data or parameter optimization to perform better.


![image](https://github.com/user-attachments/assets/eab8e27b-9aea-4500-9dc2-8f599e73ddf2)



  ![image](https://github.com/user-attachments/assets/e92402a2-cfd5-412b-bee9-629fe480b97c)

