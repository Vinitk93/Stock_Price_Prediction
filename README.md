# STOCK PRICE PREDICTION USING MACHINE LEARNING

Doing analysis on stock price prediction would help me to know the future rates based on the previous stock price available. This prediction would give me better picture whether to buy the share or not. The company that I have worked on is AT&T Communications. The Machine Learning algorithm that I have used is LSTM (Long Short Term Memory). With the help of LSTM, I was able to identify whether the stock price will increase or decrease. It predicts one day ahead value based on the recent 60 days.

# TABLE OF CONTENTS

   [Architecture](#architecture)
   
   [Importing Libraries](#importing-libraries)
   
   [Reshaping the data](#reshaping-the-data)
   
   [Model Training](#model-training)
   
       
 ## Architecture
 
 1. The data set is split into training data set and test data set.
 2. LSTM layers are added in order to increase the accuracy of the model.
 3. The model is compiled using Root Mean Squared Error (RMSE).
 4. The model is trained and tested using testing data.
 5. Model is pre-processed. The test data used is same as training data. 
 
 
 ## Importing Libraries
 
 One of the most popular programming languages for this task is Python. One of the main reason is its vast collection of libraries. Python libraries that are used in this project are:
 
•	Math
•	Pandas
•	Numpy
•	Scikit-learn
•	Keras
•	Matplotlib
•	Tensorflow


 ## Reshaping the data
 
•	The new dataframe is created with only "Close" column and then converted it into numpy array.

•	Later, the 80% of data is used for training the model. 

•	The trained data is then scaled before giving it to Neural Network Model.

•	The scaled data is splitted into training data sets namely 'x_train' and 'y_train'.Both the training data sets are converted to numpy arrays.

•	Since the LSTM model can take only three-dimensions, the data is reshaped into required format from two-dimensional array.

        
 ## Model Training
   
•	Build the LSTM model: In order to increase the accuracy of the model, more LSTM layers are added.

•	Compile the model: The model is compiled. Here, Optimizer is used to improve loss function which further determines the performance of the model on training.

•	Train the model: The model is trained using training data sets and setting an epoch.

•	Reshaping the data: The Test data set is reshaped similar to the training dataset which is mentioned in the [Reshaping the data](#reshaping-the-data) above.

•	Prediction: Models Predicted stock value is achieved.

•	Evaluation: Model is evaluated using Root Mean Squared Error(RMSE).
 
