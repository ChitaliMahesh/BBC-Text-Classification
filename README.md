# BBC-Text-Classification

# Problem Statement

Given a dataset of news articles from different categories such as business, entertainment, politics, sport, and tech, the objective is to build a neural network model that can accurately classify new articles into one of these categories based on their text content.

# Data Scource

The dataset is taken from their website. The dataset consist of 2225 documents from bbc news website each labeled with one of the categories such as business,sports,tech,entertainment and politics.

# pre-process

In the preproces step first we given labels to the each category in the category feature using pandas pd.get_dummies function.Then splitted the dataset into training and testing  by using train_test_split function.After that I removed the stopwords and done the stemminng on the text.After  that we done tokkenization on the text dataset  by  considering 5000 most frequent words from the dataset.Finally, we use the pad_sequences method from Keras to pad the sequences to a fixed length, which is necessary for feeding the data into a neural network model.

# Model Architecture

The model consist of Sequential model with a LSTM layer.The embedding layer connsist of 5000 vocabulary and 16 embedding vectors for each word.After that I added a LSTM layer with 64 units.After that I added two dense layer with activationn function as relu. And in the last dense layer there are 5  units and activation function as softmax.

# Hyperparameters 
 The model is trained using Adam optimizer. The loss we  used here  is categorical  croos-entropy and the evaluation metrics used is accuracy.The batch size used here is 128  and the model  is trained for 30 epochs  with  validationn spit as 0.3.
 
 # Conclusion
 
 In this project we proposed approach for classifying BBC text using a sequential model with a embedding layer , LSTM layer and 3 dense layer. the model achieved an accuracy of 100 % on training data and 83%  accuracy  on validation data.
