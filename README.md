# DEEP-LEARNING-PROJECT

"COMPANY":CODETECH IT SOLUTIONS

"NAME":G NAVEEN

"INTERN ID":CT04DF448

"DOMAIN":DATA SCIENCE

"DURATION": 4 WEEKS

"MENTOR":NEELA SANTOSH

I have choosen the concept of sentiment analysis which i have done on the reviews of imdb movie reviews.In the concept off sentiment anaylsis the model predicts whether the given review is positive, negative or neutral.The project was done in the following manner

Sentiment Analysis of IMDB Movie Reviews using LSTM in TensorFlow 


In today’s world of big data, making sense of text is more important than ever. One of the most useful tools for this is sentiment analysis—a process of determining whether a piece of text expresses a positive or negative emotion. Whether it's analyzing what people are saying about a new product, checking the tone of tweets, or interpreting customer feedback, sentiment analysis plays a huge role in helping businesses and developers make data-driven decisions.In this project, we focus on building a sentiment analysis model using deep learning, specifically using an LSTM (Long Short-Term Memory) network. The dataset we’ll work with is the IMDB movie reviews dataset, and we’ll use popular Python libraries like TensorFlow and Keras to implement the model. This end-to-end pipeline includes everything from preprocessing the data to training the model and saving the predictions.

The IMDB movie review dataset is a classic benchmark in Natural Language Processing (NLP). It contains 50,000 reviews of movies from the Internet Movie Database, labeled as either positive or negative. The dataset is already preprocessed: each review is turned into a sequence of integers, where each integer represents a specific word. This format makes it easier for the model to understand and learn from the data.
The dataset is evenly split into a training set and a test set, with 25,000 reviews in each. This balanced setup helps ensure that the model learns to generalize well and isn’t biased toward one label.

Steps involved:

1. Importing the Required Libraries
The first step is to import all the necessary tools. We use TensorFlow for building and training the neural network, Matplotlib to visualize the model’s performance, and Pandas to manage and export the final results in a clean, structured way. These libraries are the backbone of modern machine learning workflows in Python.

2. Loading and Preparing the Dataset
We use TensorFlow’s built-in imdb dataset, limiting the vocabulary to the top 10,000 most frequently used words. This is done to make the model simpler and faster to train while still retaining enough information to perform welll.Since the reviews vary in length, we pad them to a fixed size of 200 words. This ensures all inputs to the neural network are of the same shape, which is a requirement when feeding data into deep learning models.

3. Building the LSTM Model
At the heart of this project is the LSTM neural network, which is well-suited for processing sequences like sentences. We use a Sequential model with three main layers:
An Embedding layer that converts each word index into a 32-dimensional dense vector. This helps the model understand the semantic relationships between words.
An LSTM layer with 64 units, which allows the model to "remember" patterns and long-term dependencies in the data.
A Dense output layer with a sigmoid activation function, which outputs a probability between 0 and 1, representing the sentiment of the review.
The model is compiled with the Adam optimizer and binary cross-entropy loss, which are common and effective choices for binary classification tasks.

4. Training the Model
Next, the model is trained using the .fit() function. We train it for 5 epochs, meaning it will go through the entire training dataset 5 times. We also provide the test data as a validation set so we can track how well the model is performing on unseen data after each training round.This step is where the model actually learns. It adjusts its internal weights by comparing its predictions to the actual labels and minimizing the error using backpropagation.

5. Visualizing Accuracy
To understand how well the model is learning, we plot the training and validation accuracy over time. These plots help us see whether the model is improving, overfitting (memorizing the training data too much), or underfitting (not learning enough).By comparing both curves, we can evaluate the model’s generalization capabilities and decide whether further tuning is needed.

6. Making Predictions
Once the model is trained, we test its performance by making predictions on the first 100 reviews in the test set. The model gives us a probability score for each review. We then apply a threshold: if the score is above 0.5, we consider the review to be positive; otherwise, it's negative. This simple rule transforms raw probabilities into binary labels.

7. Saving Predictions to a CSV File
To keep a record of our results, we create a Pandas DataFrame with the predicted probabilities, predicted labels, and actual labels. This is saved to a file named imdb_predictions.csv, which can be easily opened in Excel or shared with others. It's a great way to review and analyze the model's output in a human-readable format.

This code represents a complete sentiment analysis pipeline using deep learning. From data preprocessing to model building, training, visualization, prediction, and exporting results, every step is handled efficiently. The use of an LSTM network makes it powerful in capturing the meaning and structure of sentences, especially in tasks where word order matters.


![Image](https://github.com/user-attachments/assets/565162a2-921d-413b-bf36-e94e5404934b)







