# BernoulliNB-and-ID3-algorithm


Assigmnemt for course: "**Artificial Intelligence**
<br>*Department of Computer Science, AUEB*


### Bernoulli Naive Bayes
This code implements a Naive Bayes classifier for sentiment analysis using the IMDB movie review dataset. Here's what it does:

* Imports necessary libraries such as TensorFlow, scikit-learn's classification_report, tabulate for creating tables, and matplotlib for plotting.
* Loads the IMDB dataset using TensorFlow and extracts the training and testing data.
* Decodes the sequences of integers representing words back into their original text form.
* Constructs a binary representation of the data where each word is represented by a binary value indicating its presence or absence in the review.
* Defines functions for training and applying the Naive Bayes classifier.
* Trains the Naive Bayes classifier on the training data and applies it to the testing data.
* Prints the classification report, which includes precision, recall, and F1-score for each class.
* Plots the accuracy, precision, recall, and F1-score as a function of the number of training files used.
* Displays the results in tabular form. <br>
Overall, this code performs sentiment analysis on movie reviews using a Naive Bayes classifier and evaluates its performance using various metrics and visualizations.

