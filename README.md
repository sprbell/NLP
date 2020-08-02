# Text Classification with Logistic Regression

The goal of this assignment is to develop and test two text classification systems: 
  - Task 1: sentiment analysis, in particular to predict the sentiment of movie review, i.e. positive or negative (binary classification).
  - Task 2: topic classification, to predict whether a news article is about International issues, Sports or Business (multiclass classification).
  
For that purpose, you will implement:
 - Text processing methods for extracting Bag-Of-Word features, using **unigrams, bigrams and trigrams** to obtain vector representations of documents. Two vector weighting schemes should be tested: 
   - raw frequencies
   - tf.idf.
 - Binary Logistic Regression classifiers that will be able to accurately classify movie reviews trained with 
   - BOW-count (raw frequencies)
   - BOW-tfidf (tf.idf weighted) for Task 1.
 - Multiclass Logistic Regression classifiers that will be able to accurately classify news articles trained with
   - BOW-count (raw frequencies)
   - BOW-tfidf (tf.idf weighted) for Task 2.
 - The Stochastic Gradient Descent(SGD) algorithm to estimate the parameters of your Logistic Regression models. Your SGD algorithm should:
   - Minimise the Binary Cross-entropy loss function for Task 1.
   - Minimise the Categorical Cross-entropy loss function for Task 2.
   - Use L2 regularisation (both tasks).
   - Perform multiple passes (epochs) over the training data.
   - Randomise the order of training data after each pass.
   - Stop training if the difference between the current and previous validation loss is smaller than a threshold.
   - After each epoch print the training and development loss.
 - After training the LR models, plot the learning process (i.e. training and validation loss in each epoch) using a line plot. 
 - Model interpretability by showing the most important features for each class (i.e. most pos- itive/negative weights). Give the top 10 for each class and comment on whether they make sense. 
 
 # Text Classification with a Feedforward Network
 
 The goal of this assignment is to develop a Feedforward network for text classification. 

For that purpose, you will implement:

- Text processing methods for transforming raw text data into input vectors for your network
- A Feedforward network consisting of:
    - **One-hot** input layer mapping words into an **Embedding weight matrix**
    - **One hidden layer** computing the mean embedding vector of all words in input followed by a **ReLU activation function**
    - **Output layer** with a **softmax** activation.
- The Stochastic Gradient Descent (SGD) algorithm with **back-propagation** to learn the weights of your Neural network. Your algorithm should:
    - Use (and minimise) the **Categorical Cross-entropy loss** function
    - Perform a **Forward pass** to compute intermediate outputs
    - Perform a **Backward pass** to compute gradients and update all sets of weights
    - Implement and use **Dropout** after each hidden layer for regularisation
- Discuss how did you choose hyperparameters? You can tune the learning rate (hint: choose small values), embedding size {e.g. 50, 300, 500}, the dropout rate {e.g. 0.2, 0.5} and the learning rate. Please use tables or graphs to show training and validation performance for each hyperparam combination.
- After training the model, plot the learning process (i.e. training and validation loss in each epoch) using a line plot and report accuracy.
- Re-train your network by using pre-trained embeddings ([GloVe](https://nlp.stanford.edu/projects/glove/)) trained on large corpora. Instead of randomly initialising the embedding weights matrix, you should initialise it with the pre-trained weights. During training, you should not update them (i.e. weight freezing) and backprop should stop before computing gradients for updating embedding weights. Report results by performing hyperparameter tuning and plotting the learning process. Do you get better performance?

- **BONUS:** Extend you Feedforward network by adding more hidden layers (e.g. one more). How does it affect the performance? Note: You need to repeat hyperparameter tuning, but the number of combinations grows exponentially. Therefore, you need to choose a subset of all possible combinations
 
### Data 

The data you will use for Task 2 is a subset of the [AG News Corpus](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) and you can find it in the `./data_topic` folder in CSV format:

- `data_topic/train.csv`: contains 2,400 news articles, 800 for each class to be used for training.
- `data_topic/dev.csv`: contains 150 news articles, 50 for each class to be used for hyperparameter selection and monitoring the training process.
- `data_topic/test.csv`: contains 900 news articles, 300 for each class to be used for testing.

### Pre-trained Embeddings

You can download pre-trained GloVe embeddings trained on Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip). No need to unzip, the file is large.

### Save Memory

To save RAM, when you finish each experiment you can delete the weights of your network using `del W` followed by Python's garbage collector `gc.collect()`
 
 
 
 
