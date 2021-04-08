# profile-based-information-retrieval
This project is developed to find a user whose interests fit the most to a given document. You may find the backgrounds and intention from a file called `profile_based_information_retrieval.pdf`.
# Installation
    pip -r install requirements.txt

# Start by running
    python main.py

# Refer to options
    usage: main.py [-h] [--rebuild REBUILD] [--retrain RETRAIN]
                   [--newuser NEWUSER]
    optional arguments:
      -h, --help         show this help message and exit
      --rebuild REBUILD  It scraps the dataset and retrains the machine learning
                         models.
      --retrain RETRAIN  It retrains the machine learning models.
      --newuser NEWUSER  You can add a new user and new interests.

# Acknowledgement
This project referred to [this article](https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
).
The machine learning models were built on the articles on the [Euronews](https://www.euronews.com).
