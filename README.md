PROJECT OVERVIEW

With this project I have tried to make a system/model for diagnosing heart disease. 

Since the dataset contains only raw signals, I searched for information about certain diseases, and after cleaning, analyzing, 
and saving the characteristics of each signal, I diagnosed a "possible disease" for each signal.
With this, we could say that I have created a rule-based model, which could be optimal as long as the diagnostic rules are correct.
But I wanted to do something more.

With this rule-based model, a "possible disease" is diagnosed for each signal of each patient, and I have created a diagnostic model 
using a machine learning technique, XGBOOST (parecido al random forest).

Once each signal was cleaned, I saved its features in a dictionary, but this proved to be a problem because, to create the machine learning model, 
I needed a dataframe with unique values, so I had to settle for creating a dataframe of the average of each feature.

Another problem was that I had many signals diagnosed as a few diseases, and very few of others, so I had to balance that number using randomSampler.
In the end, I was only able to diagnose 5/7 diseases, and some were found to be 100% accurate due to this randomSampler (overfitting occurred with so few samples of those diseases).


HOW TO RUN IT

The main file is in the path folder. It's called run_analysis.ipynb. This file uses functions from src/utils, py to perform operations, and the purpose of each code block is indicated above.


In the notebooks folder, there is a file that can be used to view two examples of electrocardiograms. To do this, you must change the path in that same code block.



