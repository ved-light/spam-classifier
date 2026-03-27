# Spam Message Classifier

a simple ML project i made for my CSA2001 course at VIT Bhopal. it take a 
message as input and tells you if its spam or not.

## How it works

the model is trained on 5000+ real SMS messages which are already labeled as 
spam or ham (ham just mean not spam). it use Naive Bayes algorithm to learn 
patterns from these messages and then predict new ones.

basically it notices that words like "free", "win", "click" appear way more in 
spam messages than normal ones. so when you type something it checks for these 
pattern and give a verdict.

i also added a timer to see how long the model takes to train. usually its 
under 2 seconds which i thought was pretty fast.

## Requirements

need python installed with these libraries:

- pandas
- scikit-learn

to install them just run this in terminal:

pip install pandas scikit-learn

## Dataset

download the dataset from kaggle:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

put the spam.csv file in same folder as the python file otherwise it wont run.

## How to run

1. clone or download this repo
2. download dataset and put spam.csv in the project folder
3. run the file:

python spam_classifier.py

4. type any message when it ask and it will tell you spam or not
5. type 'quit' to exit

## Example

Type a message (or 'quit' to stop): Congratulations you won a free iPhone click here
Verdict: SPAM

Type a message (or 'quit' to stop): hey are you coming to class tomorrow
Verdict: NOT SPAM

## Results

model gets around 98-99% accuracy on test data which i think is pretty good 
for such a simple approach. the training itself only take 1-2 seconds.

## What i learned

honestly before this project i had no idea how ML even worked. most interesting 
part was understanding CountVectorizer — didn't knew that models cant read words 
directly, everything have to be converted to numbers first. Naive Bayes was also 
suprisingly simple once i understood it, it just calculates probability based on 
how often certain words appear in spam vs normal messages.
