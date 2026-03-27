import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import time as t

t1=t.time()

data=pd.read_csv('spam.csv', encoding='latin-1')
data=data[['v1', 'v2']]
data.columns=['category', 'text']

data['category']=data['category'].map({'ham': 0, 'spam': 1})

messages=data['text']
labels=data['category']

msg_train, msg_test, lbl_train, lbl_test=train_test_split(messages, labels, test_size=0.2, random_state=7)

cv=CountVectorizer()
train_matrix=cv.fit_transform(msg_train)
test_matrix=cv.transform(msg_test)

nb_model=MultinomialNB()
nb_model.fit(train_matrix, lbl_train)

predictions=nb_model.predict(test_matrix)
acc=accuracy_score(lbl_test, predictions)
print(f"Model Accuracy: {acc*100:.2f}%")

t2=t.time()
print(f"{t2-t1: .4f} sec")

print("\n--- Spam Detector ---")
while True:
    u_input=input("\nType a message (or 'quit' to stop): ")
    if u_input.lower() == 'quit':
        break
    converted=cv.transform([u_input])
    output=nb_model.predict(converted)[0]
    print(f"\nVerdict: {'SPAM' if output == 1 else 'NOT SPAM'}")
