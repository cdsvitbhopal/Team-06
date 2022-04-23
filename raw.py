import pandas as pd
import numpy as np
import pickle
dataset = pd.read_csv('balanced_reviews.csv')
dataset.dropna(inplace=True)
dataset = dataset[dataset['overall'] != 3]
dataset['Positivity'] = np.where(dataset['overall'] > 3, 1,0)
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(dataset['reviewText'], dataset['Positivity'], test_size=0.25, random_state=4)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5).fit(features_train)
features_vect_data = vect.transform(features_train)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,accuracy_score
model = LogisticRegression()
model.fit(features_vect_data, labels_train)
prediction = model.predict(vect.transform(features_test))
print(accuracy_score(prediction,labels_test)*100,"%")
pickle.dump(model, open('model_data_2.pkl', 'wb'))
pickle.dump(vect.vocabulary_, open('features_data_2.pkl', 'wb'))