
from datetime import datetime
start = datetime.now()

import pandas as pd
import glob

print("Getting files list...")

bu = (glob.glob("data/bbc/business/*.txt"))


en = (glob.glob("data/bbc/entertainment/*.txt"))


po = (glob.glob("data/bbc/politics/*.txt"))


sp = (glob.glob("data/bbc/sport/*.txt"))


te = (glob.glob("data/bbc/tech/*.txt"))
category = []
filename =[]
title = []
content = []

def CreateData(cate):

    
    print("Creating "+cate+" data...")
    if(cate is 'business'):
        field = bu
    if(cate is 'entertainment'):
        field = en
    if(cate is 'politics'):
        field = po
    if(cate is 'sport'):
        field = sp
    if(cate is 'tech'):
        field = te
    for textfile in field:
        try :
            with open(textfile) as f:
                rawcontentarray = f.read().splitlines()
                rawcontent = ""
                title.append(rawcontentarray[0])
                for i in range(2,len(rawcontentarray)):
                    rawcontent+=rawcontentarray[i]
                content.append(rawcontent)
                category.append(cate)
                filename.append(textfile[textfile.rfind('/')+1:])
        except Exception as error:
            print(cate)
            print(rawcontentarray[0])
            print(textfile)
            print(rawcontent)
            print(filename[-1])
            print(title[-1])
            print(content[-1])
            print(error)

CreateData("business")
CreateData("entertainment")
CreateData("politics")
CreateData("sport")
CreateData("tech")
print("Cate\tFile\tTitle\tContent")
print(str(len(category))+"\t"+str(len(filename))+"\t"+str(len(title))+"\t"+str(len(content)))


if(len(category)==len(filename)==len(title)==len(content)):
    print("Create data successfully")
else:
    print("Number of data is not equal\n"+str(len(category))+"\t"+str(len(filename))+"\t"+str(len(title))+"\t"+str(len(content)))


print("Importing data...")
dataSet = list(zip(category,filename,title,content))
df = pd.DataFrame(data = dataSet, columns=['category', 'filename','title','content'])
print("Exporting data...")
df.to_csv('data.csv',index=False,header=True)
print("Done")

createdatatime = datetime.now()-start

df['category_id'] = df['category'].factorize()[0]

category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)


df.sample(5, random_state=0)

df.groupby('category').filename.count().plot.bar(ylim=0)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.content).toarray()
labels = df.category_id
features.shape


from sklearn.feature_selection import chi2
import numpy as np

N = 3
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

SAMPLE_SIZE = int(len(features) * 0.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
          fontdict=dict(fontsize=15))
plt.legend()


df[df.title.str.contains('Arsenal')]

dataexplorationtime = datetime.now()- createdatatime -start

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)

cv_df.groupby('model_name').accuracy.mean()



from sklearn.model_selection import train_test_split

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')



from IPython.display import display

for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 2:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['title', 'content']])
      print('')


model.fit(features, labels)

from sklearn.feature_selection import chi2

N = 5
for category, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(category))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))


trainingtime = datetime.now()-createdatatime-dataexplorationtime-start
df[df.content.str.lower().str.contains('news website')].category.value_counts()


texts = ["Hooli stock price soared after a dip in PiedPiper revenue growth.",
         "Captain Tsubasa scores a magnificent goal for the Japanese team.",
         "Merryweather mercenaries are sent on another mission, as government oversight groups call for new sanctions.",
         "Beyoncé releases a new album, tops the charts in all of south-east Asia!",
         "You won't guess what the latest trend in data analysis is!"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")
predicttime = datetime.now()-createdatatime-dataexplorationtime-trainingtime-start

print("Overall Time:\t" +str(datetime.now()-start))
print("Create Data:\t" + str(createdatatime))
print("Explore Data:\t" + str(dataexplorationtime))
print("Training:\t" + str(trainingtime))
print("Prediction Time:\t"+ str(predicttime))
