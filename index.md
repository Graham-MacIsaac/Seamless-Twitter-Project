<h1>Seamless Bay Area Tweet Analysis</h1>
<br>
<h2> Background </h2>
<br>
The San Francisco Bay Area, or officially speaking the "San Jose-San Francisco-Oakland, CA Combined Statistical Area" is the 5th largest metropolitan area in the United States, with a population of 9.7 million people, and 100+ cities. It might not seem too surprising to learn it takes 27 different public transit agencies cover this wide area.
<br><br>
That's a shame, because it SHOULD be surprising. New York City with it's 20 million people, Washington DC + Baltimore with 10 million, and Chicago with 9.8, have a grand total of 15 transit agencies between them. I wouldn't be surprised if the SF Bay Area had more public transit organizations per capita than any urban area on the planet.
<br><br>
This is a pretty absurd situation to be in, and results in exactly as much confusion and waste as one would expect. To that end Seamless Bay Area was formed as a 501(c)4 nonprofit to be, in their own words: "a not-for-profit project whose mission is to transform the Bay Area’s fragmented and inconvenient public transit into a world-class, unified, equitable, and widely-used system by building a diverse movement for change and promoting policy reforms".
<br><br>
To help them in pursuit of this laudable goal, I volunteered my data science skills to help analyze their online messaging, and since Twitter is their primary outlets of social media outreach that seemed like a high impact place to begin.
<br><br>
<h2> Overview </h2>
<br>
The first step was to collect the data, which I did through the Twitter analytics dashboard. Next was to explore the data and see if there were any obvious patterns, and get a basic sense of how the data was distributed. Once all the data was collected, I trained a classification model to put tweets into two buckets based on engagement rate (aka, how many people were liking/replying/retweeting/etc. compared to the number of people seeing that tweet) either high (above average) or low (below average). I took 80% of the data to train the model with (stratifying by class since there were roughly twice as many low engagement tweets as high engagement ones) and tested it on the other 20%.
<br><br>
To follow that up I performed some feature engineering to extract elements that could be used in a linear regression, and to make some more human readible suggestions about what kinds of things Seamless could do with their Tweets to maximize engagement.
<br><br>
All code was written in Python.
<br><br>
<h2> Cleaning </h2>
<br><br>
Seamless currently has about 4300 followers on Twitter, and has a relatively active account tweeting on average once per day. Using the Twitter analytics dashboard I collected a CSVs for each month starting in July 2022 and going back to their first tweet in June 2018. These CSV's were turned into pandas dataframes and appended together into a single table. First we need to load all libraries we'll be using for this project. <br>

```python
import pandas as pd
import numpy as np
import re
import os
import statistics as stat
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random; random.seed(53)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
```
<br>
Now we're ready to load the data. <br>

```python
  #get a list of filenames
  labels = os.listdir('{filepath}/tweets')
  #each month of tweets is in a seperate .csv, so we combine them together
  csvs = []
  for x in labels:
    csvs.append(pd.read_csv('{filepath}/tweets' + x ))
  df = pd.concat(csvs)
```
<br>
However there were a bunch of "promoted" columns that looked like they might be empty, which would make sense since Seamless had an ad budget of $0 and wasn't promoting anything. However just to be sure I looked at every row in those columns to make sure they were in fact empty.
<br><br>

```python
  # Every column from 23 to the end is a "promoted" column, which we suspect has no data
  promoted_cols = df.iloc[:,22:]
  # Loop through the columns, comparing the sum of all comparisons to the columns length
  temp1 = []
  for y in range(len(promoted_cols.columns)):
    temp = []
    for x in range(len(promoted_cols)):
      temp.append(promoted_cols.iloc[x,y] == '-')
    temp1.append(sum(temp) == len(promoted_cols))
   sum(temp1) == len(promoted_cols.columns)
```
<br>
This gave a result of "True" which meant all those columns are empty and we can safetly drop them. We're also going to drop the first two columns because Tweet ID and Permalink are just identifiers we won't be using.
<br>

```python
df = df.iloc[:,2:12]
```

<br>
This leaves us with a dataframe looking like this:<br><br>

<img width="1073" alt="Screen Shot 2022-09-06 at 5 42 00 PM" src="https://user-images.githubusercontent.com/13599213/188764104-a40f7eff-a354-4cd2-9aea-4e6a7d0b446c.png">

<br>
<h2> Exploration </h2>
<br>
My first step in exploratory data analysis was just to see if there were any obvious linear relationships between the engagement and the only other independent variable provided in the original data - time. I did this by calculating the r-squared, or, what percent of the variation in engagement rate can be explained by time. 
<br>
We need to take the string that's been provided and convert it to a datetime object first<br><br>

```python
from datetime import datetime
temp = []
for x in range(len(df['time'])):
    # It's in a weird format that datetime doesn't accomodate, so we'll need to remove the last
    # 6 characters of the string and replace it with an appropriate seconds identifier
    temp.append(datetime.strptime(str(df['time'].iloc[x][0:-6] + ':00'), '%Y-%m-%d %H:%M:%S'))
df['time'] = temp
```
<br>
It's reasonable to hypothesize that tweets during active hours would be more popular than ones during off hours like the middle of the night, but let's see if that's actually the case. <br> <br>

```python
#check the relationship between hour of the day and engagements
model = LinearRegression()
a = df['hour'].to_numpy()
x = a.reshape(-1, 1)
y = df['engagements']
model.fit(x,y)
r_sq = model.score(x, y)
print(r_aq)
```
This gave us a result of 0.000866, which is pretty terrible. That means only ~0.09% of the variation in engagement rate is explained by what hour of a date a tweet was made.
<br>
Maybe tweet length will be better. <br><br>

```python
# AVERAGE NUMBER OF WORDS
y = []
for x in df['tweet words']:
    y.append(len(x))
df['tweet length'] = y
#check the relationship between hour of the day and engagements
model = LinearRegression()
a = df['tweet length'].to_numpy()
x = a.reshape(-1, 1)
y = df['engagements']
model.fit(x,y)
r_sq = model.score(x, y)
print(r_sq)
```
0.0074, which is <b>technically</b> better, but that's not saying much. This seems like a good time to take a look at our target variable and see what's going on.<br><br>

```python
df['engagements'].hist(bins=20)
```
<br>
<img width="391" alt="Screen Shot 2022-09-06 at 6 20 52 PM" src="https://user-images.githubusercontent.com/13599213/188767855-660605fe-ed72-4eb7-a894-4d7de45eefb5.png">
<br>
Well that might explain some of it. Engagement is incredibly skewed. It looks like nearly all tweets have less than 100 total engagements. Specifically, the mean of engagements is ~42 while the standard deviation is over 200. The skew is 24.85, which means only 20% of tweets are above average.
<br>
I thought that perhaps this could be a problem of the individual components of engagement not having the same disstribution (as a reminder these are retweets, likes, replies, url clicks and profile clicks). I normalized each of the columns via maximum absolute scaling so that each value is between -1 and 1 to make direct comparison more legible. Using the follwing formula: <br><br>

<img width="218" alt="Screen Shot 2022-09-06 at 11 12 45 PM" src="https://user-images.githubusercontent.com/13599213/188802118-68cfbcdb-5228-42a1-bbe8-dc59728d6fc1.png">
<br>
Calculated with:<br><br>


```python
stat.stdev(sorted_hi_value_tweets['{variable name}'] / sorted_hi_value_tweets['{variable name}'].abs().max())
```

<ul>
  <li>Retweets - 0.076</li>
  <li>Likes - 0.088</li>
  <li>Replies - 0.106</li>
  <li>Profile Clicks - 0.072</li>
  <li>URL Clicks - 0.069</li>
</ul>
<br>
Replies has the largest standard deviation, which isn't surprising since so many tweets have a tiny number of them.
<br>
It's not wonder finding correlations is hard, most tweets don't have enough engagement to say anything in particular about them. But it does mean that there are some extreme outliers we can look at that might tell us something about what identifies a very successful tweet. I took a look at that top 20% of both engagement and engagement rate to see if there were any obvious takeaways.
<br>
Interestingly - the two were actually very different. The top 30 tweets by engagement all featured either time sensitive calls to action (vote on X immediately, come to Y event tomorrow, etc.) or had links to maps. However the top 30 tweets by engagement were almost entirely congratulations or thanks to other accounts (ex: @twitter_user Thanks for your support! 🙏🚆🚍).
<br>


<br>
<h2> Feature Engineering </h2>
<br>
Before the data can be used for modeling we'll have to go through some pre-processing. We need to perform all the feature engineering that I suspect will be necessary for the modeling step. Specifically, I want to identify links/attached media, @replies (when the tweet references another twitter account), calls to action, and sentiment score. Let's do them in that order, starting with links. <br>

```python
#add a space to the end of every tweet so we can find links at the end of tweets
temp = []
for x in range(len(df)):
    temp.append(df['Tweet text'][x] + ' ')
df['Tweet text'] = temp

#find every sub-string that's "https:// + some characters + a space"
links = []
for x in range(len(df)):
    a = re.findall(r'https://.* ', df['Tweet text'].iloc[x])
    links.append(a)
df['links'] = links

#do a bunch of annoying cleaning so that each item is a nice list of links
temp = []

for x in range(len(df)):
    temp.append(re.split('\s', str(df['links'][x])))

for x in range(len(temp)):
    temp[x] = temp[x][0:-1]

for x in range(len(temp)):
    temp[x] = re.sub('\[', '', str(temp[x]))
    
for x in range(len(temp)):
    temp[x] = re.sub('\]', '', str(temp[x]))
    
for x in range(len(temp)):
    temp[x] = re.sub('\'', '', str(temp[x]))

for x in range(len(temp)):
    temp[x] = re.sub('\"', '', str(temp[x]))

df['links'] = temp
```
<br>
This gives us a list which looks like this:
<br>
<img width="376" alt="Screen Shot 2022-09-07 at 1 02 58 AM" src="https://user-images.githubusercontent.com/13599213/188824195-434025c5-1880-4f0c-9859-46b48b04cc59.png">
<br>
Next is replies (as in, tweets from Seamless that mention another Twitter account). The idea is to build a columns of dummy variables that are 0 if that tweet doesn't contain a mention of a specific account and a 1 if it does. Bear with me, this took a lot of wrangling. <br><br>

```python
#split the tweets into individual words
replies = []
for x in range(len(df)):
    a = re.split(' ', df['Tweet text'].iloc[x])
    replies.append(a)
df['replies_sentance'] = replies

#find all the replies, aka sub-strings starting with @
temp3 = []
for z in range(len(df)):
    temp = []
    for x in df['replies_sentance'][z]:
        temp.append(re.findall(r'@.*', x))
    temp2 = []
    for y in temp:
        if len(y) > 0:
            temp2.append(y)
    temp3.append(temp2)
df['replies'] = temp3

#put it into a list
temp = []
for x in df['replies']:
    temp.append(list(x))
#I couldn't get the dummies to work further down so I converted it into a string, and then split
#it again
temp = []
for x in df['replies']:
    if len(x) > 0:
        a = re.sub('\[', '', str(x))
        b = re.sub('\]', '', a)
        c = re.sub('\'', '', b)
        d = re.sub('\,', '', c)
        temp.append(re.split(' ', d))
    else:
        temp.append('')
#split each of the first 5 replies into a seperate column, so that dummies can be made
rep = []
rep1 = []
rep2 = []
rep3 = []

for x in range(len(temp)):
    if len(temp[x]) == 0:
        rep.append('')
        rep1.append('')
        rep2.append('')
        rep3.append('')
    if len(temp[x]) == 1:
        rep.append(temp[x][0])
        rep1.append('')
        rep2.append('')
        rep3.append('')
    if len(temp[x]) == 2:
        rep.append(temp[x][0])
        rep1.append(temp[x][1])
        rep2.append('')
        rep3.append('')
    if len(temp[x]) == 3:
        rep.append(temp[x][0])
        rep1.append(temp[x][1])
        rep2.append(temp[x][2])
        rep3.append('')
    if len(temp[x]) >= 4:
        rep.append(temp[x][0])
        rep1.append(temp[x][1])
        rep2.append(temp[x][2])
        rep3.append(temp[x][3])

#build a dataframe that we'll use for the dummies
df1 = pd.DataFrame()
df1['engagements'] = df['engagements']
df1['rep'] = rep
df1['rep1'] = rep1
df1['rep2'] = rep2
df1['rep3'] = rep3
df1 = pd.get_dummies(df1)
print(df1)
```
<br>
Here's what that looks like:
<br>
<img width="763" alt="Screen Shot 2022-09-07 at 1 07 04 AM" src="https://user-images.githubusercontent.com/13599213/188825093-9d63306a-54fb-4cc6-9e48-eb5b5bcb7735.png">
<br>
Moving on to the last feature I want to create: sentiment score. This works by getting a list of positive and negative words, then comparing each tweet and assigning it a score from -1 to 1 based on how many (if any) of those words it has. This list was downloaded from Kaggle and can be found here:
https://www.kaggle.com/datasets/mukulkirti/positive-and-negative-word-listrar<br>

```python
words = pd.read_excel('/Users/grahamsmith/Documents/SpringboardWork/Positive and Negative Word List.xlsx')

# negative words first
temp = []
for x in words['Negative Sense Word List']:
    temp.append(str(x))
words['Negative Sense Words'] = temp
temp = []
for x in df['tweet words']:
    temp.append([ele for ele in list(words['Negative Sense Words']) if(ele in str(x))])
df['Negative words'] = temp

# then positive words
temp = []
for x in words['Positive Sense Word List']:
    temp.append(str(x))
words['Positive Sense Words'] = temp
temp = []
for x in df['tweet words']:
    temp.append([ele for ele in list(words['Positive Sense Words']) if(ele in str(x))])
df['Positive words'] = temp
```
<br>
Sentiment score is calculated by subtracting the number of negative words in the tweet from the number of positive words and dividing it by the total number of words to find a ratio of positive:negative.<br>

```python
temp = []
for x in range(len(df)):
    temp.append((len(df['Positive words'][x])/len(df['Tweet text'][x])) - (len(df['Negative words'][x])/len(df['Tweet text'][x])))
df['Sentiment Score'] = temp
```
<br>
Let's take date and split it into the individual time components (day/hour/minute)<br>

```python
let's make hour and minute their own columns for the regression
#for some inexplicable reason, to_csv reverts datetime objects to strings so it needs to be converted again
from datetime import datetime
temp = []
for x in range(len(df['time'])):
    temp.append(datetime.strptime(str(df['time'].iloc[x]), '%Y-%m-%d %H:%M:%S'))
df['time'] = temp

temp = []
for x in df['time']:
    temp.append(x.hour)
df['hour'] = temp

temp = []
for x in df['time']:
    temp.append(x.minute)
df['minute'] = temp

temp = []
for x in df['time']:
    temp.append(x.isoweekday())
df['day'] = temp
```
<br>
At this point we've identified several potentially important features and extracted them from the tweet text data, including links, sentiment, and replies. At this point we've also performed some baseline regressions with time and tweet length, the best of which only had an R^2 of 0.023, which is not especially powerful. We're ready to get to modeling.
<br>
<h2> Modeling </h2>
<br>
The first order of business is to build some models to predict engagement rate from tweet text. The question can be approached as a simple classification problem, where the label is “high engagement” (1) or “low engagement” (0). In this case I'm defining "high engagement" as an engagement rate above the mean. We’ll compare the results of using two models, each paired with the Multinomial Naive Bayes classifier.
<Br>
The first model is Count vectorizer, which is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
<br>
The second model is a slightly more complicated version called TFIDF, or the mouthful Term Frequency Inverse Document Frequency, which works by proportionally increasing the number of times a word appears in the document but is counterbalanced by the number of documents in which it is present. In other words, it finds words which are common in one class but not the other.
<br>
<em>A few notes on the hyperparameters: min_df and max_df Ignore terms that have a document frequency higher than 90% (very frequent), and lower than the 5% (highly infrequent), this has a similar effect of the stopwords in that it pulls out useless words.</em><br>

I attempted an ngram_range of (1,3) - meaning the vectorizers will incorporate bi- and tri-grams alongside single words, but it made almost literally no difference so I removed it. I felt unnessesary hyperparameters remove clarity without adding any additional information, although they do make your model feel cooler.<br>

Let's create our classes<br>

```python
mean = df['engagement rate'].mean()
df['target'] = np.where((df['engagement rate'] > mean), 1, 0)
```
<br>
We should check the proportion of classes in case we need to stratify our train/test sets<br>

```python
print('the proportion of our classes is: ' + str(len( df[df['engagement rate'] > mean] )/ len(df)))
```

<br> 
<h2> Conclusion </h2>
<br>
lorem ipsum
<br>
<h2> Acknowledgements </h2>
<br>
lorem ipsum
<br>
