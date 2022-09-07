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
Seamless currently has about 4300 followers on Twitter, and has a relatively active account tweeting on average once per day. Using the Twitter analytics dashboard I collected a CSVs for each month starting in July 2022 and going back to their first tweet in June 2018. These CSV's were turned into pandas dataframes and appended together into a single table.

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
This leaves us with a dataframe looking like this:
<br>
<img width="1073" alt="Screen Shot 2022-09-06 at 5 42 00 PM" src="https://user-images.githubusercontent.com/13599213/188764104-a40f7eff-a354-4cd2-9aea-4e6a7d0b446c.png">
<br>
<h2> Exploration </h2>
<br>
My first step in exploratory data analysis was just to see if there were any obvious linear relationships between the variables. I did this by calculating the r-squared, or, what percent of the variation in engagement rate can be explained by the other variables. Let's start with time, since it's reasonable to hypothesize that tweets during active hours would be more popular than ones during off hours like the middle of the night. <br> <br>

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
0.0074, which is <b>technically</b> better, but that's not saying much.
<br>
This seems like a good time to take a look at our target variable and see what's going on.<br><br>

```python
df['engagements'].hist(bins=20)
```
<br>
<img width="391" alt="Screen Shot 2022-09-06 at 6 20 52 PM" src="https://user-images.githubusercontent.com/13599213/188767855-660605fe-ed72-4eb7-a894-4d7de45eefb5.png">
<br>
Well that might explain some of it. Engagement is incredibly skewed. It looks like nearly all tweets have less than 100 total engagements. Specifically, the mean of engagements is ~42 while the standard deviation is over 200. The skew is 24.85, which means only 20% of tweets are above average.
<br>
It's not wonder finding correlations is hard, most tweets don't have enough engagement to say anything in particular about them.
<br>


<br>
<h2> Feature Engineering </h2>
<br>
lorem ipsum
<br>
<h2> Modeling </h2>
<br>
lorem ipsum
<br>
<h2> Conclusion </h2>
<br>
lorem ipsum
<br>
<h3> Acknowledgements </h3>
<br>
lorem ipsum
<br>
