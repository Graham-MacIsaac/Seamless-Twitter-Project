<script>
w3CodeColor(document.getElementById("codeBlock"));

function w3CodeColor(elmnt) {
  // click "Try it Yourself" to see the JavaScript...
}
</script>

<div id="codeBlock">
  &lt;!DOCTYPE html&gt;<br>
  &lt;html&gt;<br>
  &lt;body&gt;<br>
  <br>
  &lt;h1&gt;Testing an HTML Syntax Highlighter&lt;/h2&gt;<br>
  &lt;p&gt;Hello world!&lt;/p&gt;<br>
  &lt;a href="https://www.w3schools.com"&gt;Back to School&lt;/a&gt;<br>
  <br>
  &lt;/body&gt;<br>
  &lt;/html&gt;
</div>

<h1>Seamless Bay Area Twitter Analysis</h1>

<h2> Background </h2>

The San Francisco Bay Area, or officially speaking the "San Jose-San Francisco-Oakland, CA Combined Statistical Area" is the 5th largest metropolitan area in the United States, with a population of 9.7 million people, and 100+ cities. It might not seem too surprising to learn it takes 27 different public transit agencies cover this wide area.
<br>
That's a shame, because it SHOULD be surprising. New York City with it's 20 million people, Washington DC + Baltimore with 10 million, and Chicago with 9.8, have a grand total of 15 transit agencies between them. I wouldn't be surprised if the SF Bay Area had more public transit organizations per capita than any urban area on the planet.
<br>
This is a pretty absurd situation to be in, and results in exactly as much confusion and waste as one would expect. To that end Seamless Bay Area was formed as a 501(c)4 nonprofit to be, in their own words: "a not-for-profit project whose mission is to transform the Bay Area’s fragmented and inconvenient public transit into a world-class, unified, equitable, and widely-used system by building a diverse movement for change and promoting policy reforms".
<br>
To help them in pursuit of this laudable goal, I volunteered my data science skills to help analyze their online messaging, and since Twitter is their primary outlets of social media outreach that seemed like a high impact place to begin.

<h2> Overview </h2>

The first step was to collect the data, which I did through the Twitter analytics dashboard. Next was to explore the data and see if there were any obvious patterns, and get a basic sense of how the data was distributed. Once all the data was collected, I trained a classification model to put tweets into two buckets based on engagement rate (aka, how many people were liking/replying/retweeting/etc. compared to the number of people seeing that tweet) either high (above average) or low (below average). I took 80% of the data to train the model with (stratifying by class since there were roughly twice as many low engagement tweets as high engagement ones) and tested it on the other 20%.

To follow that up I performed some feature engineering to extract elements that could be used in a linear regression, and to make some more human readible suggestions about what kinds of things Seamless could do with their Tweets to maximize engagement.

All code was written in Python.

<h2> Cleaning </h2>

Seamless currently has about 4300 followers on Twitter, and has a relatively active account tweeting on average once per day. Using the Twitter analytics dashboard I collected a CSVs for each month starting in July 2022 and going back to their first tweet in June 2018. These CSV's were turned into pandas dataframes and appended together into a single table.
<br>
<pre class="highlight">
  <code>
  <div id="codeBlock">
  #get a list of filenames
  labels = os.listdir('{filepath}/tweets')
  #each month of tweets is in a seperate .csv, so we combine them together
  csvs = []
  for x in labels:
    csvs.append(pd.read_csv('{filepath}/tweets' + x ))
  df = pd.concat(csvs)
  </div>
  </code>
</pre>
<br>
lorem ipsum

<h2> Exploration </h2>

lorem ipsum

<h2> Modeling </h2>

lorem ipsum

<h2> Conclusion </h2>
