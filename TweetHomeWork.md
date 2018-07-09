

```python
# Dependencies
import tweepy
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#style.use('ggplot')

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key,
                    consumer_secret,
                    access_token,
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target User Account
target_user = ("@BBCNews", "@CNN", "@CBS","@FOXNEWS","@nytimes")

# set counter
counter = 1

# List
senti_results_list = []

# Loop through each user
for user in target_user:

    # Loop through 10 pages of tweets (total 100 tweets)
 
        public_tweets = api.user_timeline(user, count =100)

        # Loop through all tweets
        for tweet in public_tweets:

            # Run Vader Analysis on each tweet
            polarity = analyzer.polarity_scores(tweet["text"])
            compound = polarity["compound"]
            pos = polarity["pos"]
            neu = polarity["neu"]
            neg = polarity["neg"]
            tweets_ago = counter

# Add into list each tweet
            senti_results_list.append({"Airline_user": user,
                                         "Date":tweet["created_at"],
                                         "compound": compound,
                                         "Positive": pos,
                                         "Negative": neg,
                                         "Neutral": neg,
                                         "Tweets Ago": counter,
                                         "Tweet Text": tweet['text']
                                         })
# increase counter
            counter += 1
    
```


```python
senti_results_list_df = pd.DataFrame.from_dict(senti_results_list)
senti_results_list_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Airline_user</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet Text</th>
      <th>Tweets Ago</th>
      <th>compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCNews</td>
      <td>Mon Jul 09 02:28:18 +0000 2018</td>
      <td>0.209</td>
      <td>0.209</td>
      <td>0.182</td>
      <td>Maedeh Hojabri: Iran women dance in support of...</td>
      <td>1</td>
      <td>-0.1027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCNews</td>
      <td>Mon Jul 09 02:10:46 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Government outsourcing approach flawed, say MP...</td>
      <td>2</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCNews</td>
      <td>Mon Jul 09 01:23:33 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>How Trump's UK visit will be different to thos...</td>
      <td>3</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCNews</td>
      <td>Mon Jul 09 01:21:11 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.263</td>
      <td>The polio survivor who became a healthcare bos...</td>
      <td>4</td>
      <td>0.3612</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCNews</td>
      <td>Mon Jul 09 01:10:54 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Chris Moore: The man who's photographed 60 yea...</td>
      <td>5</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>@BBCNews</td>
      <td>Mon Jul 09 00:43:28 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Prince Louis's christening to take place in th...</td>
      <td>6</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>@BBCNews</td>
      <td>Mon Jul 09 00:18:07 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.248</td>
      <td>Thai cave rescue: Remaining boys wait for oper...</td>
      <td>7</td>
      <td>0.5106</td>
    </tr>
    <tr>
      <th>7</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 22:55:01 +0000 2018</td>
      <td>0.315</td>
      <td>0.315</td>
      <td>0.000</td>
      <td>Brexit Secretary David Davis resigns https://t...</td>
      <td>8</td>
      <td>-0.3182</td>
    </tr>
    <tr>
      <th>8</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 22:37:37 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.268</td>
      <td>Newspaper headlines: Novichok mum dies and Tha...</td>
      <td>9</td>
      <td>0.5106</td>
    </tr>
    <tr>
      <th>9</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 22:00:38 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.156</td>
      <td>Wimbledon 2018: Roger Federer, Serena Williams...</td>
      <td>10</td>
      <td>0.3400</td>
    </tr>
    <tr>
      <th>10</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 21:58:03 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Candlelit vigil on Bute for 'island angel' Ale...</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 20:56:52 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.231</td>
      <td>Justin Bieber 'engaged to Hailey Baldwin', US ...</td>
      <td>12</td>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>12</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 20:54:34 +0000 2018</td>
      <td>0.103</td>
      <td>0.103</td>
      <td>0.000</td>
      <td>World Cup 2018: Croatia can deal with Harry Ka...</td>
      <td>13</td>
      <td>-0.1531</td>
    </tr>
    <tr>
      <th>13</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 19:35:31 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>BP fuel stations 'can't take card payments' ht...</td>
      <td>14</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 19:23:17 +0000 2018</td>
      <td>0.281</td>
      <td>0.281</td>
      <td>0.000</td>
      <td>Ibiza death: British teen 'dies after being pu...</td>
      <td>15</td>
      <td>-0.5994</td>
    </tr>
    <tr>
      <th>15</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 17:23:19 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>World Cup 2018: 'Idiots' put dampener on celeb...</td>
      <td>16</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 17:15:03 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.195</td>
      <td>The waistcoat is soooo this season, thanks to ...</td>
      <td>17</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>17</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 16:43:30 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.241</td>
      <td>England v India: Rohit Sharma's unbeaten centu...</td>
      <td>18</td>
      <td>0.5859</td>
    </tr>
    <tr>
      <th>18</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 16:21:07 +0000 2018</td>
      <td>0.225</td>
      <td>0.225</td>
      <td>0.000</td>
      <td>Lewis Hamilton's fear for the future of young ...</td>
      <td>19</td>
      <td>-0.4939</td>
    </tr>
    <tr>
      <th>19</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 16:18:33 +0000 2018</td>
      <td>0.308</td>
      <td>0.308</td>
      <td>0.224</td>
      <td>Pride in London sorry after anti-trans protest...</td>
      <td>20</td>
      <td>0.0258</td>
    </tr>
    <tr>
      <th>20</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 16:04:35 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.236</td>
      <td>Tour de France: Peter Sagan wins stage two to ...</td>
      <td>21</td>
      <td>0.5719</td>
    </tr>
    <tr>
      <th>21</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 16:01:51 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.278</td>
      <td>Judge orders Brazil's Lula freed on appeal htt...</td>
      <td>22</td>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>22</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 15:32:47 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Carlisle tin containing chocolate survives 118...</td>
      <td>23</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 15:08:43 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.231</td>
      <td>World Cup: Here's what England looked like las...</td>
      <td>24</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>24</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 15:01:13 +0000 2018</td>
      <td>0.273</td>
      <td>0.273</td>
      <td>0.000</td>
      <td>Newcastle Council stresses heatwave is not mel...</td>
      <td>25</td>
      <td>-0.4588</td>
    </tr>
    <tr>
      <th>25</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 14:53:47 +0000 2018</td>
      <td>0.182</td>
      <td>0.182</td>
      <td>0.259</td>
      <td>Sebastian Vettel wins British GP, Lewis Hamilt...</td>
      <td>26</td>
      <td>0.2732</td>
    </tr>
    <tr>
      <th>26</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 14:47:28 +0000 2018</td>
      <td>0.114</td>
      <td>0.114</td>
      <td>0.111</td>
      <td>RT @BBCWorld: Nine boys and their coach will s...</td>
      <td>27</td>
      <td>-0.0258</td>
    </tr>
    <tr>
      <th>27</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 14:46:02 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Newtown burst water main clean-up begins https...</td>
      <td>28</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 14:43:10 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.155</td>
      <td>RT @bbcf1: Sebastian Vettel wins the #BritishG...</td>
      <td>29</td>
      <td>0.6784</td>
    </tr>
    <tr>
      <th>29</th>
      <td>@BBCNews</td>
      <td>Sun Jul 08 14:41:16 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>World Cup 2018: Pick your England XI for Croat...</td>
      <td>30</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 10:28:57 +0000 2018</td>
      <td>0.145</td>
      <td>0.145</td>
      <td>0.430</td>
      <td>A Good Appetite: Lamb, Under Fire, and at Its ...</td>
      <td>471</td>
      <td>0.6908</td>
    </tr>
    <tr>
      <th>471</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 10:00:05 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>A small Italian town once boasted of a “welcom...</td>
      <td>472</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>472</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 09:31:49 +0000 2018</td>
      <td>0.104</td>
      <td>0.104</td>
      <td>0.000</td>
      <td>Masih Alinejad tells the story of her near-met...</td>
      <td>473</td>
      <td>-0.2960</td>
    </tr>
    <tr>
      <th>473</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 09:14:26 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Pompeo Sharpens Tone on North Korea: ‘The Worl...</td>
      <td>474</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>474</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 08:57:41 +0000 2018</td>
      <td>0.074</td>
      <td>0.074</td>
      <td>0.207</td>
      <td>It's a ridiculous movie, but, at 64, Jackie Ch...</td>
      <td>475</td>
      <td>0.6297</td>
    </tr>
    <tr>
      <th>475</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 08:39:11 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.297</td>
      <td>Tech tools are far more efficient at keeping y...</td>
      <td>476</td>
      <td>0.6697</td>
    </tr>
    <tr>
      <th>476</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 08:21:54 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.330</td>
      <td>Don't be a bridezilla. Here are a few ways to ...</td>
      <td>477</td>
      <td>0.7096</td>
    </tr>
    <tr>
      <th>477</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 08:04:19 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.167</td>
      <td>Dog strollers, pawdicures and doga are just so...</td>
      <td>478</td>
      <td>0.5267</td>
    </tr>
    <tr>
      <th>478</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 07:47:00 +0000 2018</td>
      <td>0.289</td>
      <td>0.289</td>
      <td>0.000</td>
      <td>In China, the Opium War came to be seen as the...</td>
      <td>479</td>
      <td>-0.8074</td>
    </tr>
    <tr>
      <th>479</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 07:29:23 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.130</td>
      <td>"I took them to a party and got rave reviews a...</td>
      <td>480</td>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>480</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 07:12:01 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.238</td>
      <td>Because comfort food isn't just for cold weath...</td>
      <td>481</td>
      <td>0.3612</td>
    </tr>
    <tr>
      <th>481</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 06:54:30 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Long ago, Nick Offerman and Megan Mullally mad...</td>
      <td>482</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>482</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 06:37:17 +0000 2018</td>
      <td>0.100</td>
      <td>0.100</td>
      <td>0.000</td>
      <td>You asked, @nytimeswell answered: If one famil...</td>
      <td>483</td>
      <td>-0.2732</td>
    </tr>
    <tr>
      <th>483</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 06:19:51 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.356</td>
      <td>There’s Compliment Your Mirror Day. And Nation...</td>
      <td>484</td>
      <td>0.8074</td>
    </tr>
    <tr>
      <th>484</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 06:02:29 +0000 2018</td>
      <td>0.231</td>
      <td>0.231</td>
      <td>0.000</td>
      <td>This dog's nose knows if any of the bees have ...</td>
      <td>485</td>
      <td>-0.6705</td>
    </tr>
    <tr>
      <th>485</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 05:45:08 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>The time for fruit salad fragrances, one might...</td>
      <td>486</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>486</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 05:27:12 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.125</td>
      <td>Coffee may help you live longer, but it’s not ...</td>
      <td>487</td>
      <td>0.2144</td>
    </tr>
    <tr>
      <th>487</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 05:08:52 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Where can a book-buying baby boomer turn when ...</td>
      <td>488</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>488</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 04:51:05 +0000 2018</td>
      <td>0.088</td>
      <td>0.088</td>
      <td>0.116</td>
      <td>"Propane has no flavor, and charcoal isn’t muc...</td>
      <td>489</td>
      <td>0.1779</td>
    </tr>
    <tr>
      <th>489</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 04:32:45 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.166</td>
      <td>When summer weather turns sultry, museums can ...</td>
      <td>490</td>
      <td>0.3804</td>
    </tr>
    <tr>
      <th>490</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 04:15:53 +0000 2018</td>
      <td>0.167</td>
      <td>0.167</td>
      <td>0.000</td>
      <td>“Since my mom died, the only place I've really...</td>
      <td>491</td>
      <td>-0.5574</td>
    </tr>
    <tr>
      <th>491</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 03:56:53 +0000 2018</td>
      <td>0.080</td>
      <td>0.080</td>
      <td>0.000</td>
      <td>There’s only one catch. You have to fill out a...</td>
      <td>492</td>
      <td>-0.1027</td>
    </tr>
    <tr>
      <th>492</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 03:52:23 +0000 2018</td>
      <td>0.142</td>
      <td>0.142</td>
      <td>0.272</td>
      <td>A rescue in Thailand is underway to save 12 tr...</td>
      <td>493</td>
      <td>0.4767</td>
    </tr>
    <tr>
      <th>493</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 03:39:13 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.143</td>
      <td>"I resolutely believe," Nigella Lawson writes,...</td>
      <td>494</td>
      <td>0.4902</td>
    </tr>
    <tr>
      <th>494</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 03:23:00 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Proof of Children’s Vaccinations? Italy Will N...</td>
      <td>495</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>495</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 03:15:23 +0000 2018</td>
      <td>0.333</td>
      <td>0.333</td>
      <td>0.000</td>
      <td>Bullying, divorce, school shootings, racism an...</td>
      <td>496</td>
      <td>-0.8402</td>
    </tr>
    <tr>
      <th>496</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 02:57:22 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>For @NellieBowles, who writes about tech cultu...</td>
      <td>497</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>497</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 02:39:19 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Several large hotel chains are now trying to c...</td>
      <td>498</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>498</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 02:20:49 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Facebook Removes a Gospel Group’s Music Video ...</td>
      <td>499</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>499</th>
      <td>@nytimes</td>
      <td>Sun Jul 08 02:04:52 +0000 2018</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>Millennial pink's reign may be nearing an end....</td>
      <td>500</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python
# Create DataFrame from Results List
#results_df = pd.DataFrame(results_list).set_index("Airline").round(3)
# results_df
```


```python
senti_results_list_df.to_csv("Tweet.csv", index=False, header=True)
```


```python
# Create plot
#plot scatterplot using a for loop.
for user in target_user:
    plot_data = senti_results_list_df.loc[senti_results_list_df["Airline_user"] == user]
    plt.scatter(plot_data["Tweets Ago"],plot_data["compound"],label = user, 
               alpha=1.0, edgecolors='black')
    
#Add legend
plt.legend(bbox_to_anchor=(1, 1))

#Add title, x axis label, and y axis label.
plt.title("Sentiment Analysis of Media Tweets")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")

#Set a grid on the plot.
plt.grid()

plt.savefig("TweetPlot1.png")
plt.show()
```


![png](output_5_0.png)



```python
plt.savefig("Tweetplot.png")
```


    <matplotlib.figure.Figure at 0x1b05fab6438>



```python
avg_senti = senti_results_list_df.groupby("Airline_user")["compound"].mean()
```




    Airline_user
    @BBCNews    0.123894
    @CBS        0.363259
    @CNN        0.085868
    @FOXNEWS    0.042866
    @nytimes    0.079478
    Name: compound, dtype: float64




```python
x_axis = np.arange(len(avg_senti))
xlabels = avg_senti.index
count = 0
for score in avg_senti:
    if score < 0:
        height = score - .01
    else:
        height = score + .01
    plt.text(count, score, str(round(score,2)))
    count = count + 1
plt.bar(x_axis, avg_senti, tick_label = xlabels, color = ['lightblue', 'green', 'red', 'blue', 'yellow'])
#Set title, x axis label, and y axis label.
plt.title("Overall Sentiment of Media Tweets")
plt.xlabel("News Rooms")
plt.ylabel("Tweet Polarity")
plt.savefig("Bar plot of news tweets.png")
plt.show()
```


![png](output_8_0.png)

