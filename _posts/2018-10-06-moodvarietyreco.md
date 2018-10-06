---
title: "Mood and Variety: Recommendation System"
date: 2018-10-06
tags: [machine learning, recommendation engine, data science, content filtering, collaborative filtering, singular value decomposition]
header: 
  image: "/images/moodvarietyreco/Untitled design (3).png"
excerpt: "Machine Learning, Recommendation System, Hybrid"
mathjax: "true"

---

# H1 Heading

## H2 Heading

### H3 Heading

I wanted to get some recommendations for movies that related to my current mood, not exactly the same, as the same stuff repeated again gets boring. So, I decided why not try to make one! Doing so, I entered into the world of surprising insights like...  . 

The system is a hybrid one with first content filtering, and then collaborative filtering using Singular Value Decomposition using surprise package in Python. The collaborative filtering is a standard one, but I have played on with the content filtering algorithm. The main motive was to have content filter that would ................

The data usd is the MovieLens small data available at this [link](https://grouplens.org/datasets/movielens/). It has around  100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. 

All the code is available at this [link]().

The data has 3 files: 
- movies.csv: It contains the fields movieId, title, and genres of the movie. 
- tags.csv: It contains userId, movieId, and tag.
- ratings.csv: It contains userId, movieId, and rating. 

First, lets get started, read the tags and movies data, merge to put things together, in the code below: 

```python
import pandas as pd
import time

df_tags= pd.read_csv('data/tags.csv')
df_movies = pd.read_csv('data/movies.csv')
df_movies['genres'] = df_movies['genres'].apply(lambda x: x.split('|'))
df_tags_combined = df_tags.groupby('movieId').apply(lambda x: list(x['tag'])).reset_index().rename(columns={0:'tags'})
df_movies = pd.merge(df_movies, df_tags_combined, on = 'movieId', how = 'left')

df_movies['tags'] = df_movies['tags'].apply(lambda x: x if isinstance(x,list) else [])
df_movies['keywords'] = df_movies['genres']+df_movies['tags']
df_movies['keywords'] = df_movies['keywords'].apply(lambda x: set([str.lower(i.replace(" ", "")) for i in x]))
df_movies.set_index('movieId', inplace= True)

```
The df_movies dataframe now looks like this: 

<img src="{{ site.url }}{{ site.baseurl }}/images/moodvarietyreco/movies2.png" alt="movie_lens_small">

The tags and genres have been combined into keywords for each movie, not differentiating between the two. Now, the aim is to find the chief keyword of each movie: **the keyword that has the most predictive power to determine the mean rating of that movie by all users**. 

For that, first lets create a movies cross keywords dataframe, where each row is a movie and each column is a keyword, and the values are binary indicators indicating whether that keyword is present in that movie or not. We will also need to read the ratings data and get the mean_rating for each movie.

```python
df_ratings = pd.read_csv('data/ratings.csv')

all_keywords = set()
for this_movie_keywords in df_movies['keywords']:
    all_keywords = all_keywords.union(this_movie_keywords)

df_mxk = pd.DataFrame(0, index = df_movies.reset_index()['movieId'].unique(), columns = all_keywords)
df_mxk['mean_rating'] = df_ratings.groupby('movieId')['rating'].mean()

for index,row in df_mxk.iterrows():
    df_mxk.loc[index,df_movies.loc[index]['keywords']] = 1

df_mxk['mean_rating'].fillna(df_mxk['mean_rating'].mean(), inplace=True)
df_mxk = df_mxk.loc[:,df_mxk.sum() > 5]

```

We have dropped the rare keywords that appear in less than 6 movies. The df_mxk dataframe looks like this: 

<img src="{{ site.url }}{{ site.baseurl }}/images/moodvarietyreco/movies3.png" alt="movie_lens_small">

<img src="{{ site.url }}{{ site.baseurl }}/images/moodvarietyreco/movies4.png" alt="movie_lens_small">



And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
