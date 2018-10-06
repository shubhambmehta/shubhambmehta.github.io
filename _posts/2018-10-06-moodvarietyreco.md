---
title: "Mood and Variety: Recommendation System"
date: 2018-09-28
tags: [machine learning, recommendation engine, data science, content filtering, collaborative filtering, singular value decomposition]
header: 
  image: "/images/moodvarietyreco/Untitled design (3).png"
excerpt: "Machine Learning, Recommendation System, Hybrid"
mathjax: "true"
---

# H1 Heading

## H2 Heading

### H3 Heading

I wanted to get some recommendations for movies that related to my current mood, not exactly the same, as the same stuff repeated again gets boring.So, I decided why not try to make one! Doing so, I entered into the world of surprising insights like...  . 

The system is a hybrid one with first content filtering, and then collaborative filtering using Singular Value Decomposition using surprise package in Python. The collaborative filtering is a standard one, but I have played on with the content filtering algorithm. The main motive was to have content filter that would ................

The data I used was the MovieLens small data available here ..... All the code is available at ..... 

So, lets get started and read the data

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

all_keywords = set()
for this_movie_keywords in df_movies['keywords']:
    all_keywords = all_keywords.union(this_movie_keywords)

```

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
