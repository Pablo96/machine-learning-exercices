import pandas as pd
import numpy as np

##
# Movies recommendation with item based collaborative filtering
##

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('/home/llama/Downloads/MLCurse/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('/home/llama/Downloads/MLCurse/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

# merge the columns making it a bigger table.
# now ratings contain 4 columns:
# 'user_id', 'movie_id', 'rating' and 'title'
ratings = pd.merge(movies, ratings)

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')

# Look for starwars ratings
starWarsRatings = movieRatings['Star Wars (1977)']

# Compute pairwise correlations, meaning we compute correlations of all the movies to Starwars
# | title | similarity(correlation) |
similarMovies = movieRatings.corrwith(starWarsRatings)
#remove missin values (NaN)
similarMovies = similarMovies.dropna()

# Group ratings by title 
# and for each title count how many ppl rated it
# and the mean rating, thus creating a table of 2 columns
# rating being a complex column
# |       |    rating   |
# | title | size | mean |
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})

# Get rid of movies rated by less than 100 ppl
# NOTE: DataFrame has the operator '>=' (and others)
#   overloaded, so it creates a filter predicate
popularMovies = movieStats['rating']['size'] >= 100
# get the first 15 movies ordered by highest rating(mean) first
# NOTE: the predicate is passed and the '[]' operator will
#   use it to filter thos movies that dont match the filter
movieStats = movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)
mappedColumnsMoviestat=movieStats
# We flaten table giving us a table of 3 columns
# | title | rating-size | rating-mean |
mappedColumnsMoviestat.columns=[f'{i}|{j}' if j != '' else f'{i}' for i,j in mappedColumnsMoviestat.columns]
# Now we join the similar movies
# and thus we end up with a 4 columns table:
# | title | rating-size | rating-mean | similarity
df = mappedColumnsMoviestat.join(pd.DataFrame(similarMovies, columns=['similarity']))
# Get the 15 most similar movies (highest correlation value)
df = df.sort_values(['similarity'], ascending=False)[:15]

print()
print("similar movies")
print(df.head(15))
