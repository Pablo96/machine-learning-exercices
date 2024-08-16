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

# Pivot table ...
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')

# we'll use the min_periods argument to throw out results
# where fewer than 100 users rated a given movie pair
corrMatrix = userRatings.corr(method='pearson', min_periods=300)

my_user = 0
myRatings = userRatings.loc[my_user].dropna()
print("\nMy Ratings:")
print(myRatings.head())

# So for each movie I rated,
# I'll retrieve the list of similar movies from our correlation matrix.
# I'll then scale those correlation scores by how well I rated the movie they are similar to,
# so movies similar to ones I liked count more than movies similar to ones I hated:
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    # Retrieve similar movies to this one that I rated
    similarMovies = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie

    similarMovies = similarMovies.map(lambda similarity: similarity * myRatings.iloc[i] * (myRatings.iloc[i]//3))
    similarMovies = similarMovies[similarMovies >= 3]
    # Add the score to the list of similarity candidates
    if i == 0:
        simCandidates = similarMovies
    elif not similarMovies.empty and similarMovies.notnull().any().any() and len(similarMovies) > 0:
        simCandidates = pd.concat([simCandidates, similarMovies])


# Group by idices (movie titles)
simCandidates = simCandidates.groupby(simCandidates.index).sum()
# Sort values (similarity)
simCandidates = simCandidates.sort_values(ascending = False)

# Remove the movies I've rated
filteredSims = simCandidates.drop(myRatings.index)
print("\n--------------")
print(" filteredSims")
print("--------------\n")
print(filteredSims.head(10))