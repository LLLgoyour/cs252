'''test05__str__.py
Test Data class __str__ method for printing `Data` objects
CS 251/2: Data Analysis and Visualization
Spring 2026
Oliver Layton, Caitrin Eaton, Hannah Wolfe, Stephanie Taylor
'''
import numpy as np

from data import Data


def print_iris(iris_filename):
    iris_data = Data(iris_filename)
    print(iris_data)

    template_str = '''
-------------------------------
data/iris.csv (150x5)
Headers:
  sepal_length  sepal_width     petal_length    petal_width     species
-------------------------------
Showing first 5/150 rows.
5.1     3.5     1.4     0.2     0.0
4.9     3.0     1.4     0.2     0.0
4.7     3.2     1.3     0.2     0.0
4.6     3.1     1.5     0.2     0.0
5.0     3.6     1.4     0.2     0.0

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


def print_anscombe(ans_filename):
    ans_data = Data(ans_filename)
    print(ans_data)

    template_str = '''
-------------------------------
data/anscombe.csv (44x3)
Headers:
  dataset       x       y
-------------------------------
Showing first 5/44 rows.
0.0     10.0    8.04
0.0     8.0     6.95
0.0     13.0    7.58
0.0     9.0     8.81
0.0     11.0    8.33

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


def print_spaces(test_filename):
    test_data = Data(test_filename)
    print(test_data)

    template_str = '''
-------------------------------
data/test_data_spaces.csv (3x4)
Headers:
  headers spaces  bad     places
-------------------------------
1.0     2.0     3.0     4.0
5.0     6.0     7.0     8.0
9.0     10.0    11.0    12.0

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


def print_complex(test_filename):
    test_data = Data(test_filename)
    print(test_data)

    template_str = '''
-------------------------------
data/test_data_complex.csv (15x2)
Headers:
  catstuff      numberstuff
-------------------------------
Showing first 5/15 rows.
0.0     4.0
0.0     3.0
1.0     2.0
2.0     1.0
2.0     5.0

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


def print_spotify(test_filename):
    test_data = Data(test_filename)
    print(test_data)

    template_str = '''
-------------------------------
data/spotify.csv (277938x30)
Headers:
  artist_1_pop  artist_2_pop    artist_3_pop    artist_1_genre_1        artist_1_genre_2        artist_1_genre_3        artist_2_genre_1        artist_2_genre_2        artist_2_genre_3        artist_3_genre_1        artist_3_genre_2  artist_3_genre_3        release_date    release_decade  danceability    energy  speechiness     acousticness    instrumentalness        liveness        popularity      duration        album_type      popularity_score key      loudness        valence tempo   duration_ms     time_signature
-------------------------------
Showing first 5/277938 rows.
0.09    nan     nan     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     2015.0  0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     -8.815  0.52    128.05  195000.0  0.0
0.68    0.56    nan     1.0     1.0     1.0     0.0     0.0     0.0     0.0     0.0     0.0     2015.0  0.0     0.0     1.0     0.0     0.0     0.0     1.0     0.0     0.0     1.0     33.0    1.0     -6.848  0.25    122.985 194641.0  0.0
0.65    nan     nan     2.0     1.0     1.0     0.0     0.0     0.0     0.0     0.0     0.0     2014.0  0.0     0.0     1.0     0.0     0.0     0.0     2.0     1.0     0.0     0.0     52.0    2.0     -8.029  0.247   170.044 217573.0  0.0
0.32    nan     nan     1.0     1.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     2014.0  0.0     0.0     1.0     0.0     0.0     0.0     2.0     0.0     0.0     0.0     0.0     3.0     -4.571  0.199   92.011  443478.0  0.0
0.3     0.39    nan     1.0     0.0     0.0     1.0     0.0     0.0     0.0     0.0     0.0     2014.0  0.0     0.0     1.0     0.0     1.0     0.0     2.0     0.0     0.0     0.0     0.0     3.0     -5.863  0.163   115.917 225862.0  0.0

-------------------------------
    '''
    print('You should get something that looks like:')
    print(template_str)


if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Beginning test 1 (Print Iris data)...')
    print('---------------------------------------------')
    data_file = 'data/iris.csv'
    print_iris(data_file)
    print('---------------------------------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Beginning test 2 (Print Anscombe data)...')
    print('---------------------------------------------')
    data_file = 'data/anscombe.csv'
    print_anscombe(data_file)
    print('---------------------------------------------')
    print('Finished test 2!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Beginning test 3 (Print spaces test data)...')
    print('---------------------------------------------')
    data_file = 'data/test_data_spaces.csv'
    print_spaces(data_file)
    print('---------------------------------------------')
    print('Finished test 3!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Beginning test 4 (Print complex test data)...')
    print('---------------------------------------------')
    data_file = 'data/test_data_complex.csv'
    print_complex(data_file)
    print('---------------------------------------------')
    print('Finished test 4!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Beginning test 5 (Print Spotify data)...')
    print('---------------------------------------------')
    data_file = 'data/spotify.csv'
    print_spotify(data_file)
    print('---------------------------------------------')
    print('Finished test 5!')
    print('---------------------------------------------------------------------------------------')