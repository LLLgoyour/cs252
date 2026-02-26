'''test04_read_missing_data.py
Test `Data` class constructor and `read` method with a focus on missing data
CS 251/2: Data Analysis and Visualization
Spring 2026
Oliver W. Layton
'''
from data import Data
import numpy as np
np.set_printoptions(precision=1, suppress=True, legacy='1.25')


def read_missing_data(test_filename):
    test_data = Data(test_filename)

    print(f'Your file path is:\n  {test_data.filepath}\nand should be:\n  data/test_data_missing.csv\n')
    print(f"Your headers are\n  {test_data.headers}\nand should be\n  ['age', 'fav_color', 'shoe_size', 'fav_sport']\n")
    print(f"Your variable name-to-column mapping is\n  {test_data.header2col}\nand should be\n  {{'age': 0, 'fav_color': 1, 'shoe_size': 2, 'fav_sport': 3}}\n")
    print(f"Your categorical variable name-to-level mapping is\n  {test_data.cats2levels}\nand should be\n  {{'fav_color': ['Missing', 'Purple'], 'fav_sport': ['Juggling', 'Missing']}}\n")

    print(f'Your test data looks like:\n', test_data.data)
    print('You should see:')
    expected_data = ''' [[ nan  0.   7.5  0. ]
 [22.   0.   nan  1. ]
 [25.   1.  10.5  1. ]]'''
    print(f'Your test data looks like:\n{expected_data}')
    print('Pay attention to the data type! The numbers should be floats or nan (not have quotes around them).')


def read_spotify_data(test_filename):
    test_data = Data(test_filename)

    print(f'Your file path is:\n  {test_data.filepath}\nand should be:\n  data/spotify.csv\n')
    print(f"Your headers are\n  {test_data.headers}\nand should be")
    print("  ['artist_1_pop', 'artist_2_pop', 'artist_3_pop', 'artist_1_genre_1', 'artist_1_genre_2',"
          " 'artist_1_genre_3', 'artist_2_genre_1', 'artist_2_genre_2', 'artist_2_genre_3', 'artist_3_genre_1',"
          " 'artist_3_genre_2', 'artist_3_genre_3', 'release_date', 'release_decade', 'danceability', 'energy',"
          " 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'popularity', 'duration', 'album_type',"
          " 'popularity_score', 'key', 'loudness', 'valence', 'tempo', 'duration_ms', 'time_signature']"
          )
    print(f"Your variable name-to-column mapping is")
    for key, value in test_data.header2col.items():
        print(f'  {key:15}: {value:<2}')
    print(f'and should be')
    expected = '''  artist_1_pop   : 0
  artist_2_pop   : 1
  artist_3_pop   : 2
  artist_1_genre_1: 3
  artist_1_genre_2: 4
  artist_1_genre_3: 5
  artist_2_genre_1: 6
  artist_2_genre_2: 7
  artist_2_genre_3: 8
  artist_3_genre_1: 9
  artist_3_genre_2: 10
  artist_3_genre_3: 11
  release_date   : 12
  release_decade : 13
  danceability   : 14
  energy         : 15
  speechiness    : 16
  acousticness   : 17
  instrumentalness: 18
  liveness       : 19
  popularity     : 20
  duration       : 21
  album_type     : 22
  popularity_score: 23
  key            : 24
  loudness       : 25
  valence        : 26
  tempo          : 27
  duration_ms    : 28
  time_signature : 29'''
    print(expected)
    print(f'The first row of test data looks like:')
    print(test_data.data[0])
    print('You should see:')
    expected_data = '''[     0.1      nan      nan      0.       0.       0.       0.       0.
      0.       0.       0.       0.    2015.       0.       0.       0.
      0.       0.       0.       0.       0.       0.       0.       0.
      0.      -8.8      0.5    128.1 195000.       0. ]'''
    print(f'{expected_data}')
    print(f'The last row of test data looks like:')
    print(test_data.data[-1])
    print('You should see:')
    expected_data = '''[     0.7      nan      nan      0.       3.       2.       0.       0.
      0.       0.       0.       0.    1983.       1.       1.       1.
      0.       0.       0.       2.       1.       0.       0.      64.
      0.     -13.1      0.6    102.7 241920.       0. ]'''
    print(f'{expected_data}')
    print('Pay attention to the data type! The numbers should be floats or nan (not have quotes around them).')


if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Beginning test 1 (Data with missing data)...')
    print('---------------------------------------------')
    mixed_filename = 'data/test_data_missing.csv'
    read_missing_data(mixed_filename)
    print('---------------------------------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print('Beginning test 2 (Large Spotify dataset with missing data)...')
    print('---------------------------------------------')
    filename = 'data/spotify.csv'
    read_spotify_data(filename)
    print('---------------------------------------------')
    print('Finished test 2!')
    print('---------------------------------------------------------------------------------------')