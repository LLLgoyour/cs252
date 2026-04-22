'''rec_sys.py
Recommender system algorithms
Jack Dai
CS 252: Mathematical Data Analysis and Visualization
Spring 2026
'''
import numpy as np


class Recommender:
    '''Parent class for recommender system algorithms'''
    def __init__(self, num_bvs):
        '''Recommender system constructor.

        Parameters:
        -----------
        num_bvs: int or None. Number of basis vectors used in the recommender system (if applicable).

        TODO: Set an instance variable for the number of basis vectors based in.
        '''
        
        self.num_bvs = num_bvs

    def replace_missing_with_mean(self, A):
        '''Replaces absent movie ratings (0s) with the mean of each user's set of ratings.

        Parameters:
        -----------
        A: ndarray. shape=(N, M). User-item matrix of star ratings.

        Returns:
        -----------
        ndarray. shape=(N, M). A matrix with no 0s. Absent movie ratings replaced the mean of each user's set of ratings.

        HINT: You could use a loop and logical indexing
        '''
        
        A_filled = A.copy()
        for i in range(A.shape[0]):
            rated_mask = A_filled[i] > 0
            if rated_mask.any():
                mean_i = A_filled[i, rated_mask].mean()
                A_filled[i, ~rated_mask] = mean_i
        return A_filled

    def fit(self):
        '''This method should be overridden by child implementations and remain blank here.'''
        pass

    def predict(self):
        '''This method should be overridden by child implementations and remain blank here.'''
        pass

    def predict_user_item_ratings(self, test_userids, test_movieids, clip_preds=True, clip_bounds=(0.5, 5)):
        '''Creates a 1D array of ratings from the user-item matrix PREDICTED by the recommender system (`A_pred`)
        corresponding to user IDs listed in `test_userids` and to the movie IDs listed in `test_movieids.

        If test_userids = [1, 1, 2] and test_movieids = [0, 11, 99], this method should extract and return an array
        with 3 entries:
            [pred_rating(user=1, movie=0), pred_rating(user=1, movie=11), pred_rating(user=2, movie=99)]

            For example: [4, 2.5, 1]
            if the predicted rating for user 1 and movie 0 is 4
            if the predicted rating for user 1 and movie 99 is 2.5
            if the predicted rating for user 2 and movie 99 is 1

        Parameters:
        -----------
        test_userids: 1D ndarray of ints. User IDs for which we want to extract predicted ratings. This list will often
            NOT be unique — test_userids = [1, 2, 3] and test_userids = [1, 1, 2] are both valid.
        test_movieids: 1D ndarray of ints. Movie IDs for which we want to extract predicted ratings. This list will often
            NOT be unique — test_movieids = [0, 11, 99] and test_movieids = [0, 11, 11] are both valid.
        clip_preds: bool. Do we clip the ratings in the predicted item-user A matrix to the valid range of star ratings
            stored in `clip_bounds`? In other words, do we set replace every value smaller than the smallest
            possible rating in the predicted A matrix with the lowest possible rating and every value larger than the
            largest possible rating in the predicted A matrix with the largest possible rating?
        clip_bounds: tuple. len=2. Format is (min_possible_rating_predicted, max_possible_rating_predicted).
            These bounds define the min/max possible value in the predicted ratings if `clip_preds` is `True`.

        Returns:
        -----------
        1D ndarray. len=len(test_userids)=len(test_userids). Predicted ratings for the requested user and movie ID
            combinations.

        NOTE: You should be calling the predict method in here to get the predicted user-item matrix.
        '''
        A_pred = self.predict()
        assert A_pred is not None
        ratings = A_pred[test_userids, test_movieids]
        if clip_preds:
            ratings = np.clip(ratings, clip_bounds[0], clip_bounds[1])
        return ratings

    def rmse(self, ratings_true, ratings_pred):
        '''Computes the root mean squared error (RMSE) between arrays of true and predicted star ratings.

        Parameters:
        -----------
        ratings_true: ndarray. shape=(num_ratings,). 1D array of true ratings.
        ratings_pred: ndarray. shape=(num_ratings,). 1D array of predicted ratings.

        Returns:
        -----------
        float. The RMSE.
        '''
        return np.sqrt(np.mean((ratings_true - ratings_pred) ** 2))


class SVD(Recommender):
    '''Recommender system that computes the Truncated SVD of the user-item matrix of ratings.'''
    def __init__(self, num_bvs):
        '''SVD recommender system constructor

        Parameters:
        -----------
        num_bvs: int. Number of basis vectors used in the SVD to approximate the original user-item matrix.

        TODO:
        - Call parent constructor.
        - Define instance variables here for the 3 SVD matrices initially set to `None`.
        '''
        super().__init__(num_bvs)
        self.U = None
        self.S = None
        self.Vt = None
        self.row_means = None

    def fit(self, A, replace_missing_with_mean=True):
        '''Computes the economy SVD on the user-item `A` with the desired number of basis vectors.

        Parameters:
        -----------
        A: ndarray. shape=(N, M). User-item matrix with float values ranging from 0. to 5.
            0 means movie is unrated by a user.
        replace_missing_with_mean: bool. Whether to replace absent ratings with each user's mean rating.

        NOTE:
        - You may want to look at the full_matrices keyword argument in NumPy SVD documentation so that you compute the
        "economy SVD": https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
        - Remember that you should center your data before computing SVD!
        - Remember to set the SVD matrices as instance variables.
        '''
        A_new = A.copy()
        if replace_missing_with_mean:
            A_new = self.replace_missing_with_mean(A_new)
        
        self.row_means = A_new.mean(axis = 1, keepdims = True)
        centered_A = A_new - self.row_means
        
        self.U, self.S, self.Vt = np.linalg.svd(centered_A, full_matrices=False)

        # keep vectors the model will use
        self.U = self.U[:, :self.num_bvs]
        self.S = self.S[:self.num_bvs]
        self.Vt = self.Vt[:self.num_bvs, :]
        
        
        

    def predict(self):
        '''Computes the Truncated SVD-approximated user-item matrix

        Returns:
        -----------
        ndarray. shape=(N, M). The user-item matrix of predicted ratings.

        NOTE: You will need to undo the centering. You may have to modify `fit` to store as an instance variable the
        information required to undo the centering here.
        '''
        assert self.U is not None
        assert self.S is not None
        assert self.Vt is not None
        assert self.row_means is not None
        A_pred = self.U @ np.diag(self.S) @ self.Vt
        A_pred = A_pred + self.row_means
        return A_pred


class Mean(Recommender):
    '''Mean user rating recommender system model: Ratings for movies that are not filled in for a user are filled in
    with that user's mean rating among movies they HAVE rated.
    '''
    def __init__(self):
        '''Mean user rating model constructor.

        This is prefilled for you and should not require modification.
        '''
        super().__init__(num_bvs=None)
        self.A_fit = None

    def fit(self, A):
        '''Preserves existing ratings and fills in missing ratings (0s) with each user's mean rating. The result of this
        is stored as self.A_fit.

        Parameters:
        -----------
        A: ndarray. shape=(N, M). User-item matrix with float values ranging from 0. to 5.
            0 means movie is unrated by a user.

        NOTE: Make a copy of A before making any modifications. We are defining `A_fit`, but do not want to modify the
        `A` passed in.
        '''
        self.A_fit = self.replace_missing_with_mean(A.copy())

    def predict(self):
        '''Returns the predicted user-item matrix.

        Returns:
        -----------
        ndarray. shape=(N, M). The user-item matrix of predicted ratings according to the mean user rating model.
        '''
        assert self.A_fit is not None
        return self.A_fit


class FunkSVD(Recommender):
    '''Funk SVD recommender system'''
    def __init__(self, num_bvs):
        '''Funk SVD constructor

        Parameters:
        -----------
        num_bvs: int. Number of basis vectors used in the SVD to approximate the original user-item matrix.

        TODO:
        - Call parent constructor.
        - Define instance variables here the user-factor and factor-item matrices that will be computed by Funk SVD.
        Set them to `None` here.
        '''
        super().__init__(num_bvs)
        self.U = None
        self.I = None

    def get_user_factor_matrix(self):
        '''Returns the user-factor matrix.

        Returns:
        -----------
        ndarray. shape=(N, K). The user-factor matrix with `K` factors (i.e. basis vectors).
        '''
        
        user_mat = np.array()
        return 

    def get_factor_item_matrix(self):
        '''Returns the factor-item matrix.

        Returns:
        -----------
        ndarray. shape=(K, M). The factor-item matrix with `K` factors (i.e. basis vectors).
        '''
        pass

    def fit(self, A, step=0.009, n_iter=100, reg=0.1):
        '''Decomposes the user-item matrix `A` into a user-factor matrix (`U`) and a factor-item matrix (`I`) using
        an iterative updating scheme (gradient descent).

        Parameters:
        -----------
        A: ndarray. shape=(N, M). User-item matrix with float values.
        step: float. Step size used to update entries in `U` and `I` on every iteration of the fit process.
        n_iter: int. Number of iterations to run the algorithm. On each iteration, we visit EVERY nonzero entry in `A`
            and make corresponding updates to both `U` and `I`.
        reg: float. Regularization strength. The extent to which we "resist" each update.

        TODO:
        - Initialize your `U` and `I` matrix instance variables with uniform random numbers between 0 and 1.
        - Your goal with the remaining portion is to iteratively update `U` and `I` according to the Funk SVD algorithm
        such that they progressively do a better job at approximating `A`.
        - Before you are about to update a row of U and col of I for a certain nonzero Aij rating, freeze/make a copy
        of the U row and I column and use those "frozen" vectors to make the update so that you are not changing numbers
        mid-update.

        NOTE: It is totally fine to use loops here :)
        '''
        pass

    def predict(self):
        '''Computes the Funk SVD approximated user-item matrix

        Returns:
        -----------
        ndarray. shape=(N, M). The user-item matrix of predicted ratings.
        '''
        pass
