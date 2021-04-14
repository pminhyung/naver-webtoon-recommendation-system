# import
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# Latent factor based Collaborative Filtering
class LatentCF:
    def __init__(self):
        self.P, self.Q = None, None
        self.pred_matrix = None
        self.unseen_list = None
        self.rating_matrix = None
        self.user2id, self.id2user = None, None

    def get_rmse(self, R, P, Q, non_zeros):  # R = rating_matrix
        full_pred_matrix = np.dot(P, Q.T)

        x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
        y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
        R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
        full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
        mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
        rmse = np.sqrt(mse)

        return rmse

    def train(self, R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
        """
        matrix_factorization
        :param K:
        :param steps:
        :param learning_rate:
        :param r_lambda:
        :return:
        """
        num_users, num_items = R.shape
        np.random.seed(1)
        P = np.random.normal(scale=1. / K, size=(num_users, K))
        Q = np.random.normal(scale=1. / K, size=(num_items, K))

        non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

        for step in range(steps):
            for i, j, r in non_zeros:
                eij = r - np.dot(P[i, :], Q[j, :].T)
                P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
                Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

            rmse = self.get_rmse(R, P, Q, non_zeros)
            if (step % 40) == 0:
                print('STEP_COUNT: ', step, 'RMSE :', rmse)

        self.P, self.Q = P, Q
        self.pred_matrix = np.dot(self.P, self.Q.T)



    def get_unseen_webtoons(self, userid):
        user_rating = self.rating_matrix.loc[userid, :]
        already_seen = user_rating[user_rating > 0].index.tolist()
        webtoons_list = self.rating_matrix.columns.tolist()
        unseen_list = [webtoon for webtoon in webtoons_list if webtoon not in already_seen]
        print('평점 매긴 영화수:', len(already_seen), '추천대상 영화수:', len(unseen_list), \
              '전체 영화수:', len(webtoons_list))

        self.unseen_list = unseen_list


    def recommend_webtoons(self, userid, top_n=5, save = False):
        """
        Recommend webtoons by userid

        :param userid: userid in data
        :param top_n: number of recommendations
        :param save: save to xlsx (bool)
        :return:
        """
        pred_df = pd.DataFrame(data=self.pred_matrix, columns=self.rating_matrix.columns, index=self.rating_matrix.index)
        recomm_webtoons = pred_df.loc[userid, self.unseen_list].sort_values(ascending=False)[:top_n]
        recomm_webtoons_df = pd.DataFrame(data=recomm_webtoons.values, index=recomm_webtoons.index, columns=['pred_score'])

        print('\n', '%%%% {} %%%% 님의'.format(self.user2id[userid]), '\n')
        print('## 추천 5개 웹툰 ## ', '\n', recomm_webtoons)
        print('=' * 70)

        if save:
            recomm_webtoons_df.to_xlsx('recommendation.xlsx', index=False)

    def preprocess_data(self, df):
        """

        :param df: survey data(dataframe)
        :return: rating_matrix
        """
        prep_df = df
        self.user2id = {u:idx for idx, u in enumerate(list(df['이름']))}
        self.id2user = {idx:u for idx, u in enumerate(list(df['이름']))}

        self.rating_matrix = prep_df

# Item based Collaborative Filtering

class ItemCF(LatentCF):
    def __init__(self):
        super(ItemCF, self).__init__()
        pass

    def get_rmse_Item(self, pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()

        return np.sqrt(mean_squared_error(pred, actual))

    def get_item_sim_df(self, ratings_matrix):

        ratings_matrix_T = ratings_matrix.transpose()
        item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)
        item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns,
                                   columns=ratings_matrix.columns)
        return item_sim_df

    def predict_rating_topsim(self, ratings_arr, item_sim_arr, n=10):
        pred = np.zeros(ratings_arr.shape)

        for col in range(ratings_arr.shape[1]):
            top_n_items = [np.argsort(item_sim_arr[:, col])[:-n - 1:-1]]
            for row in range(ratings_arr.shape[0]):
                pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row, :][top_n_items].T)
                pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))

        return pred

    def get_ratings_pred_matrix(self, ratings_matrix, top_n):
        ratings_pred_arr = self.predict_rating_topsim(ratings_matrix.values, self.get_item_sim_df(ratings_matrix).values, n=top_n)
        return ratings_pred_arr

    def get_preferred_top_n(self, ratings_matrix, userId, top_n):
        user_rating_id = ratings_matrix.loc[userId, :]
        return user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:top_n]

    def show_result_Item(self, rating_matrix, userId):

        ratings_pred_arr = self.get_ratings_pred_matrix(rating_matrix, 10)
        preferred_webtoons = self.get_preferred_top_n(rating_matrix, 235, 5)
        unseen_list = self.get_unseen_webtoons(rating_matrix, userId)
        recomm_webtoons = self.recomm_webtoons_by_userid(survey, ratings_pred_arr, userId, unseen_list, top_n=5)

        return preferred_webtoons, recomm_webtoons

# 'Surprise' based recommendation system




if __name__ == '__main__':
    pass