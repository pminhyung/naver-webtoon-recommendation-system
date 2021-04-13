# import
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from train_helper import get_rmse

# Latent factor based Collaborative Filtering

class LatentCF:
    def __init__(self):
        pass

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

        prev_rmse = 10000
        break_count = 0

        non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

        for step in range(steps):
            for i, j, r in non_zeros:
                eij = r - np.dot(P[i, :], Q[j, :].T)
                P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
                Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

            rmse = get_rmse(R, P, Q, non_zeros)
            if (step % 40) == 0:
                print('STEP_COUNT: ', step, 'RMSE :', rmse)

        return P, Q


    def get_unseen_webtoons(rating_matrix, userId):
        user_rating = rating_matrix.loc[userId, :]
        already_seen = user_rating[user_rating > 0].index.tolist()
        webtoons_list = rating_matrix.columns.tolist()
        unseen_list = [webtoon for webtoon in webtoons_list if webtoon not in already_seen]
        print('평점 매긴 영화수:', len(already_seen), '추천대상 영화수:', len(unseen_list), \
              '전체 영화수:', len(webtoons_list))

        return unseen_list


    def recommend_webtoons(self, rating_matrix, pred_array, userId, unseen_list, top_n=5):
        """
        recomm_webtoons_by_userid
        :param pred_array:
        :param userId:
        :param unseen_list:
        :param top_n:
        :return:
        """
        pred_df = pd.DataFrame(data=pred_array, columns=rating_matrix.columns, index=rating_matrix.index)
        recomm_webtoons = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
        recomm_webtoons_df = pd.DataFrame(data=recomm_webtoons.values, index=recomm_webtoons.index, columns=['pred_score'])

        return recomm_webtoons_df

# Item based Collaborative Filtering


# 'Surprise' based recommendation system




if __name__ == '__main__':
    pass