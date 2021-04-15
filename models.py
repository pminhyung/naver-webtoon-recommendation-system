# import
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from utils import read_json

# Latent factor based Collaborative Filtering
class LatentCF:
    def __init__(self, config):
        """
        model of Latent factor based collaborative filtering

        :param survey_df: input_data(dataframe)
        """
        self.P, self.Q = None, None
        self.pred_matrix = None
        self.rating_matrix = None
        self.user2id = None

    def get_rating_matrix(self, survey_df):
        """

        :param df: survey data(dataframe)
        :return: rating_matrix
        """
        data = survey_df
        data['userid'] = range(len(data))
        data.set_index('userid', inplace=True)

        self.user2id = {name: id for name, id in zip(data['이름'], data['userid'])}

        drop_cols = ['타임스탬프',
                     '귀하의 성별은?',
                     '현재까지 감상한 웹툰 작품을 점수(1~5점)를 매겨주세요😃 보지 않으신 작품은 "없음"에 표시해주세요. ',
                     '이름',
                     '연락처']
        data.drop(drop_cols, axis=1, inplace=True)
        data.replace('없음', np.nan, inplace=True)
        data = data.astype('float64')

        self.rating_matrix = data
        print('rating matrix ready!')
        return self.rating_matrix

    def get_rmse(self, R, P, Q, non_zeros):  # R = rating_matrix
        full_pred_matrix = np.dot(P, Q.T)

        x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
        y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
        R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
        full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
        mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
        rmse = np.sqrt(mse)

        return rmse

    def train(self, rating_matrix, K=30, steps=200, learning_rate=0.01, r_lambda=0.01):
        """
        perform matrix_factorization

        :param R: rating matrix
        :param K:
        :param steps: train step
        :param learning_rate: learning_rate
        :param r_lambda:
        :return:
        """

        R = rating_matrix
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

    def get_unseen_webtoons(self, rating_matrix, userid):
        user_rating = rating_matrix.loc[userid, :]
        already_seen = user_rating[user_rating > 0].index.tolist()
        webtoons_list = rating_matrix.columns.tolist()
        unseen_list = [webtoon for webtoon in webtoons_list if webtoon not in already_seen]
        print('평점 매긴 영화수:', len(already_seen), '추천대상 영화수:', len(unseen_list),
              '전체 영화수:', len(webtoons_list))

        return unseen_list

    def recommend_webtoons(self, rating_matrix, username, top_n):
        """
        Recommend webtoons by userid

        :param username: username to recommend (included in survey data)
        :param top_n: number of recommendations
        :param save: save to xlsx (bool)
        :return:
        """
        userid = self.user2idx[username]
        unseen_list = self.get_unseen_webtoons(rating_matrix, userid)
        pred_df = pd.DataFrame(
                               data=self.pred_matrix,
                               columns=rating_matrix.columns,
                               index=rating_matrix.index
                              )
        recomm_webtoons = (pred_df.loc[userid, unseen_list]
                                  .sort_values(ascending=False)[:top_n])
        recomm_webtoons_df = pd.DataFrame(
                                          data=recomm_webtoons.values,
                                          index=recomm_webtoons.index,
                                          columns=['pred_score']
                                          )

        print('\n', '%%%% {} %%%% 님의'.format(username), '\n')
        print('## 추천 {}개 웹툰 ## '.format(self.top_n), '\n', recomm_webtoons_df)
        print('=' * 70)

        return recomm_webtoons_df

# Item based Collaborative Filtering
class ItemCF(LatentCF):
    def __init__(self):
        super(ItemCF, self).__init__()

    def get_rating_matrix(self, survey_df):
        """

        :param survey_df: survey data(dataframe)
        :return: rating_matrix
        """
        self.rating_matrix = super(ItemCF, self).get_rating_matrix(survey_df).fillna(0)
        return self.rating_matrix

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
        self.pred_matrix = self.predict_rating_topsim(ratings_matrix.values, self.get_item_sim_df(ratings_matrix).values, n=top_n)

    def get_preferred_top_n(self, ratings_matrix, userid, top_n):
        user_rating_id = ratings_matrix.loc[userid, :]
        return user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:top_n]

    def show_result_item(self, rating_matrix, username, top_n):
        userid = self.user2id[username]
        preferred_webtoons = self.get_preferred_top_n(rating_matrix, userid, top_n)
        recommend_webtoons_df = self.recommend_webtoons(rating_matrix, userid)

        print('\n', '%%%% {} %%%% 님의'.format(username), '\n')
        print('## 선호 {}개 웹툰 ## '.format(self.top_n), '\n', preferred_webtoons, '\n')
        print('## 추천 {}개 웹툰 ## '.format(self.top_n), '\n', recommend_webtoons_df)
        print('=' * 70)
        return preferred_webtoons, recommend_webtoons_df

if __name__ == '__main__':
    # load data
    config = read_json('config.json')
    path = 'data/naver_webtoon.csv'
    survey_df = pd.read_csv(path)

    # use latent collaborative filtering for recommendation
    latentcf = LatentCF(config)
    rating_matrix = latentcf.get_rating_matrix(survey_df)
    latentcf.train(rating_matrix, K=30, steps=200, learning_rate=0.01, r_lambda=0.01)
    recomm_webtoons_df = latentcf.recommend_webtoons(rating_matrix, username='박민형')

    # use latent collaborative filtering for recommendation
    itemcf = ItemCF(config)
    rating_matrix = itemcf.get_rating_matrix(survey_df)
    preferred_webtoons, recomm_webtoons = itemcf.show_result_item(rating_matrix, username = '박민형')

    """
    
    'example of LatentCF result'
    
    STEP_COUNT:  0 RMSE : 3.532614660716394
    STEP_COUNT:  40 RMSE : 0.4428637719198905
    STEP_COUNT:  80 RMSE : 0.3302075532662343
    STEP_COUNT:  120 RMSE : 0.2968993703202772
    STEP_COUNT:  160 RMSE : 0.28072334925892345
    
    %%%% 박민형 %%%% 님의 

    ## 추천 5개 웹툰 ##  
                             pred_score
    네이버 수요일 웹툰 [고삼무쌍]         5.522287
    네이버 완결 웹툰 [신과 함께]         5.394038
    네이버 일요일 웹툰 [마루한-구현동화전]    5.093802
    네이버 토요일 웹툰 [회춘]           5.069422
    네이버 완결 웹툰 [여중생 A]         5.045176
    
    'recommendation result file saved, recommendation.xlsx'
    
    """

