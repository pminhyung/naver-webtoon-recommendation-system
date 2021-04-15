import argparse

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise.dataset import DatasetAutoFolds

from utils import read_json


def get_unseen_surprise(ratings, webtoons, userId):
    seen_webtoons = ratings[ratings['userId'] == userId]['webtoonId'].tolist()

    total_webtoons = webtoons.columns[:-1].tolist()

    unseen_webtoons = [webtoon for webtoon in total_webtoons if webtoon not in seen_webtoons]
    print('평점 매긴 영화수:', len(seen_webtoons), '추천대상 영화수:', len(unseen_webtoons), \
          '전체 영화수:', len(total_webtoons))

    return unseen_webtoons


def recomm_webtoon_by_surprise(algo, userId, unseen_webtoons, top_n=10):
    predictions = [algo.predict(str(userId), str(webtoonId)) for webtoonId in unseen_webtoons]

    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    top_webtoon_titles = [pred.iid for pred in top_predictions]
    top_webtoon_rating = [pred.est for pred in top_predictions]

    top_webtoon_preds = [(title, rating) for title, rating in zip(top_webtoon_titles, top_webtoon_rating)]

    return top_webtoon_preds

def main(config):
    # load data and model fitting
    fname = config['data']['path']
    users = config['recommend']['names']

    webtoons = pd.read_csv('data/webtoons_survey(preprocessed).csv')
    ratings = pd.read_csv(fname)
    ratings = ratings.dropna(subset=['rating'])

    reader = Reader(rating_scale=(1.0, 5.0))

    data_folds = DatasetAutoFolds(df=ratings, reader=reader)
    trainset = data_folds.build_full_trainset()

    algo = SVD(n_epochs=200, n_factors=30, random_state=0)
    algo.fit(trainset)

    # show recommendation list
    find_user = pd.read_csv('webtoons_survey(original_form).csv')

    print('##### Surprise 모델 추천 리스트 #####')
    for user in users:
        userid = int(find_user[find_user['이름'] == user].index[0])
        unseen_webtoons = get_unseen_surprise(ratings, webtoons, userid)
        top_webtoon_preds = recomm_webtoon_by_surprise(algo, userid, unseen_webtoons, top_n=5)

        print('\n', '%%%% {} %%%% 님의'.format(user), '\n')
        print('## 추천 5개 웹툰 ## ', '\n')
        for top_webtoon in top_webtoon_preds:
            print(top_webtoon[0], ':', top_webtoon[1])
        print('=' * 70)

    cross_validate(algo, data_folds, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    param_grid = {'n_epochs': [40, 80, 120, 200], 'n_factors': [10, 20, 30, 50, 100]}

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data_folds)

    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])


if __name__ == '__main__':
    # argparse -> config
    parser = argparse.ArgumentParser(description='recommend parser')
    parser.add_argument('-c', '--config_file', help='config_filepath:str', default = None)
    parser.add_argument('-n', '--names', help='uesrnames_to_recommend:str', action='append')
    args = parser.parse_args()

    if args.config_file:
        config = read_json(args)

    main(config)