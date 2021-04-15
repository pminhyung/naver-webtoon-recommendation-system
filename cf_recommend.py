import os
import argparse

import pandas as pd

from utils import read_json, get_config
from models import LatentCF, ItemCF

def main(config):
    # config to params
    data_path = config['data']['path']
    K = config['train']['K']
    steps = config['train']['step']
    lr = config['train']['lr']
    r_lambda = config['train']['rlambda']
    save = config['recommend']['save']
    save_path = config['recommend']['save_path']
    usernames = config['recommend']['names']
    top_n = config['recommend']['topn']

    # load data
    survey_df = pd.read_csv(data_path)

    # recommendation for users
    for name in usernames:
        if config['model'] == 'latent':
            # use latent collaborative filtering for recommendation
            latentcf = LatentCF(config)
            rating_matrix = latentcf.get_rating_matrix(survey_df)
            latentcf.train(rating_matrix, K, steps, lr, r_lambda)
            recomm_webtoons_df = latentcf.recommend_webtoons(rating_matrix, username=name, top_n= top_n)


        elif config['model'] == 'item':
            itemcf = ItemCF(config)
            rating_matrix = itemcf.get_rating_matrix(survey_df)
            preferred_webtoons, recomm_webtoons_df = itemcf.show_result_item(rating_matrix, username=name, top_n = top_n)

            if save:
                # save result files
                dir_name = os.path.dirname(save_path)
                file_name = os.path.basename(save_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                recomm_webtoons_df.to_excel(dir_name + '/' + name + '_' + file_name, index=False)
                print('recommendation result file saved:', dir_name + '/' + name + '_' + file_name)


if __name__ == '__main__':
    # argparse -> config
    parser = argparse.ArgumentParser(description='recommend parser')

    parser.add_argument('-p', '--csv_path', help='csv_path:str', default = 'data/survey.csv')
    parser.add_argument('-c', '--config_file', help='config_filepath:str', default = None)

    parser.add_argument('-k', '--K', help='train_K:int', default = 30)
    parser.add_argument('-s', '--step', help='train_step:int', default=200)
    parser.add_argument('-lr', '--learning_rate', help='train_learning_rate:float', default = .01)
    parser.add_argument('-rl', '--rlambda', help='train_r_lambda:float', default = .01)

    parser.add_argument('-m', '--model', help='model-to-use:str', default= 'latent')

    parser.add_argument('-n', '--names', help='uesrnames_to_recommend:str', action='append')
    parser.add_argument('-tn', '--topn', help='recommend_top_n:int', default = 5)
    parser.add_argument('-s', '--save', help='result_save:bool', default = False)
    parser.add_argument('-sp', '--save_path', help='result_save_path:bool', default=False)

    args = parser.parse_args()

    if args.config_file:
        config = read_json(args)
    else:
        config = get_config(args)

    main(config)