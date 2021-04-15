# import
import json

# save config as default file
def save_config():
    """
    save default config file
    :return: None
    """
    # temp_config
    config = {'data': {'path': 'data/survey.csv'},
              'train': {'K': 30,
                      'step': 200,
                      'lr': .01,
                      'rlambda': .01,
                        },
              'model': 'latent',
              'recommend': {'names': ['홍길동', '김철수', '최영희'],
                       'topn': 5,
                       'save': True,
                       'save_path': 'recommendation/result.xlsx'
                       },
            }

    with open('config.json', 'w') as outfile:
        json.dump(config, outfile)

    print('config saved!')
    outfile.close()

# read config from json file
def read_json(args):
    """read config from json file
    :param args: parser arguments
    :return: config dict
    """
    json_path = args.config_file
    config = json.load(json_path)
    return config

# generate config
def get_config(args):
    """ get config dict from argparser
    :param args: parser arguments
    :return: config
    """
    config = {}

    config['data']['path'] = args.data_path

    config['train']['K'] = args.K
    config['train']['step'] = args.step
    config['train']['lr'] = args.learning_rate
    config['train']['rlambda'] = args.rlambda

    config['model'] = args.model

    config['recommend']['names'] = args.names
    config['recommend']['topn'] = args.topn
    config['recommend']['save'] = args.save
    config['recommend']['save_path'] = args.save_path
    return config

if __name__ == '__main__':
    save_config()