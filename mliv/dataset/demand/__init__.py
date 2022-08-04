import os
import json
from .demand_v1 import  generate_Demand_train, generate_Demand_test, set_Configuration, config

example = '''
from mliv.dataset.demand import gen_data
from mliv.utils import CausalDataset
gen_data()
data = CausalDataset('./Data/Demand/0.5_1.0_0.0_10000/1/')
'''

def gen_data(config=config):
    config, config_trt, config_val, config_tst = set_Configuration(config)
    exps = config['exps']
    dataName = config['dataName']
    path = './Data/{}/{}_{}_{}_{}/'.format(config['dataName'],config['rho'],config['alpha'],config['beta'],config['num'])
    print(f'The path: {path}')

    for exp in range(exps):

        print(f'Generate {dataName} datasets - {exp}/{exps}. ')

        config_trt['seed'], config_val['seed'], config_tst['seed'] = config_trt['seed'] + exp*333, config_val['seed'] + exp*444, config_tst['seed'] + exp*555

        train = generate_Demand_train(**config_trt)
        valid = generate_Demand_train(**config_val)
        test  = generate_Demand_test(**config_tst)

        data_path = path + '/{}/'.format(exp)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        train.to_csv(data_path + '/train.csv', index=False)
        valid.to_csv(data_path + '/valid.csv', index=False)
        test.to_csv(data_path + '/test.csv', index=False)

        configs = {'config':config, 'config_trt':config_trt, 'config_val':config_val, 'config_tst':config_tst}
        with open(data_path + "/configs.json", "w") as file:
            file.write( json.dumps(configs) )

    return config