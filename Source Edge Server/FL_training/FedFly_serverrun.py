import time
import torch
import pickle
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Sever import Sever
import config
import utils
import PPO

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='FedFly or classic FL mode', type= utils.str2bool, default= False)
args=parser.parse_args()

LR = config.LR
offload = args.offload
first = True # First initializaiton control
print('program started')
logger.info('Preparing Sever.')
sever = Sever(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')
sever.initialize(config.split_layer, offload, first, LR)
first = False

state_dim = 2*config.G
action_dim = config.G

if __name__ == '__main__':
	if offload:
		#Initialize trained RL agent 
		agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)
		
	if offload:
		logger.info('FedFly Training')
	else:
		logger.info('Classic FL Training')

	res = {}
	res['trianing_time'], res['test_acc_record'], res['bandwidth_record'], res['agr_time'] = [], [], [], []

	for r in range(config.R):
		logger.info('====================================>')
		logger.info('==> Round {:} Start'.format(r))

		s_time = time.time()
		#state, bandwidth = sever.train(thread_number= config.K, client_ips= config.CLIENTS_LIST)
		bandwidth = sever.train(thread_number= config.K, client_ips= config.CLIENTS_LIST, epoch = r)
		split_layers = config.split_layer
		
		a_s_time = time.time()
		#aggregrated_model = sever.aggregate(config.CLIENTS_LIST)
		aggregrated_model = sever.collectWeights(config.CLIENTS_LIST)
		#test_acc = sever.test(r)
		sever.reinitialize(split_layers, offload, first, LR)

		agr_time = time.time() - a_s_time 
		res['agr_time'].append(agr_time)
		#print(aggregrated_model)
		#print('aggregation model received')
		e_time = time.time()
		
		# Recording each round training time, bandwidth and test accuracy
		trianing_time = e_time - s_time
		res['trianing_time'].append(trianing_time)
		res['bandwidth_record'].append(bandwidth)
		

		test_acc = sever.test(r)
		
		
		res['test_acc_record'].append(test_acc)

		with open(config.home + '/results/FedFly_res.pkl','wb') as f:
					pickle.dump(res,f)

		logger.info('Round Finish')
		logger.info('==> Round Training Time: {:}'.format(trianing_time))

		logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
		if offload:
			#split_layers = sever.adaptive_offload(agent, state)
			split_layers = config.split_layer
		else:
			split_layers = config.split_layer

		if r > 49:
			LR = config.LR * 0.1

		#sever.reinitialize(split_layers, offload, first, LR)
		logger.info('==> Reinitialization Finish')


