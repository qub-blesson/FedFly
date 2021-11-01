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
import threading

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='FedFly or classic FL mode', type= utils.str2bool, default= False)
args=parser.parse_args()

edge_server_threads = {}
LR = config.LR
offload = args.offload
first = True # First initializaiton control
print('program started')

unit_server = True
logger.info('Preparing Unit model Sever.')
sever = Sever(0, config.UNIT_MODEL_SERVER, config.UNIT_PORT, 'VGG5')
#sever.initialize(config.split_layer, offload, first, LR)
#first = False
res = {}
res['test_acc_record'] = []

#state_dim = 2*config.G
#action_dim = config.G

def unitserver_start():
	for r in range(config.R):
		#if r != 0:
		#	sever.sendunitmodel()
		#	time.sleep(3.5)
		#for k in range(len(config.EDGE_SERVERS)):
			#get data from each server
			#time.sleep(5)
		#test_acc = sever.test(r)
		#if r==4:
			#add the new edge server
			#config.EDGE_SERVERS['Rehmats-MacBook-Pro.local'] = '192.168.10.104'
		#	config.N = 2         
		sever.aggregateEdges()
		print('########################################################################')
		test_acc = sever.test(r)
		print('########################################################################')
		print(test_acc)
			
		
		res['test_acc_record'].append(test_acc)

		logger.info('Round Finish')
	
		logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
		#self.printUnit()
		print('reinitialize started')
		sever.reinitialize(config.split_layer, offload, first, LR)		
			#UPDATE MODEL
			#sever.updateunitmodel()

		with open(config.home + '/results/FedAdapt_res.pkl','wb') as f:
				pickle.dump(res,f)	
	'''
	try:
		while(True):
			sever.aggregateEdges()
	except KeyboardInterrupt:
		print('interuppted')
	
	#SEND DATA
	for k in range(len(config.EDGE_SERVERS)):
		#send it back to edgeservers
		sever.sendunitmodel()
		'''


def serverrun(serverip, clients_list):
	for r in range(config.R):
			logger.info('====================================>')
			logger.info('==> Round {:} Start'.format(r))

			s_time = time.time()
			#state, bandwidth = sever.train(thread_number= config.K, client_ips= config.CLIENTS_LIST)
			bandwidth = sever.train(thread_number= config.K, client_ips= clients_list)
			
			a_s_time = time.time()
			#aggregrated_model = sever.aggregate(config.CLIENTS_LIST)
			aggregrated_model = sever.aggregateEdges()
			agr_time = time.time() - a_s_time 
			res['agr_time'].append(agr_time)
			print(aggregrated_model)
			print('aggregation model received')
			e_time = time.time()
			
			# Recording each round training time, bandwidth and test accuracy
			trianing_time = e_time - s_time
			res['trianing_time'].append(trianing_time)
			res['bandwidth_record'].append(bandwidth)
			

			test_acc = sever.test(r)
			
			
			res['test_acc_record'].append(test_acc)

			with open(config.home + '/results/FedAdapt_res.pkl','wb') as f:
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

			sever.reinitialize(split_layers, offload, first, LR)
			logger.info('==> Reinitialization Finish')

if __name__ == '__main__':
	#if offload:
		#Initialize trained RL agent 
		#agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)
		#agent.policy.load_state_dict(torch.load('./PPO_FedAdapt.pth'))

	if offload:
		logger.info('FedFly Training')
	else:
		logger.info('Classic FL Training')

	#res = {}
	#res['trianing_time'], res['test_acc_record'], res['bandwidth_record'], res['agr_time'] = [], [], [], []

	'''
	for key in config.EDGE_SERVERS.keys():
		serverip = config.EDGE_SERVERS[key]
		clients_list = config.EDGE_SERVER_AFFILIATION[key]
		self.edge_server_threads[key] = threading.Thread(target=serverrun, args=(serverip, clients_list,))
	'''
	unitserver_start()

		


