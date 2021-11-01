

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import random
import numpy as np
import sys
import time as t
from time import time
import socket as sk

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Communicator import *
import utils
import config

np.random.seed(0)
torch.manual_seed(0)

class Sever(Communicator):
	def __init__(self, index, ip_address, server_port, model_name):
		super(Sever, self).__init__(index, ip_address)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.port = server_port
		self.model_name = model_name
		self.criterion = nn.CrossEntropyLoss()
		
		self.client_socks = {}
		self.edge_server_sock = {}
		self.migration_port = 51001
		self.migration = False
		self.sock.bind((self.ip, self.port))
		#wait for the edge servers to connect
		self.establish_connection_with_edge_servers()

		'''
		while len(self.client_socks) < config.K:
			self.sock.listen(5)
			logger.info("Waiting Incoming Connections.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('Got connection from ' + str(ip))
			logger.info(client_sock)
			self.client_socks[str(ip)] = client_sock
		'''

		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)
		msg = ['MSG_UNINET_FROM_SERVER', self.uninet.state_dict()]
		

		#send all the messages
		for ip in self.edge_server_sock:
			#print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			#print(ip)
			#print(self.edge_server_sock[i])
			#print(self.edge_server_sock[str(ip)])
			#print(msg)
			
			self.send_msg(self.edge_server_sock[str(ip)], msg)
			t.sleep(3.5) #wait for some time for message to be sent completely
			self.edge_server_sock[str(ip)].close()
			#self.edge_server_sock[str(ip)].setsockopt(sk.SOL_SOCKET,sk.SO_REUSEADDR,1)
			#print(self.edge_server_sock[str(ip)])
			#t_end = t.time() + 60 * 5
			#while t.time() < t_end:
			#	pass


		
		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
		self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=False, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False)
		
	def establish_connection_with_edge_servers(self):
		connection_flag = True
		self.sock.listen(5)
		serverip_order = ['192.168.101.205', '192.168.101.106']
		i = 0
		while(connection_flag):
			logger.info("Waiting for Incoming Connections from edge servers. "+ str(serverip_order[i]))
			(server_sock, (ip, port)) = self.sock.accept()
			if str(ip) == str(serverip_order[i]):
				print(server_sock)
				logger.info('Got connection from ' + str(ip))
				logger.info(server_sock)
				self.edge_server_sock[str(ip)] = server_sock
				i = i+1
				if i == 2:
					connection_flag = False
			
			else:
				print('got wrong connection from'+str(ip))
				server_sock.shutdown(sk.SHUT_RDWR)
				server_sock.close()
		'''
		for i in range(len(config.EDGE_SERVERS)):
			self.sock.listen(5)
			logger.info("Waiting for Incoming Connections from edge servers.")
			(server_sock, (ip, port)) = self.sock.accept()
			print(server_sock)
			logger.info('Got connection from ' + str(ip))
			logger.info(server_sock)
			self.edge_server_sock[str(ip)] = server_sock
		'''
	
	def sendunitmodel(self):
		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)
		msg = ['MSG_UNINET_FROM_SERVER', self.uninet.state_dict()]
		

		#send all the messages
		for ip in self.edge_server_sock:
			self.send_msg(self.edge_server_sock[str(ip)], msg)
			t.sleep(3.5) #wait for some time for message to be sent completely

	def get_model_from_server():

		hostnanme = sk.gethostname()
		hostip = config.EDGE_SERVERS[EDGE_SERVERS]
		msg = self.recv_msg(self.edge_server_sock[hostip], 'MSG_UNINET_FROM_SERVER')
		self.uninet.load_state_dict(msg[1])   #UPDATE UNIT MODEL

	def initialize(self, split_layers, offload, first, LR):
		msg = ['MSG_UNINET_FROM_SERVER', self.uninet.state_dict()]
		#send all the messages
		for ip in self.edge_server_sock:
			print('sending weights back')
			print(self.edge_server_sock[str(ip)])
			#t.sleep(3.5)
			self.send_msg(self.edge_server_sock[str(ip)], msg)
			t.sleep(3.5) #wait for some time for message to be sent completely
			#self.edge_server_sock[str(ip)].close()

		
	def initializebackup(self, split_layers, offload, first, LR):
		if offload or first:
			self.split_layers = split_layers
			self.nets = {}
			self.optimizers= {}
			for i in range(len(split_layers)):
				client_ip = config.CLIENTS_LIST[i]
				if split_layers[i] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg, self.sock)
					msg = ['MSG_UNINET_FROM_SERVER', self.nets[client_ip]]
					self.send_msg(self.client_socks[client_ip], msg)
					#offloading weight in server also need to be initialized from the same global weight
					cweights = utils.get_model('Client', self.model_name, split_layers[i], self.device, config.model_cfg).state_dict()
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
					self.nets[client_ip].load_state_dict(pweights)

					self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
					  momentum=0.9)
				else:
					self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)
			self.criterion = nn.CrossEntropyLoss()
		if self.migration == False:
			msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
			for i in self.client_socks:
				self.send_msg(self.client_socks[i], msg)

	def train(self, thread_number, client_ips):
		
		self.net_threads = {}
		self.bandwidth = {}
		if self.migration == False:
			for i in range(len(client_ips)):
			
				self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ips[i],))
				#self._thread_network_testing(client_ips[i])
				self.net_threads[client_ips[i]].start()

			for i in range(len(client_ips)):
				self.net_threads[client_ips[i]].join()

			
			for s in self.client_socks:
				msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK')
				self.bandwidth[msg[1]] = msg[2]

		# Training start
		self.threads = {}
		for i in range(len(client_ips)):
			print('!!!!!!!!!!!!!!!!!!')
			print(len(client_ips))
			print('!!!!!!!!!!!!!!!!!!')
			if config.split_layer[i] == (config.model_len -1):
				#self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ips[i],))
				self._thread_training_no_offloading(client_ips[i])
				logger.info(str(client_ips[i]) + ' no offloading training start')
				#self.threads[client_ips[i]].start()
			else:
				logger.info(str(client_ips[i]))
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading, args=(client_ips[i],))
				#self._thread_training_offloading(client_ips[i])
				logger.info(str(client_ips[i]) + ' offloading training start')
				self.threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.threads[client_ips[i]].join()

		print('Threads Joined')
		self.ttpi = {} # Training time per iteration
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TRAINING_TIME_PER_ITERATION')
			self.ttpi[msg[1]] = msg[2]

		#self.group_labels = self.clustering(self.ttpi, self.bandwidth)
		#print(self.group_labels)
		#self.offloading = self.get_offloading(self.split_layers)
		#state = self.concat_norm(self.ttpi, self.offloading)
		self.reset_connection()

		#return state, self.bandwidth
		return self.bandwidth

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg)

	def _thread_training_no_offloading(self, client_ip):
		pass

	def _thread_training_offloading(self, client_ip):
		iteration = int((config.N / (config.K * config.B)))
		migration_sender = True
		run_network = True

		for i in range(iteration):
			if run_network:
				msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
				smashed_layers = msg[1]
				labels = msg[2]

				inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
				self.optimizers[client_ip].zero_grad()
				outputs = self.nets[client_ip](inputs)
				#print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				#print(self.nets[client_ip])
				loss = self.criterion(outputs, targets)
				loss.backward()
				self.optimizers[client_ip].step()

			# Send gradients to client
			#client is already waiting for the message so best point to migrate
				#self.testMigration(inputs,targets,client_ip)
			
			if False:
				
				
				#checkpoint the iteration count, weights and optimizer
				if migration_sender:
					
					checkpoint = {
						'epoch': i + 1,
						'state_dict': self.nets[client_ip].cpu().state_dict(),
						'optimizer': self.optimizers[client_ip].state_dict(),
						#'inputs': inputs,
						'loss' : loss,
						'grad' : inputs.grad,
						#'targets': targets,
						#'time': str(time()) # Edge migration Time
						}
					torch.save(checkpoint, 'model_test_size.pth') # Save the checkpoint data in a file to check the size
					#migrate to 192.168.0.40
					self.connect_server()
					msg = ['MSG_LOCAL_WEIGHTS_SERVER1_TO_SERVER2', checkpoint] #send weights to new server
					self.send_msg(self.sock, msg)
					print(msg[1])
					print('migration completed')
					torch.save(checkpoint, 'model_test_size.pth') # Save the checkpoint data in a file to check the size
					
					#inform client
					t.sleep(5)
					msg = ['MIGRATION_TO','192.168.0.40']
					print('sending client to 40')
					self.send_msg(self.client_socks[client_ip], msg)
					print('message sent')

				else:
					self.reset_connection()
					self.sock.bind((self.ip, self.migration_port))
					logger.info("Waiting for migration server to connect.")
					(mig_s1, (ip, port)) = self.sock.accept()
					logger.info('Got connection from ' + str(ip))
					msg = self.recv_msg(mig_s1)[1]
					self.nets[client_ip], self.optimizers[client_ip], i, inputs, loss, gradd, servertime = load_ckp(msg, self.nets[client_ip], self.optimizers[client_ip])
					timediff = time() - float(servertime) # Calculate Time difference 
					logger.info("total migration time = "+str(timediff))
					inputs.grad = gradd
					run_network = True
					#restore connection with client
					self.reset_connection()
					self.sock.bind((self.ip, self.port))
					
				
			msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
			self.send_msg(self.client_socks[client_ip], msg)

		logger.info(str(client_ip) + ' offloading training end')
		return 'Finish'

	def aggregate(self, client_ips):
		w_local_list =[]
		self.sock.bind((self.ip, self.port))
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
			if config.split_layer[i] != (config.model_len -1):
				w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[client_ips[i]].state_dict()),config.N / config.K)
				w_local_list.append(w_local)
			else:
				w_local = (msg[1],config.N / config.K)
				w_local_list.append(w_local)
		zero_model = utils.zero_init(self.uninet).state_dict()
		aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
		
		self.uninet.load_state_dict(aggregrated_model)
		self.reset_connection()
		print('aggregation finished')
		return aggregrated_model
	
	def aggregateEdges(self):
		#self.reset_connection()
		#self.sock.bind((self.ip, self.port))
		self.establish_connection_with_edge_servers()
		w_local_list =[]
		msg = {}
		#self.sock.bind((self.ip, self.port))
		for s in config.EDGE_SERVERS:
			serverip = config.EDGE_SERVERS[s]
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			#self.sock.connect((config.EDGE_SERVERS[s],self.port))
			print(self.edge_server_sock[serverip])
			msg[serverip] = self.recv_msg(self.edge_server_sock[serverip], 'MSG_EDGE_WEIGHTS_TO_SERVER')[1]
			print(msg[serverip])
			#w_local = (msg[1],config.N / config.K)
			#w_local_list.append(w_local)
		
		for i in msg.keys():
			print(i)
			print(type(msg[i]))
			for j in msg[i].keys():
				m = msg[i][j]['msg']
				nets = msg[i][j]['nets']
				num = msg[i][j]['N']
				w_local = (utils.concat_weights(self.uninet.state_dict(),m,nets),num)
				#w_local = (utils.concat_weights(self.uninet.state_dict(),m,nets),config.N / config.K)
				w_local_list.append(w_local)

			
		zero_model = utils.zero_init(self.uninet).state_dict()
		aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
		self.uninet.load_state_dict(aggregrated_model)
		#self.reset_connection()
		print('aggregation finished')
		print(self.uninet)
		t.sleep(3.5) #wait for some time before testing

		return aggregrated_model

	def testMigration(self, inputs, targets, client_ip):
		self.nets[client_ip].eval()
		correct = 0
		with torch.no_grad():
			outputs = self.nets[client_ip](inputs)
		_, predicted = outputs.max(1)
		
		correct += predicted.eq(targets).sum().item() # Batch size=100, targets=3, classes=10, GT= cat=3, input=60,000
		acc = 100.*correct/targets.size(0)
		logger.info('Accuracy: {}'.format(acc))
		self.nets[client_ip].train()
			
	def printUnit(self):
		print(self.uninet)

	def test(self, r):
		self.uninet.eval()
		test_loss = 0
		correct = 0
		total = 0
		
		with torch.no_grad():
			
			print(len(self.testloader))
			
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
				
				inputs, targets = inputs.to(self.device), targets.to(self.device)
			
				outputs = self.uninet(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		logger.info('Test Accuracy: {}'.format(acc)) 
		# Save checkpoint.
		torch.save(self.uninet.state_dict(), './'+ config.model_name +'.pth')

		return acc

	def clustering(self, state, bandwidth):
		#sort bandwidth in config.CLIENTS_LIST order
		bandwidth_order =[]
		for c in config.CLIENTS_LIST:
			bandwidth_order.append(bandwidth[c])

		#labels = [0,0,1,0,0] # Previous clustering results in RL
		labels = [0] # Previous clustering results in RL
		for i in range(len(bandwidth_order)):
			if bandwidth_order[i] < 5:
				#labels[i] = 2 # If network speed is limited under 5Mbps, we assign the device into group 2
				labels[i] = 0 # If network speed is limited under 5Mbps, we assign the device into group 2

		return labels

	def adaptive_offload(self, agent, state):
		action = agent.exploit(state)
		action = self.expand_actions(action, config.CLIENTS_LIST)

		config.split_layer = self.action_to_layer(action)
		logger.info('Next Round OPs: ' + str(config.split_layer))

		msg = ['SPLIT_LAYERS',config.split_layer]
		self.scatter(msg)
		return config.split_layer

	def expand_actions(self, actions, clients_list): # Expanding group actions to each device
		full_actions = []

		for i in range(len(clients_list)):
			full_actions.append(actions[self.group_labels[i]])

		return full_actions

	def action_to_layer(self, action): # Expanding group actions to each device
		#first caculate cumulated flops
		model_state_flops = []
		cumulated_flops = 0

		for l in config.model_cfg[config.model_name]:
			cumulated_flops += l[5]
			model_state_flops.append(cumulated_flops)

		model_flops_list = np.array(model_state_flops)
		model_flops_list = model_flops_list / cumulated_flops

		split_layer = []
		for v in action:
			idx = np.where(np.abs(model_flops_list - v) == np.abs(model_flops_list - v).min()) 
			idx = idx[0][-1]
			if idx >= 5: # all FC layers combine to one option
				idx = 6
			split_layer.append(idx)
		return split_layer

	def concat_norm(self, ttpi, offloading):
		ttpi_order = []
		offloading_order =[]
		for c in config.CLIENTS_LIST:
			ttpi_order.append(ttpi[c])
			offloading_order.append(offloading[c])

		group_max_index = [0 for i in range(config.G)]
		group_max_value = [0 for i in range(config.G)]
		
		for i in range(len(config.CLIENTS_LIST)):
			label = self.group_labels[i]
			print(self.group_labels)
			if ttpi_order[i] >= group_max_value[label]:
				group_max_value[label] = ttpi_order[i]
				group_max_index[label] = i

		ttpi_order = np.array(ttpi_order)[np.array(group_max_index)]
		offloading_order = np.array(offloading_order)[np.array(group_max_index)]
		state = np.append(ttpi_order, offloading_order)
		return state

	def get_offloading(self, split_layer):
		offloading = {}
		workload = 0

		assert len(split_layer) == len(config.CLIENTS_LIST)
		for i in range(len(config.CLIENTS_LIST)):
			for l in range(len(config.model_cfg[config.model_name])):
				if l <= split_layer[i]:
					workload += config.model_cfg[config.model_name][l][5]
			offloading[config.CLIENTS_LIST[i]] = workload / config.total_flops
			workload = 0

		return offloading


	def reinitialize(self, split_layers, offload, first, LR):
		#self.sock.close()
		#self.reset_connection()
		#self.reset_connection()
		#self.sock.bind((self.ip, self.port))
		self.establish_connection_with_edge_servers()

		self.initialize(split_layers, offload, first, LR)

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)
	
	def connect_server(self, serverip = '192.168.0.40',serv_port=51001):
		self.reset_connection()
		#self.sock.bind(('192.168.0.43', serv_port))
		logger.info('Connecting to Server2.')
		self.sock.connect(('192.168.0.40',51001))
	
	def save_ckp(self, state, checkpoint_dir):
		f_path = checkpoint_dir / 'checkpoint.pt'
		torch.save(state, f_path)
	
	def load_ckp(self, checkpoint, model, optimizer):
		
		#checkpoint = torch.load(checkpoint_fpath)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		
		return model, optimizer, checkpoint['epoch'], checkpoint['inputs'], checkpoint['loss'], checkpoint['grad'], checkpoint['targets'], checkpoint['time']
	
	def sendaggtoserver(self):
		self.reset_connection()
		self.sock.connect((config.UNIT_MODEL_SERVER,self.port))
		msg = ['MIGRATION_TO','192.168.0.40']
		print('sending client to 40')
		self.send_msg(self.UNIT_MODEL_SERVER, msg)




