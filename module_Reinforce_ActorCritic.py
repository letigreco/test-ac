"""Implement:
	- threshold che aumenta man mano automaticamete, sulla base delle performance
	- Salvataggio del progresso dell'optimizer per tuning su modello pre-trainato (x)
	- Plot multiplo
	- Domain randomization
"""
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import os
import random
from tqdm import tqdm
import time
import csv
#Problem specific
import torch
import gym
from env.custom_hopper import *
from classes import Agent, Policy, Value



def train(type_alg, hopper='S', n_episodes=5e4, trained_model=None, baseline=0, gamma=0.99, alpha=0.9, optim_lr=1e-3, layer_size=64, starting_threshold=700, csv_name='results.csv', save_every=75, print_every=1e4, print_name=True, plot=True, random_state=42, device='cpu'):
	"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE or Actor-critic algorithms"""
	# Seed setting
	random.seed(random_state)
	np.random.seed(random_state)
	torch.manual_seed(random_state)
	torch.cuda.manual_seed_all(random_state)

	# Make directory if it does not exist
	Path.mkdir(Path('./models'),exist_ok=True)
	Path.mkdir(Path('./plots'),exist_ok=True)
	csv_path = os.path.join(os.getcwd(), csv_name)
	if not os.path.exists(csv_path):
		print(f"Creation of new CSV file {csv_name}.")
		fields = ['model_name','type_alg','hopper','n_episodes','trained_model','baseline','gamma','optim_lr','layer_size','save_every','random_state','returns','returns_AvgLast','returns_AvgBeginning','times','times_AvgLast','times_AvgBeginning', 'tot_time']
		with open(csv_name, 'a') as f:
			writer = csv.writer(f)
			writer.writerow(fields)
		
	#Define model name based on timestamp
	if type_alg==0:
		model_type='Reinforce'
	elif type_alg==1:
		model_type='ActorCritic1'
	elif type_alg==2:
		model_type='ActorCritic2'
	else:
		print('Algorithm (type_alg) not valid!')
		return
	model_name = f'{model_type}_{n_episodes}_b{baseline}_h{hopper}_rs{random_state}_'+datetime.now().strftime('%y%m%d_%H-%M-%S')

	#Make environment
	if hopper=='T':
		env = gym.make('CustomHopper-target-v0')
	elif hopper=='S':
		env = gym.make('CustomHopper-source-v0')
	else:
		print('Environment type (custom hopper) not valid!')
		return
	env.seed(random_state)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	#Create policy and value
	policy = Policy(observation_space_dim, action_space_dim, type_alg, layer_size=layer_size)
	if type_alg==1:
		value = Value(observation_space_dim, action_space_dim, layer_size=layer_size)
	else:
		value = None

	#Start from a pre-trained policy
	if trained_model:
		policy.load_state_dict(torch.load(trained_model), strict=True)
	
	#Create agent
	agent = Agent(type_alg, policy, value, device=device, baseline=baseline, gamma=gamma, alpha=alpha, optim_lr=optim_lr)

	#Initialize data structures
	returns = []
	average_returns = []
	average_beginning = []
	times = []
	average_times = []
	average_beg_times = []

	#Initialize useful variables
	every_return = np.zeros((save_every*2))
	every_time = np.zeros((save_every*2))
	avg_returns = 0
	avg_times = 0
	threshold_bool = False

	#Iterate over episodes (print progress bar in terminal)
	start_tot_time = time.time()
	for episode in tqdm(range(n_episodes)):
		start_time = time.time()
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state
		agent.reset_I()  # Reset I for type_alg=2 

		#Build trajectory
		while not done:  # Loop until the episode is over
			action, action_probabilities, _ = agent.get_action(state)
			previous_state = state
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			#Update policy
			if type_alg!=0:
				agent.update_policy()
			train_reward += reward
		traject_time = time.time()-start_time
		
		#Print progress and save partial model every print_every episodes
		if (episode+1)%print_every == 0:
			torch.save(agent.policy.state_dict(), "models/"+model_name+'.mdl')
			print('Training episode:', episode+1)
			print('Episode return:', train_reward)
		#Save a partial model if return over threshold
		if not threshold_bool and every_return.mean()>starting_threshold:
			print('Threshold reached')
			torch.save(agent.policy.state_dict(), f"models/{model_name}_t{episode}.mdl")
			threshold_bool = True

		#Update policy
		start2_time = time.time()
		if type_alg==0:
			start2_time = time.time()
			agent.update_policy()
			update_time = time.time() - start2_time
		else:
			update_time = 0.0

		episode_time = traject_time + update_time


		
		#Save data
		episode_time = traject_time + (time.time()-start2_time)
		every_time[(episode+1)%(save_every*2)-1]=episode_time
		avg_times = avg_times*(episode/(episode+1))+episode_time/(episode+1)
		every_return[(episode+1)%(save_every*2)-1]=train_reward
		avg_returns = avg_returns*(episode/(episode+1))+train_reward/(episode+1)
		if (episode+1)%save_every == 0:
			returns.append(train_reward)
			average_returns.append(every_return.mean())
			average_beginning.append(avg_returns)
			times.append(episode_time)
			average_times.append(every_time.mean())
			average_beg_times.append(avg_times)
	tot_time = time.time()-start_tot_time
	
	#Print the average of the last episodes' returns
	print(f'Average of the last {save_every*2} returns: {average_returns[-1]}')

	#Save final model (overwrite privious savings) and data
	torch.save(agent.policy.state_dict(), "models/"+model_name+'.mdl')
	fields=[model_name,type_alg,hopper,n_episodes, trained_model, baseline, gamma, optim_lr, layer_size, save_every, random_state, returns, average_returns, average_beginning, times, average_times, average_beg_times, tot_time]
	with open(csv_name, 'a') as f:
		writer = csv.writer(f)
		writer.writerow(fields)

	#Print model name if desired
	if print_name:
		print()
		print(f'MODEL NAME: {model_name}.mdl')
		print()

	#Plot progress if desired
	if plot:
		plot_returns_times(save_every,np.array(returns),np.array(average_returns),np.array(average_beginning), save_every, model_name, 'return')
		plot_returns_times(save_every,np.array(times),np.array(average_times),np.array(average_beg_times), save_every, 'time_'+model_name, 'time')

	#Return results
	returns_array = np.vstack((returns,average_returns,average_beginning))
	times_array = np.vstack((times,average_times,average_beg_times))
	return returns_array, times_array, tot_time, model_name+'.mdl'


##############################################################################

def test(type_alg, model, hopper='T', n_episodes=10, render=False, gamma=0.99, optim_lr=1e-3, layer_size=64, random_state=42, device='cpu'):  ### #TODO CHANGE DEFAULT render TO FALSE and episodes to 50
	"""Test an RL agent on the OpenAI Gym Hopper environment"""

	# Seed setting
	random.seed(random_state)
	np.random.seed(random_state)
	torch.manual_seed(random_state)
	torch.cuda.manual_seed_all(random_state)

	#Make environment
	if hopper=='T':
		env = gym.make('CustomHopper-target-v0')
	elif hopper=='S':
		env = gym.make('CustomHopper-source-v0')
	else:
		print('Environment type (custom hopper) not valid!')
		return
	env.seed(random_state)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	#Create and load policy
	policy = Policy(observation_space_dim, action_space_dim, type_alg, layer_size=layer_size)
	policy.load_state_dict(torch.load('models/'+model), strict=True)

	#Create agent
	agent = Agent(type_alg, policy, device=device, gamma=gamma, optim_lr=optim_lr)  #TODO params: type_alg, policy, value = None, device='cpu', baseline=0

	returns = []
	#Iterate over episodes
	for episode in range(n_episodes):
		done = False
		test_reward = 0
		state = env.reset()

		#Build trajectory
		while not done:
			action, _ = agent.get_action(state, evaluation=True)
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			if render:  
				env.render()  #Show rendering
			test_reward += reward

		#Print results
		print(f"Episode: {episode} | Return: {test_reward}")
		returns.append(test_reward)

	return returns


##############################################################################

def plot_returns_times(save_every,return_array, average_array, beginning_array, points, name, metric):
	"""Plot progress of return over episodes"""
	num_returns = len(return_array)
	numbers_array = np.arange(save_every, save_every * (num_returns + 1), save_every)
	plt.figure(figsize=(12,10))
	plt.title(metric.upper())
	plt.plot(numbers_array, return_array, c='lightsteelblue', label=f'Episode {metric} (every {points})')
	plt.plot(numbers_array[1:], average_array[1:], c='red', linestyle='--', label=f'Average over last {points*2} episodes')
	plt.plot(numbers_array, beginning_array, c='lime', linestyle='--', label='Average over episodes from beginning')
	plt.xlabel('Episode')
	plt.ylabel(metric.capitalize())
	plt.grid()
	plt.legend()
	plt.savefig("./plots/"+name+'_Return.png',dpi=300)

