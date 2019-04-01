from ea import EvolutionaryAgent, CategoricalAgent
from env import Environment, WeaselEnvironment
from chromosome import Chromosome, CategoricalChromosome
import heapq
import statistics
import numpy as np
from typing import Dict, List, Union, Tuple

class EvolutionaryAlgorithm:
	def __init__(self, env:Environment, agents:List[EvolutionaryAgent]) -> None:
		self.env = env
		self.agents = agents

	def initialization(self) -> None:
		raise NotImplementedError

	def fitness(self) -> List[float]:
		raise NotImplementedError

	def selection(self) -> List[EvolutionaryAgent]:
		raise NotImplementedError

	def crossover(self) -> List[EvolutionaryAgent]:
		raise NotImplementedError

	def mutate(self) -> List[EvolutionaryAgent]:
		raise NotImplementedError

	def next_generation(self) -> List[EvolutionaryAgent]:
		raise NotImplementedError

	def set_env(self, env:Environment) -> None:
		self.env = env

	def get_env(self) -> Environment:
		return self.env

	def set_agents(self, agents:List[EvolutionaryAgent]) -> None:
		self.agents = agents

	def get_agents(self) -> List[EvolutionaryAgent]:
		return self.agents

class WeaselAlgorithm(EvolutionaryAlgorithm):
	def __init__(	self, 	
					env:WeaselEnvironment, 
					agents:List[CategoricalAgent],
					elitism_proportion:float,
					wheel_proportion:float,
					mutation_prob:float,
					wheel_count:Union[int, None]=None,
					elitism_count:Union[int, None]=None) -> None:

		super(WeaselAlgorithm, self).__init__(env, agents)

		self.elitism_proportion = elitism_proportion
		self.elitism_count = elitism_count
		self.wheel_proportion = wheel_proportion
		self.wheel_count = wheel_count
		self.mutation_prob = mutation_prob

		self.population_size = len(agents)
		self.elitism_size = self._calculate_size(self.elitism_proportion, self.elitism_count, self.population_size)
		self.wheel_size = self._calculate_size(self.wheel_proportion, self.wheel_count, self.population_size)
		self.breed_size = self.population_size - self.elitism_size - self.wheel_size

		self.initialize_history()

	def initialize_history(self) -> None:
		self.history = []

	def _calculate_size(self, proportion:float, count:Union[int, None], size:int) -> int:
		size_out = int(round(proportion*size)) if count is None else count
		return size_out

	def initialization(self) -> None:
		raise NotImplementedError

	def fitness(self, agents:List[CategoricalAgent]=None) -> List[float]:
		if agents is None:
			agents = self.agents

		fitnesses = [self.env.get_fitness(agent) for agent in agents]
		return fitnesses
	
	def _normalize_fitnesses(self, fitnesses:List[float]) -> List[float]:
		summ = sum(fitnesses)
		return [fitness/summ for fitness in fitnesses]

	def selection(self, fitnesses:List[float], elitism_size:int, wheel_size:int, agents:List[CategoricalAgent]=None) -> Tuple[List[CategoricalAgent], List[CategoricalAgent]]:
		if agents is None:
			agents = self.agents

		elitism_selected = []
		wheel_selected = []

		if elitism_size > 0:
			elitism_selected = self._elitism(fitnesses, elitism_size, agents)
		if wheel_size > 0:
			wheel_selected = self._roulette_wheel_selection(fitnesses, wheel_size, agents)

		return elitism_selected, wheel_selected

	def _elitism(self, fitnesses:List[float], count:int, agents:List[CategoricalAgent]=None) -> List[CategoricalAgent]:
		if agents is None:
			agents = self.agents
		# indices = heapq.nlargest(count, range(len(fitnesses)), fitnesses.take)
		indices = heapq.nlargest(count, range(len(fitnesses)), fitnesses.__getitem__)
		selected = [agents[i] for i in indices]
		return selected

	def _roulette_wheel_selection(self, fitnesses:List[float], count:int, agents:List[CategoricalAgent]=None) -> List[CategoricalAgent]:
		if agents is None:
			agents = self.agents
		# probs = self._normalize_fitnesses(fitnesses)
		# indices = np.random.choice(range(len(agents)), size=count, replace=False, p=probs)
		probs = self._normalize_fitnesses(fitnesses)
		nonzero_indices = np.nonzero(probs)[0]
		nonzero_probs = [probs[i] for i in nonzero_indices]
		is_replace = True if len(nonzero_indices) < count else False
		indices = np.random.choice(nonzero_indices, size=count, replace=is_replace, p=nonzero_probs)
		selected = [agents[i] for i in indices]
		return selected

	def crossover(self, count:int, agents:List[CategoricalAgent]=None) -> List[CategoricalAgent]:
		if agents is None:
			agents = self.agents

		new_agents = []
		for i in range(int(round(count/2))):
			indices = np.random.choice(range(len(agents)), size=2, replace=False)
			agent1, agent2 = agents[indices[0]], agents[indices[1]]
			new_agents += CategoricalAgent.crossover(agent1, agent2)
		
		return new_agents

	def mutate(self, mutation_prob:float, agents:List[CategoricalAgent]=None) -> List[CategoricalAgent]:
		if agents is None:
			agents = self.agents

		new_agents = [agent.mutate(mutation_prob) for agent in agents]
		return new_agents

	def next_generation(self, elitism_size:int, wheel_size:int, breed_size:int, mutation_prob:float, agents:List[CategoricalAgent]=None) -> List[CategoricalAgent]:
		if agents is None:
			agents = self.agents

		fitnesses = self.fitness(agents)
		self.log_generation(fitnesses)
		elitism_agents, wheel_selected = self.selection(fitnesses, elitism_size, wheel_size, agents)
		breeded_agents = self.crossover(breed_size, wheel_selected)
		all_agents = self.mutate(mutation_prob, agents=elitism_agents+breeded_agents)

		return all_agents, elitism_agents[0].chromosome.get_gene_values()

	def next_generation_with_elitism_and_mutation(self, elitism_size:int, mutation_prob:float, agents:List[CategoricalAgent]=None) -> List[CategoricalAgent]:
		if agents is None:
			agents = self.agents

		fitnesses = self.fitness(agents)
		self.log_generation(fitnesses)
		elitism_agents, _ = self.selection(fitnesses, elitism_size, 0, agents)
		
		mutated_agents = []
		for i in range(int(round(self.population_size/self.elitism_size))-1):
			mutated_agents += self.mutate(mutation_prob, agents=elitism_agents)

		all_agents = elitism_agents + mutated_agents
		return all_agents, elitism_agents[0].chromosome.get_gene_values()

	def log_generation(self, fitnesses:List[float], gen_count:int=0) -> None:
		min_f, max_f, mean_f, median_f = min(fitnesses), max(fitnesses), statistics.mean(fitnesses), statistics.median(fitnesses)
		self.history.append({"gen_count":gen_count, "min_fitness":min_f, "max_fitness":max_f, "mean_fitness":mean_f, "median_fitness":median_f})

	def run(self, num_of_generations:int=100, is_verbose:bool=True) -> List[Dict[str, float]]:
		self.initialize_history()
		i_gen = 1
		while True:
			if i_gen > num_of_generations:
				break
			
			self.agents, best_str = self.next_generation(self.elitism_size, self.wheel_size, self.breed_size, self.mutation_prob, self.agents)

			if is_verbose:
				print("-"*30)
				print(f"Generation {i_gen}:")
				print(''.join(best_str))
				print(f"min = {self.history[-1]['min_fitness']}")
				print(f"max = {self.history[-1]['max_fitness']}")
				print(f"mean = {self.history[-1]['mean_fitness']}")
				print(f"median = {self.history[-1]['median_fitness']}")

			i_gen += 1

	def run2(self, num_of_generations:int=100, is_verbose:bool=True) -> List[Dict[str, float]]:
		self.initialize_history()
		i_gen = 1
		while True:
			if i_gen > num_of_generations:
				break
			
			self.agents, best_str = self.next_generation_with_elitism_and_mutation(self.elitism_size, self.mutation_prob, self.agents)

			if is_verbose:
				print("-"*30)
				print(f"Generation {i_gen}:")
				print(''.join(best_str))
				print(f"min = {self.history[-1]['min_fitness']}")
				print(f"max = {self.history[-1]['max_fitness']}")
				print(f"mean = {self.history[-1]['mean_fitness']}")
				print(f"median = {self.history[-1]['median_fitness']}")

			i_gen += 1


if __name__ == '__main__':
	import string

	population_size = 100
	# CHARSET = string.ascii_letters + string.digits
	CHARSET = string.ascii_uppercase + ' ' 

	target_str = "METHINKS IT IS LIKE A WEASEL"
	num_of_genes = len(target_str)

	elitism_proportion = 0.1
	wheel_proportion = 0.0
	mutation_prob = 0.1

	env = WeaselEnvironment(target_str)
	# agents = [CategoricalAgent().initialize(num_of_genes, CHARSET) for i in range(population_size)]
	agents = [CategoricalAgent() for i in range(population_size)]
	[agent.initialize(num_of_genes, CHARSET) for agent in agents]

	algo = WeaselAlgorithm(env, agents, elitism_proportion, wheel_proportion, mutation_prob)
	algo.run2(100)
	