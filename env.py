from ea import EvolutionaryAgent

class Environment:
	def __init__(self) -> None:
		pass

	def get_fitness(self, agent:EvolutionaryAgent) -> float:
		raise NotImplementedError

class WeaselEnvironment(Environment):
	def __init__(self, target:str) -> None:
		super(WeaselEnvironment, self).__init__()
		self.target = target

	def get_fitness(self, agent:EvolutionaryAgent) -> float:
		input_str = agent.chromosome.get_gene_values()
		return len([1 for i in range(len(input_str)) if input_str[i] == self.target[i]])