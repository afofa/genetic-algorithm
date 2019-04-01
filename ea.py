import numpy as np
from chromosome import Chromosome, FloatChromosome, CategoricalChromosome
from typing import List

class EvolutionaryAgent:
	def __init__(self, chromosome:Chromosome) -> None:
		self.chromosome = chromosome

	@staticmethod
	def crossover(agent1, agent2):
		raise NotImplementedError

	def mutate(self, mutation_prob:float):
		# TODO: mutate inplace or return new EvolutionaryAgent???
		raise NotImplementedError

	def set_chromosome(self, chromosome:Chromosome) -> None:
		self.chromosome = chromosome

	def get_chromosome(self) -> Chromosome:
		return self.chromosome

class FloatAgent(EvolutionaryAgent):
	def __init__(self, chromosome:FloatChromosome) -> None:
		super(FloatAgent, self).__init__(chromosome)

	@staticmethod
	def crossover(agent1:EvolutionaryAgent, agent2:EvolutionaryAgent) -> List[EvolutionaryAgent]:
		pass

	def mutate(self, mutation_prob:float) -> EvolutionaryAgent:
		# TODO: mutate inplace or return new FloatAgent???
		raise NotImplementedError

class CategoricalAgent(EvolutionaryAgent):
	def __init__(self, chromosome:CategoricalChromosome=None) -> None:
		super(CategoricalAgent, self).__init__(chromosome)

	def initialize(self, num_of_genes:int, VAL_SET:List[str]) -> None:
		self.chromosome = CategoricalChromosome()
		self.chromosome.initialize(num_of_genes, VAL_SET)

	@staticmethod
	def crossover(agent1:EvolutionaryAgent, agent2:EvolutionaryAgent) -> List[EvolutionaryAgent]:
		 new_chromosomes_list = CategoricalChromosome.crossover(agent1.chromosome, agent2.chromosome)
		 new_chromosome1, new_chromosome2 = new_chromosomes_list[0], new_chromosomes_list[1]

		 new_agent1 = CategoricalAgent(new_chromosome1)
		 new_agent2 = CategoricalAgent(new_chromosome2)

		 return [new_agent1, new_agent2]

	def mutate(self, mutation_prob:float) -> EvolutionaryAgent:
		# TODO: mutate inplace or return new CategoricalAgent???
		new_chromosome = self.chromosome.mutate(mutation_prob)
		new_agent = CategoricalAgent(new_chromosome)

		return new_agent