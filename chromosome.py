import numpy as np
from gene import Gene, FloatGene, CategoricalGene
from typing import List, Union

class Chromosome:
	def __init__(self, genes:List[Gene]=[]) -> None:
		self.genes = genes

	@staticmethod
	def crossover(chromosome1, chromosome2):
		raise NotImplementedError

	def mutate(self, mutation_prob:float):
		raise NotImplementedError

	def get_gene_values(self) -> List[Union[float, str]]:
		return list(map(lambda x: x.get_value(), self.genes))

	def add_gene(self, new_gene:Gene) -> None:
		if self.genes is None:
			self.genes = [new_gene]
		else:
			self.genes.append(new_gene)

	def set_genes(genes:List[Gene]) -> None:
		self.genes = genes

	def get_genes(self) -> List[Gene]:
		return self.genes

class FloatChromosome(Chromosome):
	def __init__(self, genes:List[FloatGene]=[]) -> None:
		super(FloatChromosome, self).__init__(genes)

	@staticmethod
	def crossover(chromosome1:Chromosome, chromosome2:Chromosome) -> List[Chromosome]:
		raise NotImplementedError

	def mutate(self, mutation_prob:float) -> Chromosome:
		raise NotImplementedError

class CategoricalChromosome(Chromosome):
	def __init__(self, genes:List[CategoricalGene]=[]) -> None:
		super(CategoricalChromosome, self).__init__(genes)

	def initialize(self, num_of_genes:int, VAL_SET:List[str]) -> None:
		self.genes = [CategoricalGene(None, VAL_SET) for i in range(num_of_genes)]
		[gene.set_value_randomly() for gene in self.genes]

	@staticmethod
	def crossover(chromosome1:Chromosome, chromosome2:Chromosome) -> List[Chromosome]:
		min_length = min(len(chromosome1.genes), len(chromosome2.genes))
		r = np.random.randint(0, min_length, size=2)

		if r[0] > r[1]:
			r[0], r[1] = r[1], r[0]

		new_chromosome1_genes = chromosome1.genes[:r[0]] + chromosome2.genes[r[0]:r[1]] + chromosome1.genes[r[1]:]
		new_chromosome2_genes = chromosome2.genes[:r[0]] + chromosome1.genes[r[0]:r[1]] + chromosome2.genes[r[1]:]

		new_chromosome1 = CategoricalChromosome(new_chromosome1_genes)
		new_chromosome2 = CategoricalChromosome(new_chromosome2_genes)

		return [new_chromosome1, new_chromosome2]

	def mutate(self, mutation_prob:float) -> Chromosome:
		new_chromosome = CategoricalChromosome(self.genes)

		for i, gene in enumerate(new_chromosome.genes):
			if np.random.uniform() < mutation_prob:
				index = np.random.choice(len(gene.VAL_SET), size=1)[0]
				new_val = gene.VAL_SET[index]
				gene.set_value(new_val)

		return new_chromosome