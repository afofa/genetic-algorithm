import numpy as np
from typing import List

class Gene:
	def __init__(self) -> None:
		pass

class FloatGene(Gene):
	def __init__(self, value:float, lower:float=None, upper:float=None) -> None:
		super(FloatGene, self).__init__()
		self.lower = lower
		self.upper = upper
		self.value = self.set_value(value)

	def set_value(self, value:float) -> None:
		# if self.lower is not None and value < self.lower:
		# 	value = self.lower

		# if self.upper is not None and value > self.upper:
		# 	value = self.upper
		value = np.clip(value, lower, upper)
		self.value = value

	def get_value(self) -> float:
		return self.value
	
	def set_upper(self, upper:float) -> None:
		self.upper = upper

	def get_upper(self) -> float:
		return self.upper

	def set_lower(self, lower:float) -> None:
		self.lower = lower

	def get_lower(self) -> float:
		return self.lower

class CategoricalGene(Gene):
	def __init__(self, value:str, VAL_SET:List[str]) -> None:
		super(CategoricalGene, self).__init__()
		self.VAL_SET = VAL_SET
		self.set_value(value)

	def set_value_randomly(self, probs:List[float]=None) -> None:
		if probs is not None and len(probs) != len(self.VAL_SET):
			probs = None
		index = np.random.choice(len(self.VAL_SET), size=1, replace=True, p=probs)[0]
		self.value = self.VAL_SET[index]

	def set_value(self, value:str) -> None:
		if isinstance(value, str) and value in self.VAL_SET:
			self.value = value
		else:
			self.value = None

	def get_value(self) -> str:
		return self.value

	def set_VAL_SET(self, VAL_SET:List[str]) -> None:
		self.VAL_SET = VAL_SET

	def get_VAL_SET(self) -> List[str]:
		return self.VAL_SET