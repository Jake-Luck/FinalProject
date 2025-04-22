"""
Provides Genetic base class and those that inherit it.
"""
from algorithms.algorithm import Algorithm
from core.plotting import Plotting

from abc import abstractmethod
import numpy as np
from numpy import ndarray  # For type hints
import random


class Genetic(Algorithm):
    """
    Base class for genetic algorithms, includes abstract method for crossover.
    """
    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 crossover_probability: float,
                 mutation_probability: float,
                 generations_per_update: int | None = 1,
                 plotting: bool = True,
                 random_seed: int | None = None):
        """
        Initialises genetic class with given parameters.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param generations_per_update: Number of generations between each
        progress update. If None or less than 0, no updates given.
        :param plotting: Whether to display plots on each update.
        :param random_seed: Specified seed for random number generators.
        """
        if num_generations < 1:
            raise ValueError("num_generations must be at least 1.")
        if population_size < 3:
            raise ValueError("population_size must be at least 3.")

        self.num_generations = num_generations
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.plotting = plotting

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        if (not isinstance(generations_per_update, int)
                or generations_per_update < 1):
            generations_per_update = None
        self.generations_per_update = generations_per_update

    @abstractmethod
    def _crossover(self,
                   parent1: ndarray,
                   parent2: ndarray) -> ndarray:
        """
        Create offspring by performing crossover on two given individuals.

        Abstract method to be implemented by subclasses.
        :param parent1: First parent's genome.
        :param parent2: Second parent's genome.
        :return: Offspring of each parent.
        """
        pass
