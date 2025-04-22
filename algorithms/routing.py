"""
Provides routing class that contains various routing algorithms.
"""
from algorithms.algorithm import Algorithm
from algorithms.genetic import Genetic
from core.plotting import Plotting

import numpy as np
from numpy import ndarray
import random


class Routing(Algorithm):
    """
    Provides various routing algorithms as static methods.
    """
    @staticmethod
    def gift_wrapping(num_locations: int,
                      num_days: int,
                      coordinates: ndarray,
                      graph: ndarray,
                      durations: ndarray) -> ndarray:
        """
        Finds a convex hull around the coordinates, then attempts to add
        the interior points in an order that minimises route length.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param coordinates: Coordinates of each location.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing the best route.
        """
        # Gets the westernmost point (guaranteed to be on outside).
        starting_index = np.argmin(coordinates[:, 0])
        hull = []

        current_index = starting_index

        while True:
            hull.append(current_index)
            next_index = (current_index + 1) % num_locations

            for i in range(num_locations):
                if i == current_index:
                    continue

                orientation = np.cross(
                    coordinates[next_index] - coordinates[current_index],
                    coordinates[i] - coordinates[current_index])

                # If point i is more counter-clockwise
                if orientation > 0:
                    next_index = i

            # If returned to start, finish
            if next_index == starting_index:
                break

            current_index = next_index

        all_points = np.arange(num_locations)
        interior_points = np.setdiff1d(all_points, hull)

        route = Routing.greedy_insertion(np.array(hull), interior_points,
                                         graph, durations)

        if num_days > 1:
            new_locations = np.zeros(num_days - 1)
            route = Routing.greedy_insertion(route, new_locations, graph,
                                             durations)

        # Shift the route so that it ends at 0
        route = np.roll(route, -np.where(route == 0)[0] - 1)
        return route


    @staticmethod
    def brute_force(num_locations: int,
                    num_days: int,
                    graph: ndarray,
                    durations: ndarray) -> ndarray:
        """
        A brute force solver for the problem. Can be used on its own or with
        clustering. When num_days equals 1, this becomes a TSP solver.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing the best route.
        """
        if num_days < 1:
            raise ValueError(f"num_days={num_days}, must be at least 1.")
        if num_locations < 4:
            # 3 locations only has 1 result, so just use greedy
            return Routing.greedy_routing(num_locations, num_days,
                                          graph, durations)

        route_length = num_locations + num_days - 1

        # get (route_length - 1)! Do -1 because all routes end at '0'
        n_routes = 1
        for i in range(2, route_length):
            n_routes *= i

        best_route = Algorithm.generate_route(0, n_routes,
                                              num_locations, num_days,
                                              route_length)
        best_evaluation, _, _ = Algorithm.evaluate_route(best_route, num_days,
                                                         graph, durations)

        # iterations_per_update = n_routes / 10
        # progress = 0

        for i in range(1, n_routes):  # yikes
            route = Algorithm.generate_route(i, n_routes, num_locations,
                                             num_days, route_length)
            evaluation, _, _ = Algorithm.evaluate_route(route, num_days, graph,
                                                        durations)

            if evaluation < best_evaluation:
                best_route = np.array(route, copy=True)
                best_evaluation = evaluation

            # if (i+1) % iterations_per_update == 0:
            #     progress += 10
            #     print(f"Brute force {progress}% complete: {i+1}/{n_routes}.
            #     Best evaluation: {best_evaluation}")

        return best_route

    @staticmethod
    def greedy_routing(num_locations: int,
                       num_days: int,
                       graph: ndarray,
                       durations: ndarray):
        """
        A very simple solver for use with clustering, will not find a valid
        route on its own. This is a conventional TSP solver.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location.
        :return: 1D np array representing the route found.
        """
        route_length = num_locations
        route = np.empty(route_length, dtype=int)
        index = 0
        working_graph = np.array(graph, copy=True)
        for i in range(route_length):
            working_graph[:, index] = np.iinfo(np.int32).max
            index = working_graph[index].argmin()
            route[i] = index

        if num_days > 1:
            new_locations = np.zeros(num_days-1)
            route = Routing.greedy_insertion(route, new_locations, graph,
                                             durations)

        return route

    @staticmethod
    def greedy_insertion(route: ndarray,
                         new_locations: ndarray,
                         graph: ndarray,
                         durations: ndarray) -> ndarray:
        """
        Iteratively inserts new locations into the route by finding the best
        insertion point at each iteration.
        :param route: 1D ndarray representing the current route.
        :param new_locations: 1D ndarray containing locations to insert.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location in minutes.
        :return: 1D ndarray representing the updated route with new locations.
        """
        num_days = np.count_nonzero(route == 0)
        num_days += np.count_nonzero(new_locations == 0)
        working_route = np.array(route, copy=True)

        # For each new location, find the best insertion point in the route

        for location in new_locations:
            best_route = np.insert(working_route, 0, location)
            best_evaluation, _, _ = Routing.evaluate_route(best_route, num_days,
                                                           graph, durations)

            for i in range(1, len(working_route) + 1):
                new_route = np.insert(working_route, i, location)
                evaluation, _, _ = Routing.evaluate_route(new_route, num_days,
                                                         graph, durations)

                if evaluation < best_evaluation:
                    best_route = new_route
                    best_evaluation = evaluation

            working_route = best_route

        return working_route


class GeneticRouting(Genetic):
    """
    Class for genetic routing. Genome contains the order of locations in the
    route.
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
        Initialises genetic routing with given parameters.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param generations_per_update: Number of generations between each
        progress update. If None or less than 0, no updates given.
        :param plotting: Whether to display plots on each update.
        :param random_seed: Specified seed for random number generators.
        """
        super().__init__(num_generations, population_size,
                         crossover_probability, mutation_probability,
                         generations_per_update, plotting, random_seed)

    def find_route(self,
                   num_locations: int,
                   num_days: int,
                   graph: ndarray,
                   durations: ndarray,
                   coordinates: ndarray | None = None) -> ndarray:
        """
        Generates a route that visits all locations using a genetic algorithm
        approach. Genome contains the order of locations in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param coordinates: Coordinates of each location. Used for plotting.
        :return: A route that visits all locations.
        """

        if coordinates is None and self.plotting is True:
            print("No coordinates provided, setting plotting to False")
            self.plotting = False

        evaluations = np.empty(self.population_size)
        evaluations[:] = float('inf')

        route_length = num_locations + num_days - 1

        n_routes = 1
        for i in range(2, route_length):
            n_routes *= i
        # Assign random routes to each location
        population = self._generate_random_routes(num_locations, num_days,
                                                  n_routes, route_length)

        # Assign these before loop just in case num_generations is 0 and these
        # are used before initialisation
        index1 = 0
        parent1 = population[index1]

        for generation_number in range(self.num_generations):
            evaluations = self._evaluate_population(population, num_days, graph,
                                                    durations)

            index1, index2 = np.argpartition(evaluations, 2)[:2]
            parent1 = population[index1]
            parent2 = population[index2]

            if self.generations_per_update is not None \
                    and generation_number % self.generations_per_update == 0:
                progress = 100 * generation_number/self.num_generations
                print(f"Route evolution {int(progress)}% complete: "
                      f"{generation_number}/{self.num_generations} completed. "
                      f"Best evaluation: {evaluations[index1]}")

                if self.plotting:
                    Plotting.display_route(coordinates, parent1)

            # Crossover
            population[0] = parent1
            population[1] = parent2

            for i in range(2, self.population_size):
                use_crossover = random.random() < self.crossover_probability

                # Generate random individual (increases genetic diversity)
                if not use_crossover:
                    route_number = random.randint(0, n_routes)
                    population[i] = self.generate_route(route_number, n_routes,
                                                        num_locations, num_days,
                                                        route_length)
                    continue

                population[i] = self._crossover(parent1, parent2)

                mutate = random.random() < self.mutation_probability
                if not mutate:
                    continue
                # swap two elements
                mutation_index1, mutation_index2 = random.sample(
                    range(num_locations), 2)
                temp = population[i, mutation_index1]
                population[i, mutation_index1] = population[i, mutation_index2]
                population[i, mutation_index2] = temp

        print(f"Route evolution complete. "
              f"Best evaluation: {evaluations[index1]}")
        return parent1

    def _crossover(self,
                   parent1: ndarray,
                   parent2: ndarray) -> ndarray:
        """
        Performs crossover between two given individuals.
        :param parent1: First parent's genome.
        :param parent2: Second parent's genome.
        :return: Offspring of each parent.
        """

        offspring = np.empty_like(parent1)
        offspring[:] = -1

        crossover_point = random.randint(0, len(parent1)-1)

        # Adds parent1 to offspring up to crossover point
        offspring[:crossover_point] = parent1[:crossover_point]

        days_left = (np.count_nonzero(parent1 == 0) -
                     np.count_nonzero(offspring == 0)) - 1

        # Adds remaining cities in order that they appear in parent2
        offspring_index = crossover_point
        for i in range(0, parent2.shape[0]-1):
            if parent2[i] == 0 and days_left > 0 or parent2[i] not in offspring:
                offspring[offspring_index] = parent2[i]
                offspring_index += 1
                if parent2[i] == 0:
                    days_left -= 1

        # Return to start at end of route
        offspring[offspring.shape[0]-1] = 0
        return offspring

    def _evaluate_population(self,
                             population: ndarray,
                             num_days: int,
                             graph: ndarray,
                             durations: ndarray) -> ndarray:
        """
        Evaluates the fitness of a given population.
        :param population: Population to evaluate.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing each individual's fitness.
        """
        evaluations = np.zeros(self.population_size)
        for individual in range(self.population_size):
            evaluations[individual], _, _ = self.evaluate_route(
                population[individual], num_days, graph, durations)
        return evaluations

    def _generate_random_routes(self,
                                num_locations: int,
                                num_days: int,
                                n_routes: int,
                                route_length: int) -> ndarray:
        """
        Generates a random route for each individual in the population.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param n_routes: Total number of possible routes
        :return: A population of random routes.
        """
        population = np.empty(shape=(self.population_size, route_length),
                              dtype=int)
        for i in range(self.population_size):
            route_number = random.randint(0, n_routes)
            population[i] = self.generate_route(route_number, n_routes,
                                                num_locations, num_days,
                                                route_length)
        return population
