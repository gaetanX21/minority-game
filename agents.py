"""
Module containing the agents for the Minority Game.
The agents are implemented using the Mesa framework.
The agents are:
- BasicAgent: classic Minority Game agent
- MarkovianAgent: Markovian Minority Game agent
- Producer: Producer Minority Game agent
- Speculator: Speculator Minority Game agent
"""

import mesa
import numpy as np


# Convert a binary array to its decimal equivalent (e.g. [0, 1, 1] -> 3)
bin2dec = lambda x: np.sum(np.array([2**i for i in reversed(range(x.shape[-1]))]) * x, axis=-1)


class BasicAgent(mesa.Agent):
    """Classic Minority Game agent."""

    def __init__(self, model: mesa.Model, M: int, S: int):
        super().__init__(model)
        self.M = M # Memory size
        self._strategies = self.model.rng.choice([-1, 1], size=(S, 2**M)).astype(np.int32) # Random strategies, fixed at creation
        self._run_strategies = lambda x: self._strategies[:, bin2dec(x)] # Function to run all strategies on input x
        self._strategies_fitness = np.zeros(S).astype(np.int32) # Fitness of each strategy, updated after each turn (even for unplayed strategies --> virtual update)
        self._current_choices = None # will store outcomes for all strategies for the current turn
        self.action = None # will store the chosen action for the current turn

    def _choose_action(self):
        """Get the outcomes for all strategies from given input and return chosen action"""
        x = self.model.get_history(self.M)
        # Compute outputs for all strategies
        self._current_choices = self._run_strategies(x)  
        # Chosen action comes from best strategy
        best_strategies = np.where(self._strategies_fitness == np.max(self._strategies_fitness))[0]
        chosen_strategy = self.model.rng.choice(best_strategies)
        self.action = self._current_choices[chosen_strategy]

    def _update_scores(self, outcome):
        """Update the agent's strategies' fitness values from current game outcome"""
        self._strategies_fitness += (self._current_choices == outcome).astype(np.int32)


class MarkovianAgent(mesa.Agent):
    """
    Markovian Minority Game agent.
    Fixed S=2 (lower complexity for same qualitative results).
    Similar to BasicAgent but indexed strategies with an integer instead of a binary array.
    Also, lighter implementation to save computation time.
    """

    def __init__(self, model: mesa.Model, P: int):
        super().__init__(model)
        S = 2
        self._strategies = self.model.rng.choice([-1, 1], size=(S,P)).astype(np.int32) # S=2 random strategies, fixed at creation
        self._run_strategies = lambda mu_idx: self._strategies[:,mu_idx] # Function to run the strategy on input x
        self._strategies_fitness = np.zeros(S).astype(np.int32) # Fitness of each strategy, updated after each turn (even for unplayed strategies --> virtual update)
        self._current_choices = None # will store outcomes for all strategies for the current turn
        self.action = None # will store the chosen action for the current turn

    def _choose_action(self, mu_idx):
        """Get the outcomes for all strategies from given input and return chosen action"""
        # Compute outputs for all strategies
        self._current_choices = self._run_strategies(mu_idx)  
        # Chosen action comes from best strategy
        best_strategies = np.where(self._strategies_fitness == np.max(self._strategies_fitness))[0]
        chosen_strategy = self.model.rng.choice(best_strategies) # in case of tie, choose randomly
        self.action = self._current_choices[chosen_strategy] # +1 or -1

    def _update_scores(self, outcome):
        """Update the agent's score and strategies' fitness values from current game outcome"""
        self._strategies_fitness += (self._current_choices == outcome).astype(np.int32)


class Producer(mesa.Agent):
    """Producer Minority Game agent."""

    def __init__(self, model: mesa.Model, P: int):
        super().__init__(model)
        self.S = 1
        self.wealth = 0 # Agent's wealth
        self._strategy = self.model.rng.choice([-1, 1], size=(P,)).astype(np.int32) # Random strategy, fixed at creation
        self._run_strategy = lambda mu_idx: self._strategy[mu_idx] # Function to run the strategy on input x
        self.action = None # will store the chosen action for the current turn

    def _choose_action(self, mu_idx):
        """Get the outcomes for all strategies from given input and return chosen action"""
        self.action = self._run_strategy(mu_idx) 

    def _update_scores(self, outcome, A):
        """Update the agent's score and strategies' fitness values from current game outcome"""
        payoff = - self.action * A
        self.wealth += payoff


class Speculator(mesa.Agent):
    """Speculator Minority Game agent."""

    def __init__(self, model: mesa.Model, P: int, eta: float, epsilon: float):
        super().__init__(model)
        self.eta = eta # Market impact (0=not taken into account, 1=fully taken into account)
        self.epsilon = epsilon # Threshold to participate in the game (0=participate if edge, 1=never participate)
        self.S = 2
        self.wealth = 0 # Agent's wealth
        self._strategies = self.model.rng.choice([-1,1], size=(self.S,P)).astype(np.int32) # S=2 random strategies, fixed at creation
        self._run_strategies = lambda mu_idx: self._strategies[:, mu_idx] # Function to run all strategies on input x
        self._strategies_fitness = np.zeros(self.S).astype(np.int32) # Fitness of each strategy, updated after each turn (even for unplayed strategies --> virtual update)
        self._do_nothing_fitness = 0 # Fitness of the do-nothing strategy
        self._current_choices = None # will store outcomes for all strategies for the current turn
        self.chosen_strategy = None # will store the chosen strategy for the current turn
        self.action = None # will store the chosen action for the current turn

    def _choose_action(self, mu_idx):
        """Get the outcomes for all strategies from given input and return chosen action"""
        # Compute outputs for all strategies
        self._current_choices = self._run_strategies(mu_idx)  
        # Chosen action comes from best strategy
        best_fitness = np.max(self._strategies_fitness)
        if best_fitness < self._do_nothing_fitness:
            self.chosen_strategy = None
            self.action = 0
        else:
            best_strategies = np.where(self._strategies_fitness == best_fitness)[0]
            self.chosen_strategy = self.model.rng.choice(best_strategies) # in case of tie, choose randomly
            self.action = self._current_choices[self.chosen_strategy] # +1 or -1

    def _update_scores(self, outcome, A):
        """Update the agent's score and strategies' fitness values from current game outcome"""
        payoff = - self.action * A
        self.wealth += payoff
        self._strategies_fitness += (self._current_choices == outcome).astype(np.int32)
        if self.chosen_strategy is not None: # if we played, the in-vivo strategy gets an extra reward for market impact
            self._strategies_fitness[self.chosen_strategy] += self.eta
        self._do_nothing_fitness += self.epsilon # each step, increase fitness of do-nothing strategy by epsilon
