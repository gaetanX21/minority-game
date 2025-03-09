# inspired from https://gist.github.com/RuizSerra/80d4175a33f3ba01ece7a98335786981

import mesa
import mesa.model
import numpy as np

# Convert a binary array to its decimal equivalent (e.g. [0, 1, 1] -> 3)
bin2dec = lambda x: np.sum(np.array([2**i for i in reversed(range(x.shape[-1]))]) * x, axis=-1)

class Agent(mesa.Agent):
    """Minority Game agent"""

    def __init__(self, model: mesa.Model, M: int, S: int):
        super().__init__(model)
        self.M = M # Memory size
        self._strategies = self.model.rng.choice([-1, 1], size=(S, 2**M)).astype(np.int32) # Random strategies, fixed at creation
        self._run_strategies = lambda x: self._strategies[:, bin2dec(x)] # Function to run all strategies on input x
        self._strategies_fitness = np.zeros(S).astype(np.int32) # Fitness of each strategy, updated after each turn (even for unplayed strategies --> virtual update)
        self._current_choices = None # will store outcomes for all strategies for the current turn
        self.chosen_strategy = None # will store the chosen strategy for the current turn
        self.action = None # will store the chosen action for the current turn

    def _choose_action(self):
        """Get the outcomes for all strategies from given input and return chosen action"""
        x = self.model.get_history(self.M)
        # Compute outputs for all strategies
        self._current_choices = self._run_strategies(x)  
        # Chosen action comes from best strategy
        best_strategies = np.where(self._strategies_fitness == np.max(self._strategies_fitness))[0]
        self.chosen_strategy = self.model.rng.choice(best_strategies)
        self.action = self._current_choices[self.chosen_strategy]

    def _update_scores(self, outcome):
        """Update the agent strategies' fitness values from current game outcome"""
        self._strategies_fitness += (self._current_choices == outcome).astype(np.int32)


class MinorityGameModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N: int, M: int=5, S: int=4, seed: int=None):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Simulation configuration
        self.running = True

        # Agent configuration
        self.N = N
        if isinstance(M, int):
            M = np.ones(N).astype(np.int32) * M
        if isinstance(S, int):
            S = np.ones(N).astype(np.int32) * S

        # Create agents
        for i in range(N):
            _ = Agent(self, M[i], S[i])

        # Create synthetic history to begin with
        self.history = self.rng.choice([0, 1], size=M.max())
        self.attendance = 0

        # Data collection
        self.datacollector = mesa.DataCollector(model_reporters={"Attendance": "attendance"})

    def step(self):
        self.agents.do("_choose_action")
        # Compute and store attendance and minority side (outcome)
        self.attendance = self.compute_attendance()
        outcome = -np.sign(self.attendance)
        self.agents.do("_update_scores", outcome)
        bit = int(outcome>0) # 1 if minority side is 1, 0 otherwise
        self.history = np.append(self.history, bit)
        self.datacollector.collect(self)

    def get_history(self, M):
        return self.history[-M:]

    def compute_attendance(self):
        """Compute the attendance for side A in the current tick"""
        A = 0
        for agent in self.agents:
            A += agent.action
        # A += self.N/2
        return A
    
    def run_model(self, T):
        for t in range(T):
            self.step()