"""
Module containing the models for the Minority Game.
The models are implemented using the Mesa framework.
The models are:
- Classic Minority Game
- Markovian Minority Game
- Grand Canonical Minority Game
"""

import mesa
import numpy as np
import agents
from tqdm.notebook import tqdm


class ClassicMinorityGame(mesa.Model):
    """Classic Minority Game model."""

    def __init__(self, N: int, M: int, S: int=2, seed: int=None):
        # fix seed for reproducibility
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
            _ = agents.BasicAgent(self, M[i], S[i])

        # Create synthetic history to begin with
        self.history = self.rng.choice([0, 1], size=M.max())
        self.attendance = 0

        # Data collection
        self.datacollector = mesa.DataCollector(model_reporters={"Attendance": "attendance"})

    def step(self):
        # Choose action for all agents
        self.agents.do("_choose_action")
        # Compute and store attendance and minority side (outcome)
        self.attendance = self.compute_attendance()
        outcome = -np.sign(self.attendance) # -1 if attendance<0, 1 if attendance>0 (0 is not possible since N is odd)
        # Update scores for all agents
        self.agents.do("_update_scores", outcome)
        # Update history
        bit = int(outcome>0) # history is encoded as a single bit 0/1 (more convenient than -1/1)
        self.history = np.append(self.history, bit) 
        # Collect data
        self.datacollector.collect(self)

    def get_history(self, M):
        return self.history[-M:]

    def compute_attendance(self):
        """Compute the attendance for side A in the current tick."""
        return np.sum([agent.action for agent in self.agents])
    
    def run_model(self, T):
        for _ in tqdm(range(T)):
            self.step()


class MarkovianMinorityGame(mesa.Model):
    """
    Markovian Minority Game model.
    Fixed S=2 (lower complexity for same qualitative results).
    """

    def __init__(self, N: int, P: int, seed: int=None):
        # fix seed for reproducibility
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Simulation configuration
        self.running = True

        # Agent configuration
        self.N = N

        # Create agents
        for i in range(N):
            _ = agents.MarkovianAgent(self, P)

        # Create synthetic history to begin with
        self.P = P
        self.attendance = 0

        # Data collection
        self.datacollector = mesa.DataCollector(model_reporters={"Attendance": "attendance", "mu_idx": "mu_idx"})

    def step(self):
        # Choose action for all agents
        self.mu_idx = self.rng.choice(self.P)
        self.agents.do("_choose_action", self.mu_idx)
        # Compute and store attendance and minority side (outcome)
        self.attendance = self.compute_attendance()
        outcome = -np.sign(self.attendance) # -1 if attendance<0, 1 if attendance>0 (0 is not possible since N is odd)
        # Update scores for all agents
        self.agents.do("_update_scores", outcome)
        # Collect data
        self.datacollector.collect(self)

    def compute_attendance(self):
        """Compute the attendance for side A in the current tick."""
        return np.sum([agent.action for agent in self.agents])
    
    def run_model(self, T):
        for _ in tqdm(range(T)):
            self.step()


class GrandCanonicalMinorityGame(mesa.Model):
    """Grand Canonical Minority Game model."""

    def __init__(self, N_p: int, N_s: int, P: int, eta: float, epsilon: float, S: int=2, seed: int=None):
        # fix seed for reproducibility
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Simulation configuration
        self.running = True

        # Agent configuration
        self.N_s = N_s
        self.N_p = N_p
        # Create agents
        for i in range(N_p):
            _ = agents.Producer(self, P)
        for i in range(N_s):
            _ = agents.Speculator(self, P, eta, epsilon)

        self.P = P
        self.attendance = 0 

        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={"Attendance": "attendance"},
            agent_reporters={"Action": "action"}    
        )

    def step(self):
        # Choose action for all agents
        mu_idx = self.rng.choice(self.P)
        self.agents.do("_choose_action", mu_idx)
        # Compute and store attendance and minority side (outcome)
        self.attendance = self.compute_attendance()
        outcome = -np.sign(self.attendance) # -1 if attendance<0, 1 if attendance>0 (0 is not possible since N is odd)
        # Update scores for all agents
        self.agents.do("_update_scores", outcome, self.attendance)
        # Collect data
        self.datacollector.collect(self)

    def compute_attendance(self):
        """Compute the attendance for side A in the current tick."""
        return np.sum([agent.action for agent in self.agents])
    
    def run_model(self, T):
        for _ in tqdm(range(T)):
            self.step()

