import numpy as np
from copy import deepcopy
import time
from itertools import combinations
from multiprocessing import Pool
from poker_env.agents.lbr_agent.Toypoker_basebr_agent import ToyPokerBaseBRAgent
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.action import Action as ToyPokerAction
from poker_env.ToyPoker.state import PokerState


class ToyPokerBRAgent(ToyPokerBaseBRAgent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, agent_id, mode='agent', agent=None, agent_name='random', player_num=2):
        '''
        Initialize the agent
        #TODO this LBR is just for two players

        Args:
            agent(Agent): the agent should be tested, if None then this is a independent agent
            mode(str): mode of LBR
                    'agent' means LBR perform like a agent without knowing opponent's strategy
                    'test' means LBR is a test tool to get exploitability

        Init:
            range (dict): opponent's range with, key(int): cards_couple_index, value(float): probability
            all_cards (set): all possible opponent's cards
        '''
        super().__init__(mode, agent_id, agent=agent, player_num=player_num, agent_name=agent_name, agent_type='BRAgent')

    def eval_step(self, state):
        '''
        Given a state, predict the best action according to LBR formula

        Args:
            state (PokerState): the current state

        Returns:
            action (str): local best response action
        '''
        start = time.time()
        if self.Is_start(state):
            self.init_information(state)
        else:
            # opponent's step and my step
            self.update_information(state)
        private_state = self.env.get_state()
        action_utilities = self.parallel_traverse(private_state)
        max_value_action = np.argmax(action_utilities)
        if action_utilities[max_value_action] <= 0:
            action = ToyPokerAction.FOLD.value
        else:
            action = self.decode_action(max_value_action)
        end = time.time()
        print(end - start)
        return action

    def parallel_traverse(self, state):
        legal_actions = self.encode_action(state)
        action_utilities = np.zeros(self.action_num)
        parameters = []
        for a in legal_actions:
            str_action = self.decode_action(a)
            for cards in self.all_cards:
                parameters.append([a, str_action, cards, deepcopy(self.env), deepcopy(state)])
        print(len(parameters))
        process = Pool()
        result = process.map(self.traverse, parameters)
        process.close()
        process.join()
        for action_utility in result:
            action_utilities += action_utility
        return action_utilities

    def traverse(self, parameter):
        a, str_action, cards, env, state = parameter
        action_utilities = np.zeros(self.action_num)
        env.reset_cards(hand=cards, player_id=self.opp_id)
        previous_public = deepcopy(state.public_cards)
        next_state, _ = env.step(str_action)
        cards_str = PokerState.sort_hand_cards(cards)
        new_public_num = len(next_state.public_cards) - len(previous_public)
        if (new_public_num > 0) and (not env.is_over()):
            env.give_back_cards(public_num=new_public_num)
            deck = env.get_deck()
            for com in combinations(deck, 2):
                next_state = env.reset_cards([card.get_index() for card in com])
                state_value = self.get_value(next_state, env)
                action_utilities[a] += self.range[cards_str] * state_value / len(list(combinations(deck, 2)))
                env.give_back_cards(public_num=new_public_num)
        else:
            state_value = self.get_value(next_state, env)
            action_utilities[a] += self.range[cards_str] * state_value
        state, _ = env.step_back()
        env.give_back_cards(player_id=self.opp_id)
        return action_utilities

    def get_value(self, state, env):
        '''
        Go through all the legal actions of LBR agent at any decision point to
        get the action_utilities

        Args:
            state (PokerState): the current state

        Returns:
            (float): maximal action_utilities
            (int): the action with greatest value
        '''
        player_id = state.player_id
        if env.is_over():
            return env.get_payoffs()[self.player_id]
        else:
            legal_actions = self.encode_action(state)
            # if it isn't our decision point
            expected_value = 0
            action_utilities = np.zeros(self.action_num)
            for a in legal_actions:
                str_action = self.decode_action(a)
                if player_id != self.player_id:
                    action_prob = self.get_opp_action_prob(state)
                else:
                    action_prob = np.zeros(self.action_num)
                previous_public = deepcopy(state.public_cards)
                next_state, _ = env.step(str_action)
                new_public_num = len(next_state.public_cards) - len(previous_public)
                if (new_public_num > 0) and (not env.is_over()):
                    env.give_back_cards(public_num=new_public_num)
                    deck = env.get_deck()
                    for com in combinations(deck, 2):
                        next_state = env.reset_cards([card.get_index() for card in com])
                        state_value = self.get_value(next_state, env)
                        if player_id != self.player_id:
                            expected_value += action_prob[a] * state_value / len(list(combinations(deck, 2)))
                        else:
                            action_utilities[a] += state_value / len(list(combinations(deck, 2)))
                        env.give_back_cards(public_num=new_public_num)
                else:
                    state_value = self.get_value(next_state, env)
                    if player_id != self.player_id:
                        expected_value += action_prob[a] * state_value
                    else:
                        action_utilities[a] = state_value
                state, _ = env.step_back()
            if player_id != self.player_id:
                return expected_value
            else:
                return max(action_utilities)


class ToyPokerMCBRAgent(ToyPokerBaseBRAgent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, agent_id, mode='agent', agent=None, agent_name='random', player_num=2, training_times=10):
        '''
        Initialize the agent
        #TODO this LBR is just for two players

        Args:
            agent(Agent): the agent should be tested, if None then this is a independent agent
            mode(str): mode of LBR
                    'agent' means LBR perform like a agent without knowing opponent's strategy
                    'test' means LBR is a test tool to get exploitability

        Init:
            range (dict): opponent's range with, key(int): cards_couple_index, value(float): probability
            all_cards (set): all possible opponent's cards
        '''
        super().__init__(mode, agent_id, agent=agent, player_num=player_num, agent_name=agent_name, agent_type='MCBRAgent')
        self.training_times = training_times

    def eval_step(self, state):
        '''
        Given a state, predict the best action according to LBR formula

        Args:
            state (PokerState): the current state

        Returns:
            action (str): local best response action
        '''
        if self.Is_start(state):
            self.init_information(state)
        else:
            # opponent's step and my step
            self.update_information(state)
        action_utilities = np.zeros(self.action_num)
        for _ in range(self.training_times):
            action_utilities += self.traverse(state) / self.training_times
        max_value_action = np.argmax(action_utilities)
        if action_utilities[max_value_action] <= 0:
            action = ToyPokerAction.FOLD.value
        else:
            action = self.decode_action(max_value_action)
        return action

    def traverse(self, state):
        action_utilities = np.zeros(self.action_num)
        legal_actions = self.encode_action(state)
        for a in legal_actions:
            str_action = self.decode_action(a)
            for cards in self.all_cards:
                self.env.reset_cards(hand=cards, player_id=self.opp_id)
                next_state, _ = self.env.step(str_action)
                cards_str = PokerState.sort_hand_cards(cards)
                state_value = self.get_value(next_state)
                action_utilities[a] += self.range[cards_str] * state_value
                self.env.step_back()
                self.env.give_back_cards(player_id=self.opp_id)
        return action_utilities

    def get_value(self, state):
        '''
        Go through all the legal actions of LBR agent at any decision point to
        get the action_utilities

        Args:
            state (PokerState): the current state

        Returns:
            (float): maximal action_utilities
            (int): the action with greatest value
        '''
        if self.env.is_over():
            return self.env.get_payoffs()[self.player_id]
        else:
            legal_actions = self.encode_action(state)
            # if it isn't our decision point
            expected_value = 0
            action_utilities = np.zeros(self.action_num)
            for a in legal_actions:
                str_action = self.decode_action(a)
                if state.player_id != self.player_id:
                    action_prob = self.get_opp_action_prob(state)
                    next_state, _ = self.env.step(str_action)
                    state_value = self.get_value(next_state)
                    expected_value += action_prob[a] * state_value
                else:
                    next_state, _ = self.env.step(str_action)
                    action_utilities[a] = self.get_value(next_state)
                state, _ = self.env.step_back()
            if state.player_id != self.player_id:
                return expected_value
            else:
                return max(action_utilities)

