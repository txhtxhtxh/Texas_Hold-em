import numpy as np
import time
from poker_env.agents.lbr_agent.Toypoker_basebr_agent import ToyPokerBaseBRAgent
from multiprocessing import Pool
from copy import deepcopy
from itertools import combinations

from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.state import PokerState
from poker_env.ToyPoker.action import Action as ToyPokerAction
from poker_env.ToyPoker.data.eval_potential import calc_final_potential



class ToyPokerLBRAgent(ToyPokerBaseBRAgent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, mode, agent_id, agent=None, player_num=2, training_times=32, agent_name='random'):
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
        super().__init__(mode, agent_id, agent=agent, player_num=player_num, agent_name=agent_name, agent_type='LBRAgent')
        self.training_times = training_times

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
            # my step and opponent's step
            self.update_information(state)
        action_utilities = self.get_value(deepcopy(state), deepcopy(self.env))
        if max(action_utilities) <= 0:
            action = ToyPokerAction.FOLD.value
        else:
            action = self.decode_action(np.argmax(action_utilities))
        end = time.time()
        print(end - start)
        return action

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
        wp = self.WpRollout(state)
        pot_lbr = state.pot[self.player_id]
        asked = sum(state.pot) - 2 * pot_lbr
        pot = asked + 2 * pot_lbr
        # Go through all the legal actions of current state
        legal_actions = self.encode_action(state)
        action_utilities = np.zeros(self.action_num)
        for a in legal_actions:
            fp, is_end = 0, False
            str_action = self.decode_action(a)
            for cards in self.all_cards:
                env.reset_cards(player_id=self.opp_id, hand=cards)
                cards_str = PokerState.sort_hand_cards(cards)
                oppo_state, _ = env.step(str_action)
                if env.is_over():
                    action_utilities[a] += self.range[cards_str] * (env.get_payoffs()[self.player_id] + pot_lbr)
                    is_end = True
                else:
                    action_prob = self.get_opp_action_prob(oppo_state)
                    fp += self.range[cards_str] * action_prob[0]
                env.step_back()
                env.give_back_cards(player_id=self.opp_id)
            if not is_end:
                if (str_action == ToyPokerAction.CALL.value) or (str_action == ToyPokerAction.CHECK.value):
                    action_utilities[a] = wp * pot - (1 - wp) * asked
                elif str_action == ToyPokerAction.FOLD.value:
                    continue
                else:
                    raise_size = state.raise_money
                    action_utilities[a] = fp * pot + (1 - fp) * (wp * (pot + raise_size) - (1 - wp) * (asked + raise_size))
        return action_utilities

    def WpRollout(self, state):
        '''
        Get LBR wining probability
        '''
        lossless_state = state.get_infoset()
        # Search in precomputed table for the first round
        if len(state.public_cards) == 3:
            found_row = self.first_round_table[self.first_round_table['cards_str'] == lossless_state]
            ehs_value = np.mean(found_row.iloc[:, 2:].values)
        # Calculate ehs for the final round
        elif len(state.public_cards) == 5:
            ehs_value = calc_final_potential(state.hand_cards, state.public_cards)
        else:
            raise Exception("Public_Cards_Error")
        return ehs_value


class ToyPokerDLBRAgent(ToyPokerBaseBRAgent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, agent_id, max_depth=6, mode='agent', agent=None, agent_name='random', player_num=2, training_times=1):
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
        super().__init__(mode, agent_id, agent=agent, player_num=player_num, agent_name=agent_name, agent_type='DLBRAgent')
        self.max_depth = max_depth
        self.training_times = training_times

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
        action_utilities = np.zeros(self.action_num)
        for _ in range(self.training_times):
            action_utilities += self.parallel_traverse(state) / self.training_times
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
                self.env.shuffle_deck()
        print(len(parameters))
        process = Pool()
        result = process.map(self.traverse, parameters)
        process.close()
        process.join()
        for action_utility in result:
            action_utilities += action_utility
        return action_utilities

    def traverse(self, parameters):
        a, str_action, cards, env, state = parameters
        action_utilities = np.zeros(self.action_num)
        self.env.reset_cards(player_id=self.opp_id, hand=cards)
        next_state, _ = self.env.step(str_action)
        cards_str = PokerState.sort_hand_cards(cards)
        state_value = self.get_value(next_state, 1)
        action_utilities[a] += self.range[cards_str] * state_value
        self.env.step_back()
        self.env.give_back_cards(player_id=self.opp_id)
        return action_utilities

    def get_value(self, state, depth):
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
        elif (depth >= self.max_depth) and (state.player_id == self.player_id):
            return self.get_reward(state)
        else:
            legal_actions = self.encode_action(state)
            # if it isn't our decision point
            if state.player_id != self.player_id:
                expected_value = 0
                for a in legal_actions:
                    str_action = self.decode_action(a)
                    action_prob = self.get_opp_action_prob(state)
                    next_state, _ = self.env.step(str_action)
                    state_value = self.get_value(next_state, depth + 1)
                    self.env.step_back()
                    expected_value += action_prob[a] * state_value
                return expected_value
            # if it is our decision point
            else:
                action_utilities = np.zeros(self.action_num)
                for a in legal_actions:
                    str_action = self.decode_action(a)
                    next_state, _ = self.env.step(str_action)
                    state_value = self.get_value(next_state, depth + 1)
                    self.env.step_back()
                    action_utilities[a] = state_value
                return max(action_utilities)

    def get_reward(self, state):
        '''
        Go through all the legal actions of LBR agent at any decision point to
        get the action_utilities

        Args:
            state (PokerState): the current state

        Returns:
            (float): maximal action_utilities
            (int): the action with greatest value
        '''
        wp = self.WpRollout(state)
        pot_lbr = state.pot[self.player_id]
        asked = sum(state.pot) - 2 * pot_lbr
        pot = asked + 2 * pot_lbr
        # Go through all the legal actions of current state
        legal_actions = self.encode_action(state)
        action_utilities = np.zeros(self.action_num)
        for a in legal_actions:
            fp, is_end = 0, False
            str_action = self.decode_action(a)
            cards = self.env.get_all_player_handcards()[self.opp_id]
            cards_str = PokerState.sort_hand_cards(cards)
            oppo_state, _ = self.env.step(str_action)
            if self.env.is_over():
                action_utilities[a] = self.env.get_payoffs()[self.player_id] + pot_lbr
                is_end = True
            else:
                action_prob = self.get_opp_action_prob(oppo_state)
                fp += self.range[cards_str] * action_prob[0]
            self.env.step_back()
            if not is_end:
                if (str_action == ToyPokerAction.CALL.value) or (str_action == ToyPokerAction.CHECK.value):
                    action_utilities[a] = wp * pot - (1 - wp) * asked
                elif str_action == ToyPokerAction.FOLD.value:
                    continue
                else:
                    raise_size = state.raise_money
                    action_utilities[a] = fp * pot + (1 - fp) * (
                                wp * (pot + raise_size) - (1 - wp) * (asked + raise_size))
        return max(action_utilities)

    def WpRollout(self, state):
        '''
        Get LBR wining probability
        '''
        lossless_state = state.get_infoset()
        # Search in precomputed table for the first round
        if len(state.public_cards) == 3:
            found_row = self.first_round_table[self.first_round_table['cards_str'] == lossless_state]
            ehs_value = np.mean(found_row.iloc[:, 2:].values)
        # Calculate ehs for the final round
        elif len(state.public_cards) == 5:
            ehs_value = calc_final_potential(state.hand_cards, state.public_cards)
        else:
            raise Exception("Public_Cards_Error")
        return ehs_value
