import numpy as np
import time
import collections
from multiprocessing import Pool
from copy import deepcopy
from poker_env.agents.lbr_agent.Toypoker_basebr_agent import ToyPokerBaseBRAgent
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.state import PokerState
from poker_env.ToyPoker.action import Action as ToyPokerAction
from poker_env.ToyPoker.data.eval_potential import calc_final_potential

class ToyPokerCFRPlusAgent(ToyPokerBaseBRAgent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, agent_id, max_depth=6, search_mode='normal', mode='agent', agent=None, agent_name='random', player_num=2, training_times=1, infoset_mode='all_action'):
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
        self.policy = collections.defaultdict(np.array)
        self.regrets = collections.defaultdict(np.array)
        self.max_depth = max_depth
        self.training_times = training_times
        self.infoset_mode = infoset_mode
        self.search_mode = search_mode
        self.iterations = 0
        super().__init__(mode, agent_id, agent=agent, player_num=player_num, agent_name=agent_name, agent_type='CFRPlusAgent')

    def encode_state(self, state):
        '''
        Get abstracted infoset of state.

        Args:
            state (PokerState): the state of the game

        Returns:
            (string): infoset keys.
        '''
        lossless_state = state.get_infoset()
        if self.infoset_mode == 'no_action':
            return lossless_state
        elif self.infoset_mode == 'own_action':
            return lossless_state + '/' + state.previous_own_actions
        elif self.infoset_mode == 'all_action':
            return lossless_state + '/' + state.previous_all_actions
        else:
            raise ValueError("Not valid infoset_mode!")

    def calculate_strategy(self, info_set, legal_actions, regrets=None):
        '''
        Calculates the strategy based on regrets. Set zero if this infoset hasn't been initialized in memory,

        Args:
            info_set (str): key in policy dictionary which represent the information of state
            legal_actions (list): indices of legal actions

        Returns:
            (np.ndarray): the action probabilities
        '''
        sum_regret = 0
        # calculate sum positive-regrets of current infoset
        if info_set not in self.regrets:
            self.regrets[info_set] = np.zeros(self.action_num)
        else:
            regrets = self.regrets[info_set]
            for action in legal_actions:
                sum_regret += max(regrets[action], 0)

        # calculate strategy based on regrets
        action_probs = np.zeros(self.action_num)
        for action in legal_actions:
            if sum_regret > 0:
                action_probs[action] = max(regrets[action], 0) / sum_regret
            else:
                action_probs[action] = 1 / len(legal_actions)

        return action_probs

    def update_policy(self, regrets=None):
        '''
        Update policy based on the current regrets
        '''
        if regrets:
            for info_set in regrets:
                self.regrets[info_set] += regrets[info_set] / 2
        for info_set, regret in self.regrets.items():
            positive_regret_sum = np.sum(np.maximum(regret, 0))
            if positive_regret_sum > 0:
                action_probs = np.maximum(regret, 0) / positive_regret_sum
            else:
                action_probs = np.ones(regret.shape) / len(regret)
            self.policy[info_set] = action_probs


    def get_action_probs(self, state):
        '''
        Get the action probabilities of the current state.

        Args:
            state (PokerState): the state of the game

        Returns:
            (numpy.ndarray): the action probabilities
        '''
        action_length = self.action_num
        info_set = self.encode_state(state)
        legal_actions = self.encode_action(state)
        if info_set not in self.policy:
            action_probs = np.array([1.0 / action_length for _ in range(action_length)])
            self.policy[info_set] = action_probs
        else:
            action_probs = self.policy[info_set]
        # Remove illegal actions
        legal_probs = np.zeros(action_length)
        legal_probs[legal_actions] = action_probs[legal_actions]
        # Normalization
        if np.sum(legal_probs) == 0:
            legal_probs[legal_actions] = 1 / len(legal_actions)
        else:
            legal_probs /= np.sum(legal_probs)
        return legal_probs

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
        for _ in range(self.training_times):
            self.iterations += 1
            self.parallel_traverse(state)
            discount = self.iterations / (self.iterations + 1)
            for info_set in self.regrets.keys():
                self.regrets[info_set] *= discount
        action_prob = self.get_action_probs(state)
        action = np.random.choice(self.action_space, p=action_prob)
        end = time.time()
        print(end - start)
        return action

    def parallel_traverse(self, state):
        parameters = []
        for cards in self.all_cards:
            reset_state = self.env.reset_cards(player_id=self.opp_id, hand=cards)
            value = 0
            parameters.append([cards, deepcopy(self.env), deepcopy(state), value, deepcopy(self.regrets)])
            self.env.shuffle_deck()
            self.env.give_back_cards(player_id=self.opp_id)
        print(len(parameters))
        process = Pool()
        result = process.map(self.traverse, parameters)
        process.close()
        process.join()
        for regrets in result:
            self.update_policy(regrets)
        return

    def traverse(self, parameters):
        cards, env, state, value, regrets = parameters
        legal_actions = self.encode_action(state)
        action_utilities = np.zeros(self.action_num)
        info_set = self.encode_state(state)
        value_expectation = 0
        action_probs = self.calculate_strategy(info_set, legal_actions)
        for action in legal_actions:
            str_action = self.decode_action(action)
            next_state, _ = env.step(str_action)
            cards_str = PokerState.sort_hand_cards(cards)
            if self.search_mode == 'normal':
                state_value = self.get_value(next_state, env, value)
            else:
                state_value = self.get_dl_value(next_state, env, 1)
            action_utilities[action] += self.range[cards_str] * state_value
            value_expectation += action_utilities[action] * action_probs[action]
            state, _ = env.step_back()
        for action in legal_actions:
            regrets[info_set][action] += action_utilities[action] - value_expectation
        return regrets

    def get_value(self, state, env, value):
        '''
        Go through all the legal actions of LBR agent at any decision point to
        get the action_utilities

        Args:
            state (PokerState): the current state

        Returns:
            (float): maximal action_utilities
            (int): the action with greatest value
        '''
        if env.is_over():
            return env.get_payoffs()[self.player_id] - value
        else:
            legal_actions = self.encode_action(state)
            # if it isn't our decision point
            if state.player_id != self.player_id:
                expected_value = 0
                info_set = self.encode_state(state)
                action_prob = self.calculate_strategy(info_set, legal_actions)
                for action in legal_actions:
                    str_action = self.decode_action(action)
                    next_state, _ = env.step(str_action)
                    state_value = self.get_value(next_state, env, value)
                    expected_value += action_prob[action] * state_value
                    env.step_back()
            # if it is our decision point
            else:
                action_utilities = np.zeros(self.action_num)
                expected_value = 0
                info_set = self.encode_state(state)
                action_prob = self.calculate_strategy(info_set, legal_actions)
                for action in legal_actions:
                    str_action = self.decode_action(action)
                    next_state, _ = env.step(str_action)
                    action_utilities[action] = self.get_value(next_state, env, value)
                    expected_value += action_prob[action] + action_utilities[action]
                    env.step_back()
                for action in legal_actions:
                    self.regrets[info_set][action] += action_utilities[action] - expected_value
            return expected_value

    def get_dl_value(self, state, env, depth):
        '''
        Go through all the legal actions of LBR agent at any decision point to
        get the action_utilities

        Args:
            state (PokerState): the current state

        Returns:
            (float): maximal action_utilities
            (int): the action with greatest value
        '''
        if env.is_over():
            return env.get_payoffs()[self.player_id]
        elif (depth >= self.max_depth) and (state.player_id == self.player_id):
            return self.get_reward(state, env)
        else:
            legal_actions = self.encode_action(state)
            # if it isn't our decision point
            if state.player_id != self.player_id:
                expected_value = 0
                action_prob = self.get_opp_action_prob(state)
                for action in legal_actions:
                    str_action = self.decode_action(action)
                    next_state, _ = env.step(str_action)
                    state_value = self.get_value(next_state, env, depth + 1)
                    expected_value += action_prob[action] * state_value
                    env.step_back()

            # if it is our decision point
            else:
                action_utilities = np.zeros(self.action_num)
                expected_value = 0
                info_set = self.encode_state(state)
                action_prob = self.calculate_strategy(info_set, legal_actions)
                for action in legal_actions:
                    str_action = self.decode_action(action)
                    next_state, _ = env.step(str_action)
                    action_utilities[action] = self.get_value(next_state, env, depth + 1)
                    expected_value += action_prob[action] + action_utilities[action]
                    env.step_back()
                for action in legal_actions:
                    self.regrets[info_set][action] += action_utilities[action] - expected_value
            return expected_value

    def get_reward(self, state, env):
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
            cards = env.get_all_player_handcards()[self.opp_id]
            cards_str = PokerState.sort_hand_cards(cards)
            oppo_state, _ = env.step(str_action)
            if env.is_over():
                action_utilities[a] = env.get_payoffs()[self.player_id] + pot_lbr
                is_end = True
            else:
                action_prob = self.get_opp_action_prob(oppo_state)
                fp += self.range[cards_str] * action_prob[0]
            env.step_back()
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
