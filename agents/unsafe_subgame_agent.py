import numpy as np
import pandas as pd
from itertools import combinations
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.action import Action as ToyPokerAction
from poker_env.ToyPoker.data.eval_potential import calc_final_potential
from poker_env.LimitTexas.state import PokerState
from poker_env.ToyPoker.env import ToyPokerEnv
from poker_env.agents.baseline_agent.Toypoker_baseline_agent import ToypokerTPAgent, ToypokerLPAgent, ToypokerLAAgent
from poker_env.agents.lbr_agent.opponent_modeling import Classifier



class ToyPokerLBRAgent(Agent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, agent_id, agent=None, mode='agent', player_num=2):
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
        super().__init__(agent_type='LBRAgent')
        self.mode = mode
        self.action_space = []
        self.player_num = player_num
        self.all_cards = None
        self.range = {}
        self.public = []
        self.set_player_id(agent_id)
        self.opp_id = (agent_id + 1) % self.player_num
        self.opponent = agent
        self.opponent_name = None
        self.env = ToyPokerEnv(num_players=player_num, allow_step_back=True)
        self.classifier = Classifier(self.opp_id)
        # read first round EHS
        self.first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_EHS_vector.csv', index_col=None)
        for action in ToyPokerAction:
            self.action_space.append(action.value)

    @property
    def action_num(self):
        return len(self.action_space)

    def encode_action(self, state):
        encode_actions = []
        for action in state.legal_actions:
            encode_actions.append(self.action_space.index(action))
        return encode_actions

    def decode_action(self, action):
        return self.action_space[action]

    def Is_start(self, state):
        '''
        Judge if it is the beginning of the game
        '''
        if (not state.previous_all_actions) or (len(state.previous_all_actions) == 1):
            return True
        return False

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
            self.opponent_name = 'random'
        else:
            # my step and opponent's step
            self.update_information(state)
        max_action_value, action_index = self.get_value(state)
        if max_action_value <= 0:
            action = ToyPokerAction.FOLD.value
        else:
            action = self.decode_action(action_index)
        return action

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
        wp = self.WpRollout(state)
        pot_lbr = state.pot[self.player_id]
        asked = sum(state.pot) - 2 * pot_lbr
        pot = asked + 2 * pot_lbr
        # Go through all the legal actions of current state
        legal_actions = self.encode_action(state)
        action_utilities = np.zeros(self.action_num)
        for a in legal_actions:
            fp = 0
            str_action = self.decode_action(a)

            for cards in self.all_cards:
                self.env.reset_hand(cards, self.opp_id)
                oppo_state, _ = self.env.step(str_action)
                action_prob = self.get_opp_action_prob(oppo_state)
                cards_str = ''.join(PokerState.get_suit_normalization(cards, []))
                fp += self.range[cards_str] * action_prob[0]
                self.env.step_back()
                self.env.give_back_hand(self.opp_id)
            if (str_action == ToyPokerAction.CALL.value) or (str_action == ToyPokerAction.CHECK.value):
                action_utilities[a] = wp * pot - (1 - wp) * asked
            elif str_action == ToyPokerAction.FOLD.value:
                continue
            else:
                raise_size = state.raise_money
                action_utilities[a] = fp * pot + (1 - fp) * (wp * (pot + raise_size) - (1 - wp) * (asked + raise_size))

        max_value_action = np.argmax(action_utilities)
        return action_utilities[max_value_action], max_value_action

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

    def init_information(self, state):
        '''
        Initialize private environment and update information to the private information
        '''
        hand = [[] for _ in range(state.num_player)]
        hand[self.player_id] = state.hand_cards
        self.public = state.public_cards
        state, player_id = self.env.init_game(button=state.button, hand_cards=hand, public_cards=self.public)
        self.init_range()
        # if opponent makes an action first
        if state.previous_all_actions:
            self.update_information(state, n=1)

    def init_range(self):
        '''
        Initialize opponent's cards range according to the cards have been dealt
        '''
        deck = self.env.get_deck()

        # get all possible cards of opponent and init range
        self.all_cards = []
        self.range = {}

        for com in combinations(deck, 2):
            hands = [card.get_index() for card in com]
            self.all_cards.append(hands)
            cards_str = ''.join(PokerState.get_suit_normalization(hands, []))
            self.range[cards_str] = float(1 / 231)

    def update_information(self, state, n=2):
        '''
        Update the situation of eval_env to private environment
        Args:
            state: (PokerState)
            n(int 1 or 2): since the private environment has to be late for two steps
                or one steps at the beginning if opponent make action first.
        '''
        actions = state.latest_action(n=n)
        # if there are new public cards
        if not len(self.public) == len(state.public_cards):
            for i in range(n):
                # since I don't know which step give the new_public_cards
                self.env.step(actions[i], new_public_cards=state.public_cards[-2:])
            self.update_range(state, new_public_cards=state.public_cards[-2:])
            self.public = state.public_cards
        else:
            for i in range(n):
                self.env.step(actions[1])
            self.update_range(state)

    def update_range(self, state, new_public_cards=None):
        '''
        Update the cards range of opponent according to the change of public state
        and opponent's strategy and latest action
        '''
        action = state.latest_action()[0]
        new_range = {}
        # There are new public cards
        if new_public_cards:
            self.all_cards = []
            deck = self.env.get_deck()
            for com in combinations(deck, 2):
                hands = [card.get_index() for card in com]
                self.all_cards.append(hands)
                cards_str = ''.join(PokerState.get_suit_normalization(hands, []))
                new_range[cards_str] = self.range[cards_str]
            # normalization
            for cards in self.all_cards:
                cards_str = ''.join(PokerState.get_suit_normalization(cards, []))
                new_range[cards_str] = float(new_range[cards_str]) / sum(new_range.values())
        else:
            new_range = self.range

        # Update opponent's range according to opponent's strategy and latest action
        previous_state, player_id = self.env.step_back()
        for cards in self.all_cards:
            self.env.reset_hand(cards, player_id)
            action_prob = self.get_opp_action_prob(previous_state)  # get action prob
            cards_str = ''.join(PokerState.get_suit_normalization(cards, []))
            new_range[cards_str] *= action_prob[self.action_space.index(action)]
            self.env.give_back_hand(player_id)
        if new_public_cards:
            self.env.step(action, new_public_cards)
        else:
            self.env.step(action)
        # normalization
        for cards in self.all_cards:
            cards_str = ''.join(PokerState.get_suit_normalization(cards, []))
            new_range[cards_str] = float(new_range[cards_str]) / sum(new_range.values())

        self.range = new_range
        return

    def get_opp_action_prob(self, state):
        '''
        Agent mode: Get opponent action probability according to opponent modeling
        Test mode: directly use opponent strategy
        '''
        # TODO now only for agent mode
        if self.mode == 'agent':

            action_length = self.action_num
            action_prob = np.zeros(range(action_length))
            if self.opponent_name == 'random':
                action_prob = np.array([1.0 / action_length for _ in range(action_length)])
            else:
                action = self.opponent.eval_step(state)
                action_prob[self.action_space.index(action)] = 1
            return action_prob

    def opponent_modeling(self, trajectories_list):
        '''
        Find out opponent's type
        Args: (list) the trajectories of several episode
        '''
        name = self.classifier.get_agent_type([trajectories_list])
        if name == 'TP':
            self.opponent = ToypokerTPAgent()
        elif name == 'LP':
            self.opponent = ToypokerLPAgent()
        elif name == 'LA':
            self.opponent = ToypokerLAAgent()
        elif name == 'INIT' or name == 'TA':
            self.opponent_name = 'random'
        else:
            raise Exception("TypeError")

