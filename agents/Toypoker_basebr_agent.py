import numpy as np
from copy import deepcopy
import pandas as pd
from itertools import combinations
from poker_env.agents.base_agent import Agent
from poker_env.ToyPoker.action import Action as ToyPokerAction
from poker_env.ToyPoker.data.eval_potential import calc_final_potential
from poker_env.ToyPoker.state import PokerState
from poker_env.ToyPoker.env import ToyPokerEnv
from poker_env.agents.baseline_agent.Toypoker_baseline_agent import ToypokerTPAgent, ToypokerLPAgent, ToypokerLAAgent
from poker_env.agents.lbr_agent.opponent_modeling import Classifier


class ToyPokerBaseBRAgent(Agent):
    '''
    A Implementation of a simple version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, mode, agent_id, agent=None, player_num=2, agent_name='random', agent_type='BaseAgent'):
        '''
        Initialize the agent

        Args:
            agent(Agent): the agent should be tested, if None then this is a independent agent
            mode(str): mode of LBR
                    'agent' means LBR perform like a agent without knowing opponent's strategy
                    'test' means LBR is a test tool to get exploitability

        Init:
            range (dict): opponent's range with, key(int): cards_couple_index, value(float): probability
            all_cards (set): all possible opponent's cards
        '''
        self.mode = mode
        self._type = agent_type
        self.action_space = []
        self.player_num = player_num
        self.all_cards = None
        self.range = {}
        self.public = []
        self.set_player_id(agent_id)
        self.opp_id = (agent_id + 1) % self.player_num
        self.opponent = agent
        self.opponent_name = agent_name
        self.env = ToyPokerEnv(num_players=player_num, allow_step_back=True)
        self.classifier = Classifier(self.opp_id)
        # read first round EHS
        self.first_round_table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_EHS_vector.csv', index_col=None)
        for action in ToyPokerAction:
            self.action_space.append(action.value)
        super().__init__(agent_type=agent_type)

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
        if (len(state.history) == 0) or (len(state.history) == 1):
            return True
        return False

    def init_information(self, state):
        '''
        Initialize private environment and update information to the private information
        '''
        hand = [[] for _ in range(state.num_player)]
        hand[self.player_id] = deepcopy(state.hand_cards)
        self.public = deepcopy(state.public_cards)
        self.env.init_game(button=state.button, hand_cards=hand, public_cards=self.public)
        self.init_range()
        # if opponent makes an action first
        if state.previous_all_actions:
            self.update_information(state)

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
            cards_str = PokerState.sort_hand_cards(hands)
            self.range[cards_str] = 1
        sum_up = sum(self.range.values())
        for cards_str in self.range.keys():
            self.range[cards_str] /= sum_up

    def update_information(self, state):
        '''
        Update the situation of eval_env to private environment
        Args:
            state: (PokerState)
            n(int 1 or 2): since the private environment has to be late for two steps
                or one steps at the beginning if opponent make action first.
        '''
        actions = deepcopy(state.latest_actions)
        if not (len(self.public) == len(state.public_cards)):
            for i, action in enumerate(actions):
                # since I don't know which step give the new_public_cards
                self.env.step(action, new_public_cards=state.public_cards[-2:])
            self.update_range(state, new_public_cards=state.public_cards[-2:])
            self.public = deepcopy(state.public_cards)
        else:
            for i, action in enumerate(actions):
                self.env.step(action)
            self.update_range(state)

    def update_range(self, state, new_public_cards=None):
        '''
        Update the cards range of opponent according to the change of public state
        and opponent's strategy and latest action
        '''
        new_range = {}
        # There are new public cards
        if new_public_cards:
            self.all_cards = []
            deck = self.env.get_deck()
            for com in combinations(deck, 2):
                hands = [card.get_index() for card in com]
                self.all_cards.append(hands)
                cards_str = PokerState.sort_hand_cards(hands)
                new_range[cards_str] = self.range[cards_str]
            # normalization
            sum_up = sum(new_range.values())
            for cards_str in new_range.keys():
                new_range[cards_str] /= sum_up
        else:
            new_range = self.range
        # Update opponent's range according to opponent's strategy and latest action
        actions = deepcopy(state.latest_actions)
        if len(actions) >= 2:
            for _ in actions[1:]:
                self.env.step_back()
            for action in actions[1:]:
                for cards in self.all_cards:
                    previous_state = self.env.reset_cards(hand=cards, player_id=self.opp_id)
                    action_prob = self.get_opp_action_prob(previous_state)  # get action prob
                    cards_str = PokerState.sort_hand_cards(cards)
                    new_range[cards_str] *= action_prob[self.action_space.index(action)]
                    self.env.give_back_cards(player_id=self.opp_id)
                self.env.step(action, new_public_cards)
            if sum(new_range.values()) == 0:
                print('###### wrong ########')
                raise Exception("Wrong action")
            # normalization
            sum_up = sum(new_range.values())
            for cards_str in new_range.keys():
                new_range[cards_str] /= sum_up

        self.range = new_range
        return

    def get_opp_action_prob(self, state):
        '''
        Agent mode: Get opponent action probability according to opponent modeling
        Test mode: directly use opponent strategy
        '''
        action_prob = np.zeros(self.action_num)
        if self.mode == 'agent':
            if self.opponent_name == 'random':
                action_prob = np.array([1.0 / self.action_num for _ in range(self.action_num)])
            else:
                action = self.opponent.eval_step(state)
                action_prob[self.action_space.index(action)] = 1.0
        elif self.mode == 'test':
            if self.opponent_name == 'mccfr':
                action_prob = self.opponent.get_action_probs(state)
            else:
                action_length = self.action_num
                action_prob = np.array([1.0 / action_length for _ in range(action_length)])
        if 1 in action_prob:
            action_prob = np.array([0.2 / (self.action_num - 1)] * self.action_num)
            action_prob[action_prob == 1] = 0.8
        return action_prob

    def opponent_modeling(self, trajectories_list):
        '''
        Find out opponent's type
        Args: (list) the trajectories of several episode
        '''
        self.opponent_name = self.classifier.get_agent_type([trajectories_list])
        if self.opponent_name == 'TP':
            self.opponent = ToypokerTPAgent()
        elif self.opponent_name == 'LP':
            self.opponent = ToypokerLPAgent()
        elif self.opponent_name == 'LA':
            self.opponent = ToypokerLAAgent()
        elif self.opponent_name == 'INIT' or self.opponent_name == 'TA':
            self.opponent_name = 'random'
        else:
            raise Exception("TypeError")

