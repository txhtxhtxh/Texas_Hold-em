import numpy as np
from itertools import combinations
from copy import deepcopy
from poker_env.ToyPoker.state import PokerState
from poker_env.agents.base_agent import Agent
from poker_env.agents.baseline_agent.NoLimitToypoker_baseline_agent import NolimitToypokerTPAgent, NolimitToypokerLPAgent, NolimitToypokerLAAgent
from poker_env.ToyPoker.action import Action as NLToypokerAction
from poker_env.agents.lbr_agent.opponent_modeling import Classifier
from poker_env.NoLimitToyPoker.env import NoLimitToyPokerEnv
from poker_env.dependency.pokerstove.utils import get_data

class NoLimitToyPokerLBR(Agent):
    '''
    A Implementation of a full version of Local Best Response agent

    paper: Equilibrium Approximation Quality of Current No-Limit Poker Bots
    '''

    def __init__(self, mode, agent_id, agent=None, player_num=2, training_times=10, agent_name='random'):
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
        self.range = {}
        self.public = []
        self.set_player_id(agent_id)
        self.training_times = training_times
        self.opp_id = (agent_id + 1) % self.player_num
        self.opponent = agent
        self.opponent_name = agent_name
        self.env = NoLimitToyPokerEnv(num_players=player_num, allow_step_back=True)
        self.classifier = Classifier(self.opp_id)
        for action in NLToypokerAction:
            self.action_space.append(action.value)

    @property
    def action_num(self):
        return len(self.action_space)

    def encode_action(self, state):
        encode_actions = []
        print(self.action_space)
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
            self.update_information(state)

        action_utilities = self.get_value(deepcopy(state), deepcopy(self.env))
        if max(action_utilities) <= 0:
            action = NLToypokerAction.FOLD.value
        else:
            action = self.decode_action(np.argmax(action_utilities))
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
            for cards in self.all_cards(env):
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
                if str_action == NLToypokerAction.CALL.value:
                    action_utilities[a] = wp * pot - (1 - wp) * asked
                elif str_action == NLToypokerAction.FOLD.value:
                    continue
                else:
                    raise_size = state.raise_money
                    action_utilities[a] = fp * pot + (1 - fp) * (
                                wp * (pot + raise_size) - (1 - wp) * (asked + raise_size))
        return action_utilities

    def WpRollout(self, state):
        '''
        Get LBR wining probability
        '''

        if len(state.public_cards) == 0:
            return 0.5
        else:
            hand_cards = [''.join(state.hand_cards)]
            public_cards = ''.join(state.public_cards)
            ehs_value = get_data(hand_cards, public_cards)
        return ehs_value

    def all_cards(self, env=None):
        all_cards = []
        if not env:
            env = self.env
        deck = env.get_deck()
        for com in combinations(deck, 2):
            hands = [card.get_index() for card in com]
            all_cards.append(hands)
        return all_cards

    def init_information(self, state):
        '''
        Initialize private environment and update information to the private information
        '''
        self.public = []
        hand = [[] for _ in range(state.num_player)]
        hand[self.player_id] = deepcopy(state.hand_cards)
        self.env.init_game(button=state.button, hand_cards=hand)
        all_cards = self.all_cards()
        self.init_range(all_cards)
        # if opponent makes an action first
        if state.previous_all_actions:
            self.update_information(state)

    def init_range(self, all_cards):
        '''
        Initialize opponent's cards range according to the cards have been dealt
        '''
        # get all possible cards of opponent and init range
        self.range = {}
        for hands in all_cards:
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
        new_cards_num = len(state.public_cards) - len(self.public)
        if new_cards_num > 0:
            for i, action in enumerate(actions):
                # since I don't know which step give the new_public_cards
                self.env.step(action, new_public_cards=state.public_cards[-new_cards_num:])

            self.update_range(state, new_public_cards=state.public_cards[-new_cards_num:])
            self.public = deepcopy(state.public_cards)
        else:
            for action in actions:
                self.env.step(action)
            self.update_range(state)

    def update_range(self, state, new_public_cards=None):
        '''
        Update the cards range of opponent according to the change of public state
        and opponent's strategy and latest action
        '''
        new_range = {}
        # There are new public cards
        all_cards = self.all_cards()
        if new_public_cards:
            for hands in all_cards:
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
                for cards in all_cards:

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
            self.opponent = NolimitToypokerTPAgent()
        elif self.opponent_name == 'LP':
            self.opponent = NolimitToypokerLPAgent()
        elif self.opponent_name == 'LA':
            self.opponent = NolimitToypokerLAAgent()
        elif self.opponent_name == 'TA' or self.opponent_name == 'INIT':
            self.opponent_name = 'random'
        else:
            raise Exception("TypeError")

