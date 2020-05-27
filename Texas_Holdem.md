# Texas Holdem Poker AI

## Introduction

德州扑克AI目前可分为两大部分：GTO部分(Game Theoretically Optimal)，以及EO部分(Exploitability Optimal)。目前GTO部分已经比较成熟，本综述主要介绍GTO部分内容，涉及一小部分EO。



## 一、抽象

### 信息集抽象

1. 基于EMD距离的K-means聚类方法
   1. 蓝图策略：根据CMU论文聚类数200、500为宜，1000将会使训练难以收敛，
   2. 子博弈搜索：轮数越少，可进行越细的抽象，如在第二轮使用500类，第三轮1000类。
2. Distributional-aware的K-means聚类方法

### 动作抽象

1. 底池倍数抽象
2. 根据动作的价值来进行实时抽象（缺）



## 二、蓝图策略：粗抽象下的纳什均衡策略

### 1. CFR系列

#### (1) CFR/CFR+: 遍历所有信息集，遍历所有动作

```python
 def traverse_cfr(self, player_id):
        '''
        Traverse the game tree, update the regrets.

        Args:
            player_id (int): The traverser id of this traverse

        Returns:
            state_utilities (float): the expected value/payoff of current player
        '''
        # (1) For terminal node, return the game payoff.
        if self.env.is_over():
            return self.env.get_payoffs()[player_id]
        current_player = self.env.get_player_id()
        state = self.env.get_state()
        legal_actions = self.encode_action(state)
        # determine the strategy at this infoset
        action_probs = self.calculate_strategy(info_set, legal_actions)
        # calculate the value_expectation of current state/history
        for set_cards in self.all_possible_cards():
        	value_expectation = 0  # v_h
          # traverse each action
          action_utilities = np.zeros(self.action_num)
          for action in legal_actions:
            self.env.step(self.decode_action(action), set_cards)
            action_utilities[action] = self.traverse_cfr(player_id)
            value_expectation += action_probs[action] * action_utilities[action]
            self.env.step_back()
        	# (2) If current player is the traverser, update regret.
        	if current_player == player_id:
            # update the regret of each action
            info_set = self.encode_state(state, set_cards)
            for action in legal_actions:
              self.regrets[info_set][action] += action_utilities[action] - value_expectation
        return value_expectation
       
```

#### (2) MCCFR - external sampling: 采样信息集，遍历自己动作，采样对手动作

```python
    def traverse_mccfr(self, player_id):
        '''
        Traverse the game tree, update the regrets.

        Args:
            player_id (int): The traverser id of this traverse

        Returns:
            state_utilities (float): the expected value/payoff of current player
        '''
        # (1) For terminal node, return the game payoff.
        if self.env.is_over():
            return self.env.get_payoffs()[player_id]
        current_player = self.env.get_player_id()
        state = self.env.get_state()
        info_set = self.encode_state(state)
        legal_actions = self.encode_action(state)
        # determine the strategy at this infoset
        action_probs = self.calculate_strategy(info_set, legal_actions)
        # (2) If current player is the traverser, traverse each action.
        if current_player == player_id:
            # calculate the value_expectation of current state/history
            value_expectation = 0  # v_h
            # traverse each action
            action_utilities = np.zeros(self.action_num)
            for action in legal_actions:
                self.env.step(self.decode_action(action))
                action_utilities[action] = self.traverse_mccfr(player_id)
                value_expectation += action_probs[action] * action_utilities[action]
                self.env.step_back()
            # update the regret of each action
            for action in legal_actions:
                self.regrets[info_set][action] += action_utilities[action] - value_expectation
            return value_expectation
        # (3) For the opponent node, sample an action from the probability distribution.
        else:
            action = np.random.choice(self.action_num, p=action_probs)
            self.env.step(self.decode_action(action))
            value_expectation = self.traverse_mccfr(player_id)
            self.env.step_back()
            return value_expectation
```

对于MCCFR - outcome sampling，只需要将遍历自己动作改成采样自己动作即可

### 2. XFP系列

```python
def traverse_xfp(self, player_id):
        '''
        Traverse the game tree, update the regrets.

        Args:
            player_id (int): The traverser id of this traverse

        Returns:
            state_utilities (float): the expected value/payoff of current player
        '''
        # (1) For terminal node, return the game payoff.
        if self.env.is_over():
            return self.env.get_payoffs()[player_id]
        current_player = self.env.get_player_id()
        state = self.env.get_state()
        legal_actions = self.encode_action(state)
        # determine the strategy at this infoset
        action_probs = self.calculate_strategy(info_set, legal_actions)
        # calculate the value_expectation of current state/history
        for set_cards in self.all_possible_cards():
          # traverse each action
          action_utilities = np.zeros(self.action_num)
          for action in legal_actions:
            self.env.step(self.decode_action(action), set_cards)
            action_utilities[action] = self.traverse_mccfr(player_id)
            self.env.step_back()
        	# (2) If current player is the traverser, update regret.
        	if current_player == player_id:
            # update the regret of each action
            info_set = self.encode_state(state, set_cards)
            self.policy[info_set] = self.decode_action(np.argmax(action_utilities))
        return value_expectation
```



## 二、子博弈搜索：细抽象下的纳什均衡策略

### 1. 安全子博弈

### 2. 深度限制子博弈搜索



## 三、子博弈搜索：细抽象下的EO策略

### 1. 不安全子博弈