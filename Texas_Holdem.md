# Texas Holdem Poker AI

## Introduction

德州扑克AI目前可分为两大部分：GTO部分(Game Theoretically Optimal)，以及EO部分(Exploitability Optimal)。目前GTO部分已经比较成熟，本综述主要介绍GTO部分内容，涉及一小部分EO。



## 一、抽象

### 信息集抽象

1. 基于EHS的K-means聚类方法
   1. 蓝图策略抽象：根据CMU论文，每一轮的信息集聚类数200、500为宜，1000以上将会使训练难以收敛
   2. 子博弈搜索抽象：轮数越少，可进行越细的抽象，如在第二轮使用500类，第三轮1000类。
   3. 算法：
      1. 随机初始化200个类的中心点，每个点的坐标为该点可观测牌型的获胜概率，点间距离定义为获胜概率绝对差
      2. 计算所有样本点与各个类中心之间的距离，然后把样本点划入最近的类中
      3. 根据类中已有的样本点，重新计算类中心（平均值）
      4. 重复上两步直至收敛
2. 基于EMD的K-means聚类方法
   1. 该方法只是重新定义了各点坐标与距离，将所有可能的公共牌都发一遍，对不同的公共牌获得一个获胜概率，故此时坐标点为向量。可利用EMD距离或者L2距离来定义坐标点之间的距离。

### 动作抽象

1. 动作转译：直接假设动作的单位增加为\$100，将不在动作抽象内的动作归类到抽象动作，例如$101的属于\$100这一个下注动作。

2. 池内金额的倍数

3. 冷扑大师

   1. 前两轮的动作抽象为细粒度，所以直接进行动作转译，误差较小

   2. 自我提升机制，在对打过程中完善蓝图策略，往蓝图策略中细化粗粒度动作抽象，同时能够学习对手是如何利用动作来对局的(一种对手建模)：

      1. 在休息时间根据对局数据找出k个对手动作：出现频率高，距离抽象动作远的对手动作。

      2. 将这些动作放入蓝图策略的计算当中，能够计算并收敛的那一些off-tree action的响应的价值，将在第二天加入蓝图策略。

         



## 二、传统方法

## 1.蓝图策略：粗抽象下的纳什均衡策略

### 	(1). CFR系列

#### 	a) CFR/CFR+: 遍历所有信息集，遍历所有动作

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

#### 	b) MCCFR - external sampling: 采样信息集，遍历自己动作，采样对手动作

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

​	对于MCCFR - outcome sampling，只需要将遍历自己动作改成采样自己动作即可

### 	(2) XFP系列

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
            action_utilities[action] = self.traverse_xfp(player_id)
            self.env.step_back()
        	# (2) If current player is the traverser, update regret.
        	if current_player == player_id:
            # update the regret of each action
            info_set = self.encode_state(state, set_cards)
            self.policy[info_set] = self.decode_action(np.argmax(action_utilities))
        return value_expectation
```

## 2.子博弈搜索：细抽象下的纳什均衡策略

### 	(1) 安全子博弈: CFR+

```python
def traverse_cfr(self, player_id, blue_print_value):
        '''
        Traverse the game tree, update the regrets.

        Args:
            player_id (int): The traverser id of this traverse

        Returns:
            state_utilities (float): the expected value/payoff of current player
        '''
        # (1) For terminal node, return the game payoff.
        if self.env.is_over():
            return self.env.get_payoffs()[player_id] - blue_print_value # shifted
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
            action_utilities[action] = self.traverse_cfr(player_id, blue_print_value)
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

​		安全子博弈即为在细抽象下，对子博弈进行CFR+算法求出子博弈的纳什均衡策略，只是在叶子结点的reward需要减去子博弈入口处的状态价值(precompute)，在上述代码中为 blue_print_value。

## 3.子博弈搜索：细抽象下的EO策略

### 	1. 深度限制子博弈搜索

```python
def traverse_cfr(self, player_id, depth):
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
        elif depth == self.max_search_depth:
          	return self.get_DLvalue()
          
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
            action_utilities[action] = self.traverse_cfr(player_id, depth + 1)
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

​	深度限制子博弈搜索限制了搜索深度，当迭代到达搜索深度self.max_depth时，如果对局还未结束，则利用一个深度限制奖励来代替：self.get_DLvalue()。self.get_DLvalue()基于对手建模，假设对手会选择10个策略的其中一个，维护一个10维的概率分布，最后按概率加权每个策略下在这个状态的价值(precompute)。

### 2. 不安全子博弈

```python
def traverse_cfr(self, player_id, blue_print_value):
        '''
        Traverse the game tree, update the regrets.

        Args:
            player_id (int): The traverser id of this traverse

        Returns:
            state_utilities (float): the expected value/payoff of current player
        '''
        # (1) For terminal node, return the game payoff.
        if self.env.is_over():
            return self.env.get_payoffs()[player_id] - blue_print_value # shifted
        current_player = self.env.get_player_id()
        legal_actions = self.encode_action(state)
        # determine the strategy at this infoset
        action_probs = self.calculate_strategy(info_set, legal_actions)
        # calculate the value_expectation of current state/history
        for set_cards in self.all_possible_cards():
        	value_expectation = 0  # v_h
          # traverse each action
          if current_player == player_id:
            action_utilities = np.zeros(self.action_num)
            for action in legal_actions:
              self.env.step(self.decode_action(action), set_cards)
              action_utilities[action] = self.traverse_cfr(player_id, blue_print_value)
              value_expectation += action_probs[action] * action_utilities[action]
              self.env.step_back()
            # update the regret of each action
            state = self.env.get_state()
            info_set = self.encode_state(state, set_cards)
            for action in legal_actions:
              self.regrets[info_set][action] += action_utilities[action] - value_expectation
        	else:
            state = self.env.get_state()
            self.env.step(self.blue_print(state), set_cards)
        return value_expectation
```
## 三、神经网络方法

### 1. DeepCFR

拟合遗憾值的网络+拟合策略的网络

类似于强化学习的GPI，此处$$\pi, V_{\pi}$$均用NN拟合

![GPI](/Users/txh/Texas_work/Github/Texas_Hold-em/figure/GPI.png)

![11381591245280_.pic_hd](/Users/txh/Texas_work/Github/Texas_Hold-em/figure/11381591245280_.pic_hd.jpg)

实验结果：

由于不基于抽象，需要较大训练量。batch size目前为50000，训练效果仍然一般。

### 2.DeepStack

拟合价值的网络

随机生成残局（3轮以下的残局），用NN记录所有节点的Value和Regret

#### 以上两种方法均可基于MCCFR收集memory

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
            for action in legal_actions:
                self.regrets[info_set][action] += action_utilities[action] - value_expectation
            self.memory.append([state, value_expectation, self.get_policy(self.regrets)])
            return value_expectation
        # (3) For the opponent node, sample an action from the probability distribution.
        else:
            action = np.random.choice(self.action_num, p=action_probs)
            self.env.step(self.decode_action(action))
            value_expectation = self.traverse_mccfr(player_id)
            self.env.step_back()
            self.memory.append([state, value_expectation])
            return value_expectation
          
def train()
			for _ in training_times:
        for _ in range(self.sampling_times):
          self.traverse_mccfr()
        self.train_NN(self.memory)
```

## 四、检测方法

### 1. Best Response

```python
def get_value(self, player_id):
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
        	if current_player == player_id:
						for action in legal_actions:
              self.env.step(self.decode_action(action), set_cards)
              action_utilities[action] = self.get_value(player_id)
              self.env.step_back()
        		return max(actino_utility), np.argmax(action_utility)
          else:
            action = self.opponent_strategy(self.env.get_state())
      			self.env.step(action, set_cards)
            utility = self.get_value(player_id)
            self.env.step_back()
						return utility

def predict(self, state)
				action_utility = self.get_value(state)
  			return self.action_space(np.argmax(action_utility))
```

### 2. Local Best Response

```python
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

def predict(self, state)
				action_utility = self.get_value(state)
  			return self.action_space(np.argmax(action_utility))
```

