class Classifier():
    '''
    classifier: used for classifying a specified agent and save useful information

    Attributes:
        agent_id (int): id of specified agent
        added_traj_num (int): number of added trajectories
        num_call (int): the number of spcified agent calling
        num_raise (int): the number of spcified agent raising
        AF (float): agressive factor, ratio of raise number to call number
        RAOT (int): number of raising at least one time
        VPIP (float): ratio of RAOT to added trajectoires number
        agent_type (str): type if specified agent (TP/TA/LA/LP)
        is_updated (boolean): a switch to determine whether AF and VPIP have been updated
    '''

    def __init__(self, agent_id):
        '''
        Initialize classifier class.
        '''
        self.agent_id = agent_id
        self.added_traj_num = 0
        self.num_call = 0
        self.num_raise = 0
        self.AF = None
        self.RAOT = 0
        self.VPIP = None
        self.agent_type = []

    def update_call_raise_num(self, trajectories):
        '''
        update call number and raise number base on one trajectories.

        Args:
            trajectories (list): a list of trajectories generated from the environment.
        '''
        rato = 0
        for element in trajectories[self.agent_id]:
            if isinstance(element, str) and element == 'call':
                self.num_call += 1
            if isinstance(element, str) and 'raise' in element:
                self.num_raise += 1
                rato = 1
        # update the number of raising at least one time
        self.RAOT += rato

    def update_AF_VPIP(self):
        '''
        calculate AF and VPIP.
        '''
        self.AF = self.num_raise / self.num_call
        self.VPIP = self.RAOT / self.added_traj_num

    def update_agent_type(self):
        '''
        update the type of agent base on (TightPassive/TightAggressive/LooseAggressive/LoosePassive) updated AF and VPIP
        '''
        if self.VPIP < 0.28 and self.AF >= 1:
            self.agent_type = 'TA'
        elif self.VPIP >= 0.28 and self.AF >= 1:
            self.agent_type = 'LA'
        elif self.VPIP < 0.28 and self.AF < 1:
            self.agent_type = 'TP'
        elif self.VPIP >= 0.28 and self.AF < 1:
            self.agent_type = 'LP'

    def get_agent_type(self, trajectories_list):
        '''
        update classifier base on list of trajectories.

        Args:
            trajectories_list (list): a list of trajectories for each episode.
        '''
        # update call and raise number
        for trajectories in trajectories_list:
            self.update_call_raise_num(trajectories)
            self.added_traj_num += 1
        if self.check():
            self.update_AF_VPIP()
            self.update_agent_type()
        else:
            self.agent_type = 'INIT'
        return self.agent_type

    def check(self):
        '''
        check whether the denominator of AF is zero.

        Args:
            call_num (list): a list of times of every player calling in each episodes.
            agent_id (int): id of specified agent.

        Returns:
            (boolean): True/False.
        '''
        if self.num_call > 0:
            return True
        return False
