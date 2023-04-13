from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
import random


class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <The getValue method,it returns the best value of the state,
    if there's no state exist it'll return 0.0 instead. the getPolicy method
    uses the getQvalue method to determine the best action to take,also it'll check
    if multiple actions have the same Q-value, it'll make one action in random. the update method
    will update the Q-value for a state and action using the BELLMAN equation. The getAction
    method it'll return the best action to take using the getPolicy method, if there's not
    actions to take, it'll return None.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.weights = {}
        self.value = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        if (state, action) in self.value:
            return self.value[(state, action)]
        else:
            return 0.0

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        bestValue = float('-inf')
        for action in legalActions:
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
        return bestValue

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        maxVal = float("-inf")
        bestActions = []
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > maxVal:
                maxVal = qValue
                bestActions = [action]
            elif qValue == maxVal:
                bestActions.append(action)
        return random.choice(bestActions)

    def update(self, state, action, nextState, reward):
        alp = reward + self.getDiscountRate() * self.getValue(nextState)
        qValue = self.getQValue(state, action)
        self.value[(state, action)] = qValue + self.getAlpha() * (alp - qValue)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        return self.getPolicy(state)


class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <for the getQValue method,it returns the q-value of the state and action pair.
    then calculates the feature vector and weight vector for each feature and add it up.update
    method, it updates the weight of the q-learning agent. it calculates the
    reward from it's action of the current state and predicted qvalue of the next state.
    for the final method, it checks if the number of episode to the number of the training
    . if it's equal it'll print out the weight for debugging.>
    """

    def __init__(self, index,
                 extractor='pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = {}

    def getQValue(self, state, action):
        """
        Should return `Q(state, action) = w * featureVector`,
        where `*` is the dotProduct operator.
        """
        features = self.featExtractor.getFeatures(state, action)
        qValue = 0.0
        for feature, value in features.items():
            qValue += self.weights.get(feature, 0.0) * value
        return qValue

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(state, action)
        reward += self.discountRate() * self.getValue(nextState) - self.getQValue(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * reward * features[feature]

    def final(self, state):
        """
        Called at the end of each game.
        """
        # Call the super-class final method.
        super().final(state)
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            print("Weights is:", self.weights)
        raise NotImplementedError()
