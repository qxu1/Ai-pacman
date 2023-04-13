import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        foodList = oldFood.asList()
        for food in foodList:
            mhfood = min(float("inf"), distance.manhattan(newPosition, food))
        for ghost in successorGameState.getGhostPositions():
            if (distance.manhattan(newPosition, ghost) <= 1):
                return float('-inf')
        # prevent the floor division by 0
        if mhfood == 0:
            mhfood = 1
        return successorGameState.getScore() + 1.0 / mhfood


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        best_action = None
        best_utility = float("-inf")
        legal_actions = state.getLegalActions(0)
        for action in legal_actions:
            utility = self.value(1, 0, state.generateSuccessor(0, action))
            if utility > best_utility:
                best_utility = utility
                best_action = action

        return best_action

    def value(self, state, depth, gamestate):
        legalactions = gamestate.getLegalActions()
        if gamestate.isLose() or gamestate.isWin() or depth == self.index:
            return self._evaluationFunction(gamestate)
        if not legalactions:
            return self._evaluationFunction(gamestate)
        else:
            nextstate = (state + 1) % gamestate.getNumAgents()
            depth = depth + 1 if nextstate == 0 else depth
            if nextstate == 0:
                for action in legalactions:
                    gamesuccessor = gamestate.generateSuccessor(state, action)
                    maxvalue = max(self.value(nextstate, depth, gamesuccessor))
                    return maxvalue
            else:
                for action in legalactions:
                    gamesuccessor = gamestate.generateSuccessor(state, action)
                    minvalue = min(self.value(nextstate, depth, gamesuccessor))
                    return minvalue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(self.index)
        if not legalActions:
            return None
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in legalActions:
            successor = gameState.generateSuccessor(self.index, action)
            value = self.minimax(successor, self.index + 1, 0, alpha, beta)
            if value > bestValue:
                bestAction = action
                bestValue = value
            alpha = max(alpha, bestValue)
        return bestAction

    def minimax(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        if agentIndex == self.index:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        value = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.minimax(successor, agentIndex + 1, depth, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        value = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            value = min(value, self.minimax(successor, agentIndex + 1, depth, alpha, beta))
            if value < alpha:
                return value
            beta = min(beta, value)
        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return None
        results = [self.expectimax(gameState.generateSuccessor(self.index, action),
                                   self.index, self.index + 1) for action in actions]
        return actions[results.index(max(results))]

    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self._evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)
        numLegalActions = len(legalActions)
        if agentIndex == self.index:
            values = [self.expectimax(gameState.generateSuccessor(agentIndex, action),
                                      depth, (agentIndex + 1) % gameState.getNumAgents())
                      for action in legalActions]
            return max(values)
        else:
            values = [self.expectimax(gameState.generateSuccessor(agentIndex, action),
                                      depth, (agentIndex + 1) % gameState.getNumAgents())
                      for action in legalActions]
            return sum(values) / numLegalActions


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <the code calculates the score for the current state of the pacman game
    ,at first i declare 3 variable to gets the pacman's position,position of food and
    position of the ghost,then i calculate the minimum distance from pacman to food and
    minimum distance of from pacman to ghost, if the distance between pacman and ghost
    is less than 2, the score will set it to negative infinity which is loss, if it's
    winning it'll calculate the score and return the result>
    """
    newPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghost_positions = currentGameState.getGhostPositions()

    min_food_dist = min([distance.manhattan(newPosition, f) for f in food], default=float('inf'))
    ghost_dist = min([distance.manhattan(newPosition, g) for g in ghost_positions])

    if ghost_dist < 2:
        return float('-inf')

    remainingfood = currentGameState.getNumFood()
    remainingcap = len(currentGameState.getCapsules())

    food_score = 10000 / (remainingfood + 1)
    caps_score = 100 / (remainingcap + 1)
    food_dist_score = 10 / (min_food_dist + 1)
    ghost_score = ghost_dist
    if currentGameState.isLose():
        end_game_score = float('-inf')
    elif currentGameState.isWin():
        end_game_score = float('inf')
    else:
        end_game_score = 0
    total_score = food_score + caps_score + food_dist_score + ghost_score + end_game_score

    return total_score


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
