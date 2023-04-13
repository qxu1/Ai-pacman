from pacai.agents.capture.capture import CaptureAgent
from pacai.util import reflection
import random
def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Randomly pick an action.
        """

        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

class offenseAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        actions = state.getLegalActions(0)
        actions.remove('Stop')
        score = -float('inf')
        move = ''
        for action in actions:
            nextState = state.generateSuccessor(0, action)
            newScore = self.minValue(nextState, 0, 1)
            if newScore > score:
                score, move = newScore, action
        return move

    def maxValue(self, state, depth, agentNum):
        depth += 1
        if state.isOver() or depth == self.getTreeDepth():
            return self.getEvaluationFunction()(state)
        maxval = -float('inf')
        actions = state.getLegalActions(agentNum)
        nextNum = (agentNum + 1) % state.getNumAgents()
        for action in actions:
            nextState = state.generateSuccessor(agentNum, action)
            maxval = max(maxval, self.minValue(nextState, depth, nextNum))
        return maxval

    def minValue(self, state, depth, agentNum):
        if state.isOver():
            return self.getEvaluationFunction()(state)
        minval = float('inf')
        actions = state.getLegalActions(agentNum)
        nextNum = (agentNum + 1) % state.getNumAgents()
        for action in actions:
            nextState = state.generateSuccessor(agentNum, action)
            if agentNum == state.getNumAgents() - 1:
                minval = min(minval, self.maxValue(nextState, depth, nextNum))
            else:
                minval = min(minval, self.minValue(nextState, depth, nextNum))
        return minval