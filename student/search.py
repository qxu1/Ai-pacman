"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
import heapq
from pacai.util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    # *** Your Code Here ***\
    states = [(problem.startingState(), [])]
    explored = set()
    while states:
        state, path = states.pop()
        if problem.isGoal(state):
            return path
        if state in explored:
            continue
        explored.add(state)
        for newstate, action, cost in problem.successorStates(state):
            states.append((newstate, path + [action]))
    raise NotImplementedError()


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    states = [(problem.startingState(), [])]
    explored = set()
    while states:
        state, path = states.pop(0)
        if problem.isGoal(state):
            return path
        if state in explored:
            continue
        explored.add(state)
        for newstate, action, cost in problem.successorStates(state):
            states.append((newstate, path + [action]))
    raise NotImplementedError()


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    states = [(0, problem.startingState(), [])]
    explored = set()
    while states:
        cost, state, path = heapq.heappop(states)
        if problem.isGoal(state):
            return path
        if state in explored:
            continue
        explored.add(state)
        for newstate, action, newcost in problem.successorStates(state):
            heapq.heappush(states, (cost + newcost, newstate, path + [action]))
    # *** Your Code Here ***
    raise NotImplementedError()


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    explored = []
    queue = PriorityQueue()
    queue.push(((problem.startingState()), []), 0)
    while not queue.isEmpty():
        state, path = queue.pop()
        if problem.isGoal(state):
            return path
        if state in explored:
            continue
        explored.append(state)
        newstate = problem.successorStates(state)
        for i in newstate:
            if i[0] in explored:
                continue
            else:
                newcost = problem.actionsCost(path + [i[1]]) + heuristic(i[0], problem)
                queue.push((i[0], path + [i[1]]), newcost)
    # *** Your Code Here ***
    raise NotImplementedError()
