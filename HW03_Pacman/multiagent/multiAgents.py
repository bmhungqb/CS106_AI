# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)
        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)
        #
        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))
        #
        # return result

# class AlphaBetaAgent(MultiAgentSearchAgent):
#     """
#     Your minimax agent with alpha-beta pruning (question 3)
#     """
#
#     def getAction(self, gameState):
#         """
#         Returns the minimax action using self.depth and self.evaluationFunction
#         """
#         "*** YOUR CODE HERE ***"
#         def alphabeta(state, alpha, beta):
#             bestValue, bestAction = None, None
#             print(state.getLegalActions(0))
#             value = []
#             for action in state.getLegalActions(0):
#                 #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
#                 succ  = minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta)
#                 value.append(succ)
#                 if bestValue is None:
#                     bestValue = succ
#                     bestAction = action
#                 else:
#                     if succ > bestValue:
#                         bestValue = succ
#                         bestAction = action
#             print(value)
#             return bestAction
#
#         def minValue(state, agentIdx, depth, alpha, beta):
#             if agentIdx == state.getNumAgents():
#                 return maxValue(state, 0, depth + 1, alpha, beta)
#             value = None
#             for action in state.getLegalActions(agentIdx):
#                 succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
#                 if value is None:
#                     value = succ
#                 else:
#                     value = min(value, succ)
#                 if value <= alpha:
#                     return value
#                 beta = min(beta, value)
#             if value is not None:
#                 return value
#             else:
#                 return self.evaluationFunction(state)
#
#
#         def maxValue(state, agentIdx, depth, alpha, beta):
#             if depth > self.depth:
#                 return self.evaluationFunction(state)
#             value = None
#             for action in state.getLegalActions(agentIdx):
#                 succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
#                 if value is None:
#                     value = succ
#                 else:
#                     value = max(value, succ)
#                 if value >= beta:
#                     return value
#                 alpha = max(alpha, value)
#
#             if value is not None:
#                 return value
#             else:
#                 return self.evaluationFunction(state)
#
#         action = alphabeta(gameState, float("-inf"), float("inf"))
#         print("Final action: ", action)
#         return action
#         # util.raiseNotDefined()
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maximizer(agent, depth, game_state, a, b):  # maximizer function
            v = float("-inf")
            for newState in game_state.getLegalActions(agent):
                v = max(v, alphabetaprune(1, depth, game_state.generateSuccessor(agent, newState), a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def minimizer(agent, depth, game_state, a, b):  # minimizer function
            v = float("inf")

            next_agent = agent + 1  # calculate the next agent and increase depth accordingly.
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1

            for newState in game_state.getLegalActions(agent):
                v = min(v, alphabetaprune(next_agent, depth, game_state.generateSuccessor(agent, newState), a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v

        def alphabetaprune(agent, depth, game_state, a, b):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return self.evaluationFunction(game_state)

            if agent == 0:  # maximize for pacman
                return maximizer(agent, depth, game_state, a, b)
            else:  # minimize for ghosts
                return minimizer(agent, depth, game_state, a, b)

        """Performing maximizer function to the root node i.e. pacman using alpha-beta pruning."""
        utility = float("-inf")
        action = Directions.WEST
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            ghostValue = alphabetaprune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if ghostValue > utility:
                utility = ghostValue
                action = agentState
            if utility > beta:
                return utility
            alpha = max(alpha, utility)

        return action
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state):
            bestValue, bestAction = None, None
            value = []
            for action in state.getLegalActions(0):
                succ = expectValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            return bestAction

        def expectValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            succs = []
            for action in state.getLegalActions(agentIdx):
                succs.append(state.generateSuccessor(agentIdx, action))
            value = None
            for succ in succs:
                v = expectValue(succ, agentIdx+1, depth)
                if value is None:
                    value = v
                else:
                    value += v
            if value is not None:
                return float(value) / len(succs)
            else:
                return self.evaluationFunction(state)
        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = expectValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = expectimax(gameState)
        # print("Final action: ", action)
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostPositions = currentGameState.getGhostPositions()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    ghostScare = [i for i in newScaredTimes if i > 0]
    score = currentGameState.getScore()
    remainingFood = len(newFood.asList())
    totalCapsule = len(newCapsules)
    # FOOD
    minFoodDistance = 0
    for food in newFood.asList():
        dis_Pacman_Food = manhattanDistance(newPos, food)
        if dis_Pacman_Food < minFoodDistance or minFoodDistance == 0:
            minFoodDistance = dis_Pacman_Food
    # GHOST
    proximity_to_ghosts = 0
    min_ghost_distance = 0
    isGreater4 = 1
    for ghost in newGhostPositions:
        distance = manhattanDistance(newPos, ghost)
        if distance <= 4:
            isGreater4 = 0
        if min_ghost_distance == 0 or min_ghost_distance > distance:
            min_ghost_distance = distance
        if distance == 0:
            proximity_to_ghosts += 1
    # WEIGHTS
    score_weight = 10
    food_weight = -10
    ghost_weight = 5
    remaining_food_weight = -20
    capsule_weight = -200
    proximity_weight = -300
    if isGreater4:
        remaining_food_weight = -40
    else:
        ghost_weight = 10
    if len(ghostScare) > 0:
        capsule_weight = 150
    return score_weight * score + \
           food_weight * minFoodDistance + \
           ghost_weight * min_ghost_distance + \
           remaining_food_weight * remainingFood + \
           proximity_weight * proximity_to_ghosts + \
           capsule_weight * totalCapsule
# Abbreviation
better = betterEvaluationFunction
