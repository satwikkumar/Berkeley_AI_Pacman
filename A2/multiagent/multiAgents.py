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
        #print(gameState.getPacmanState().configuration.pos)
        legalMoves = gameState.getLegalActions()
        #print(legalMoves)
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print(scores)
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

        distance = float("-Inf")
        if action == 'Stop':
            return distance

        for state in newGhostStates:
            if state.getPosition() == tuple(newPos) and (state.scaredTimer == 0):
                return distance

        for food in currentGameState.getFood().asList():
            tempDistance = -1 * (manhattanDistance(newPos, food))
            # * -1 because we are checking for max in the parent function, but we need to get the nearest food first.
            if tempDistance > distance:
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
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
        def minValue(gameState, depth, agentcounter):
            minimum = ["", float("inf")]
            ghostActions = gameState.getLegalActions(agentcounter)

            if not ghostActions:
                return ["",self.evaluationFunction(gameState)]

            for action in ghostActions:
                nextState = gameState.generateSuccessor(agentcounter, action)
                newVal = minOrMax(nextState, depth, agentcounter + 1)
                if newVal[1] < minimum[1]:
                    minimum = [action, newVal[1]]
            return minimum

        def maxValue(gameState, depth, agentcounter):
            maximum = ["", -float("inf")]
            actions = gameState.getLegalActions(agentcounter)

            if not actions:
                return ["",self.evaluationFunction(gameState)]

            for action in actions:
                nextState = gameState.generateSuccessor(agentcounter, action)
                newVal = minOrMax(nextState, depth, agentcounter + 1)
                if newVal[1] > maximum[1]:
                    maximum = [action, newVal[1]]
            return maximum


        def minOrMax(gameState, depth, agentcounter):
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0

            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return ["",self.evaluationFunction(gameState)]
            elif (agentcounter == 0):
                return maxValue(gameState, depth, agentcounter)
            else:
                return minValue(gameState, depth, agentcounter)

        actionsList = minOrMax(gameState, 0, 0)
        #print(actionsList[0])
        return actionsList[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minValue(gameState, depth, agentcounter, alpha, beta):
            minimum = ["", float("inf")]
            ghostActions = gameState.getLegalActions(agentcounter)

            if not ghostActions:
                return ["",self.evaluationFunction(gameState)]

            for action in ghostActions:
                currState = gameState.generateSuccessor(agentcounter, action)
                newVal = alphaOrbeta(currState, depth, agentcounter + 1, alpha, beta)

                if newVal[1] < minimum[1]:
                    minimum = [action, newVal[1]]
                if newVal[1] < alpha:
                    return [action, newVal[1]]
                beta = min(beta, newVal[1])
            return minimum

        def maxValue(gameState, depth, agentcounter, alpha, beta):
            maximum = ["", -float("inf")]
            actions = gameState.getLegalActions(agentcounter)

            if not actions:
                return ["",self.evaluationFunction(gameState)]

            for action in actions:
                currState = gameState.generateSuccessor(agentcounter, action)
                newVal = alphaOrbeta(currState, depth, agentcounter + 1, alpha, beta)

                if newVal[1] > maximum[1]:
                    maximum = [action, newVal[1]]
                if newVal[1] > beta:
                    return [action, newVal[1]]
                alpha = max(alpha, newVal[1])
            return maximum

        def alphaOrbeta(gameState, depth, agentcounter, alpha, beta):
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return ["",self.evaluationFunction(gameState)]
            elif agentcounter == 0:
                return maxValue(gameState, depth, agentcounter, alpha, beta)
            else:
                return minValue(gameState, depth, agentcounter, alpha, beta)

        actionsList = alphaOrbeta(gameState, 0, 0, -float("inf"), float("inf"))
        return actionsList[0]

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

        def findExpect(gameState, depth, agentcounter):
            expectation = ["", 0]
            ghostActions = gameState.getLegalActions(agentcounter)
            probability = 1.0 / len(ghostActions)

            if not ghostActions:
                return ["",self.evaluationFunction(gameState)]

            for action in ghostActions:
                currState = gameState.generateSuccessor(agentcounter, action)
                newVal = expectimax(currState, depth, agentcounter + 1)
                expectation[0] = action
                expectation[1] += newVal[1] * probability
            return expectation

        def maxValue(gameState, depth, agentcounter):
            maximum = ["", -float("inf")]
            actions = gameState.getLegalActions(agentcounter)

            if not actions:
                return ["",self.evaluationFunction(gameState)]

            for action in actions:
                currState = gameState.generateSuccessor(agentcounter, action)
                newVal = expectimax(currState, depth, agentcounter + 1)
                if newVal[1] > maximum[1]:
                    maximum = [action, newVal[1]]
            return maximum

        def expectimax(gameState, depth, agentcounter):
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0

            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return ["",self.evaluationFunction(gameState)]
            elif (agentcounter == 0):
                return maxValue(gameState, depth, agentcounter)
            else:
                return findExpect(gameState, depth, agentcounter)

        actionsList = expectimax(gameState, 0, 0)
        return actionsList[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

   DESCRIPTION:
      The following features are considered and combined:
        - Compute the manhattan distance to the closest food pellet
        - Current score of the game, as the shorter the game higher will be the score
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foodPosition = currentGameState.getFood().asList()
    foodDistance = []

    for food in foodPosition:
        distance = manhattanDistance(position, food)
        foodDistance.append(distance)

    if not foodDistance:
        foodDistance.append(0)

    return currentGameState.getScore() + -1 * min(foodDistance)

# Abbreviation
better = betterEvaluationFunction
