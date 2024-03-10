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

        "*** YOUR CODE HERE ***"
        # Constants so we can influence Pacman's behavior
        FOOD_WEIGHT = 10.0
        GHOST_WEIGHT = -20.0
        SCARED_BONUS = 100.0

        # Calculate the distance to the closest food
        foodList = newFood.asList()
        if len(foodList) > 0:  # I.e., if there is food left
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]  # Compute distance to food
            minFoodDistance = min(foodDistances)  # What is our closest food?
        else:
            minFoodDistance = 0.0  # no food >:(

        # Need ghost positions
        ghostStates = []
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if distance == 0:
                distance = 0.1  # So we don't ever divide by 0
            if ghostState.scaredTimer == 0:  # If ghosts are not scared, we need to avoid them
                ghostStates.append((1.0 / distance, GHOST_WEIGHT))
            else:
                # We should weight states where ghosts are scared more highly so that we can remove them
                ghostStates.append((1.0 / distance, SCARED_BONUS))

        # Use the reciprocal so that we can get Pacman closer to food
        if minFoodDistance > 0:
            reciprocalFoodDistance = 1.0 / minFoodDistance
        else:
            reciprocalFoodDistance = 0.0

        # Now, determine overall ghost impact of the move
        ghostImpact = sum(distance * weight for (distance, weight) in ghostStates)

        # All the components sum to our final score
        return successorGameState.getScore() + (reciprocalFoodDistance * FOOD_WEIGHT) + ghostImpact


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        def minimax(state, depth, agent):
            """
            This should return the best action for the agent
            """
            # Go down if we're Mr. Pacman
            nextDepth = depth
            if agent == 0:
                nextDepth = depth - 1

            if nextDepth == 0 or state.isWin() or state.isLose():
                # I.e., we're done, so return
                return self.evaluationFunction(state), None

            # Defining best cases for maximizer and minimizer
            if agent == 0:
                optimal_val = -99999
            else:
                optimal_val = 99999

            # Pivot agents (pacman -> ghost, ghost1 -> ghost2, etc.), then wrap
            nextAgent = (agent + 1) % state.getNumAgents()

            # Store optimal action for next agent
            bestAction = None

            # Placeholder
            bestValueSoFar = optimal_val

            # Loop to consider all possible actions
            for mm_action in state.getLegalActions(agent):
                # Determine what the gameState will be after a given action
                successorState = state.generateSuccessor(agent, mm_action)
                # Recursive call to find value of that gameState
                valueOfTheCurrentAction, _ = minimax(successorState, nextDepth, nextAgent)

                # If the current agent is a maximizer, we want to update our best value accordingly
                if agent == 0:  # Mr. Pacman
                    if valueOfTheCurrentAction > bestValueSoFar:
                        bestValueSoFar, bestAction = valueOfTheCurrentAction, mm_action
                # If the current agent is a minimizer, we want to update our best value accordingly
                else:
                    if valueOfTheCurrentAction < bestValueSoFar:
                        bestValueSoFar, bestAction = valueOfTheCurrentAction, mm_action
            # After checking all actions, return best we found
            return bestValueSoFar, bestAction

        # For the top-level call, get the best action for Pacman without needing its value.
        topLvL_val, optimalAction = minimax(gameState, self.depth + 1, self.index)

        # Return this best action.
        return optimalAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    """
    To implement, apply a-b pruning to minimax search tree
    Prune the subtree when alpha (min.) exceeds or is equal to beta (max.)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def alphaBetaPruning(state, depth, agent, alpha, beta):
            """ Should return best action for agent from a pruned tree. """
            # Go down if we are Mr. Pacman
            nextDepth = depth
            if agent == 0:
                nextDepth = depth - 1

            if nextDepth == 0 or state.isWin() or state.isLose():
                # I.e., we're done, so return
                return self.evaluationFunction(state), None

            # Set cases for agent types
            if agent == 0:  # Pacman
                optimal_val = -99999
            else:  # Ghosts
                optimal_val = 99999

            # Use the same wrap-around
            nextAgent = (agent + 1) % state.getNumAgents()

            # Store optimal action for next agent
            bestAction = None

            # Placeholder
            bestValueSoFar = optimal_val

            # Same loop as minimax for all possible actions
            for aB_action in state.getLegalActions(agent):
                # Determine what gameState will be given an action
                successorState = state.generateSuccessor(agent, aB_action)
                # Recursive call to find value of gS
                valueOfTheCurrentAction, _ = alphaBetaPruning(successorState, nextDepth, nextAgent, alpha, beta)

                # if agent is maximizer, we need to update best value accordingly
                if agent == 0:
                    if valueOfTheCurrentAction > bestValueSoFar:
                        bestValueSoFar, bestAction = valueOfTheCurrentAction, aB_action
                    if bestValueSoFar > beta:  # If our best value is greater than beta, we prune
                        return bestValueSoFar, bestAction
                    alpha = max(alpha, bestValueSoFar)  # Else, update alpha
                # if agent is minimizer, need to update value accordingly
                else:  # minimizer
                    if valueOfTheCurrentAction < bestValueSoFar:  # choose whatever is less since we're minimizing
                        bestValueSoFar, bestAction = valueOfTheCurrentAction, aB_action
                    if bestValueSoFar < alpha:  # If alpha > bestValue, return
                        return bestValueSoFar, bestAction
                    beta = min(beta, bestValueSoFar)  # Else, lower beta
            return bestValueSoFar, bestAction
        _, aB_action = alphaBetaPruning(gameState, self.depth + 1, self.index, -99999, 99999)
        return aB_action


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

        def e_max(state, depth, agent):
            """ Returns the best action based on a non-optimal adversary. """
            # Go down if we're Mr. Pacman
            nextDepth = depth
            if agent == 0:
                nextDepth = depth - 1

            # What can we do?
            legalActions = state.getLegalActions(agent)

            if nextDepth == 0 or state.isWin() or state.isLose():
                # I.e, we're done, so return
                return self.evaluationFunction(state), None

            # Use the same wrap-around
            nextAgent = (agent + 1) % state.getNumAgents()

            if agent == 0:  # Pacman
                optimal_val = -99999  # Don't need this at the top, ghosts don't get a val
                bestAction = None  # Ghosts also have no best action
                for em_action in legalActions:
                    # What gS will this action result in?
                    successorState = state.generateSuccessor(agent, em_action)
                    # Find value of that gS
                    valueOfTheCurrentAction, _ = e_max(successorState, nextDepth, nextAgent)
                    # If that value is better than what we have seen so far
                    if valueOfTheCurrentAction > optimal_val:
                        # ... update it
                        optimal_val, bestAction = valueOfTheCurrentAction, em_action
                return optimal_val, bestAction

            else:  # Ghosts
                # This is what we use to randomly generate state selection
                ex_value = 0  # Init to zero
                for em_action in legalActions:
                    # What gs will we be in
                    successorState = state.generateSuccessor(agent, em_action)
                    # What is the exp value of that action
                    valueOfTheCurrentAction, _ = e_max(successorState, nextDepth, nextAgent)
                    # Change it to the current, we aren't looking for optimism
                    ex_value += valueOfTheCurrentAction
                avgValue = ex_value / len(legalActions)
                # On average how well do the ghosts do?
                return avgValue, None

        _, em_action = e_max(gameState, self.depth + 1, self.index)
        return em_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Take into account score, distance to food and ghost, the state of the ghosts,
    and the # of remaining capsules.
    """
    "*** YOUR CODE HERE ***"
    # Where is everything?
    pac_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    num_capsules = currentGameState.getCapsules()

    # Weights to modify behavior
    food_weight = -1.0
    ghost_weight = 2.0
    scared_weight = -2.0
    capsule_weight = -20.0

    # Capsule consideration
    # Large, negative number to penalize capsule loss
    capsuleEffect = len(num_capsules) * capsule_weight

    # Ghost consideration
    ghostEffect = 0
    for ghostState in ghostStates:
        distance = manhattanDistance(pac_pos, ghostState.getPosition())  # How far are we from a threat?
        if distance == 0:
            distance = 0.1  # again, avoid div/0
        if ghostState.scaredTimer > 0:
            ghostEffect += scared_weight / distance  # go after scared ghosts
        else:
            ghostEffect += ghost_weight / distance  # otherwise, award Mr. Pac for staying away

    # Food consideration
    foodList = food.asList()
    if len(foodList) > 0:
        foodDistances = [manhattanDistance(pac_pos, foodPoint) for foodPoint in foodList]  # Where da food
        nearestFood = min(foodDistances)  # where da closest food?
    else:
        nearestFood = 0  # no food :'(

    # Sum of states
    return currentGameState.getScore() + food_weight * nearestFood + ghostEffect + capsuleEffect


# Abbreviation
better = betterEvaluationFunction
