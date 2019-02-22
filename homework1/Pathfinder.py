'''
The Pathfinder class is responsible for finding a solution (i.e., a
sequence of actions) that takes the agent from the initial state to all
of the goals with optimal cost.

This task is done in the solve method, as parameterized
by a maze pathfinding problem, and is aided by the SearchTreeNode DS.
'''

from MazeProblem import MazeProblem
from SearchTreeNode import SearchTreeNode
import unittest
import itertools

def overallCost(node):
    return node.totalCost + node.heuristicCost


class minHeap():
    def __init__(self):
        self.heap = []

    def __getitem__(self, index):
        return self.heap[index]

    def pop(self):
        return self.heap.pop(0)

    def append(self, node):
        self.heap.append(node)
        self.heap.sort(key=overallCost)

    def empty(self):
        if len(self.heap) == 0:
            return True
        return False

    def erase(self):
        self.heap.clear()


def heuristic(currentState, goalState):
    return abs(currentState[0] - goalState[0]) + abs(currentState[1] - goalState[1])


def mergeTuple(first, second):
    result = [first]
    for tup in enumerate(second):
        result += [tup[1]]
    return result


def getSolution(node):
    soln = []
    maxCost = node.totalCost
    while node.parent is not None:
        soln.append(node.action)
        maxCost += node.parent.totalCost
        node = node.parent
    soln.reverse()
    return (soln, maxCost)


def aStar (problem, initial, goal):
    root = SearchTreeNode(initial, None, None, 0, heuristic(initial, goal))
    frontier = minHeap()
    frontier.append(root)
    graveyard = set()

    while not frontier.empty():
        node = frontier.pop()
        graveyard.add(node.state)
        if node.state == goal:
            return getSolution(node)
        possibleMoves = problem.transitions(node.state)
        for moves in possibleMoves:
            if (moves[2] not in graveyard):
                frontier.append(SearchTreeNode(moves[2], moves[0], node, moves[1], heuristic(moves[2], goal)))


def solveTour(problem, initial, goals):

    solution = []
    cost = 0
    search = mergeTuple(initial, goals)

    for val, destination in enumerate(search[:-1]):
         result = aStar(problem, search[val], search[val+1])
         if (result == None):
             return None
         solution += result[0]
         cost += result[1]
    return (solution, cost)


def solve (problem, initial, goals):

    tours = list(itertools.permutations(goals))

    finalSolution = []
    finalCost = 0

    for tour in enumerate(tours):
        journey = solveTour(problem, initial, tour[1])
        if journey == None:
            return None
        if journey[1] <= finalCost or finalCost == 0:
            finalCost = journey[1]
            finalSolution = journey[0]
    return finalSolution


class PathfinderTests(unittest.TestCase):

    def test_maze1(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.M.X",
                "X.X.X.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (1, 3)
        goals   = [(5, 3)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 8)

    def test_maze2(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.M.X",
                "X.X.X.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (1, 3)
        goals   = [(3, 3),(5, 3)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 12)

    def test_maze3(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.MMX",
                "X...M.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (5, 1)
        goals   = [(5, 3), (1, 3), (1, 1)]
        soln = solve(problem, initial, goals)
        (soln_cost, is_soln) = problem.soln_test(soln, initial, goals)
        self.assertTrue(is_soln)
        self.assertEqual(soln_cost, 12)

    def test_maze4(self):
        maze = ["XXXXXXX",
                "X.....X",
                "X.M.XXX",
                "X...X.X",
                "XXXXXXX"]
        problem = MazeProblem(maze)
        initial = (5, 1)
        goals   = [(5, 3), (1, 3), (1, 1)]
        soln = solve(problem, initial, goals)
        self.assertTrue(soln == None)


if __name__ == '__main__':
    unittest.main()
