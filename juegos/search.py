"""Implementation of search functions for reversi.

    Authors:
        Alejandro Bellogin <alejandro.bellogin@uam.es>
"""

import search
import random

from timeit import default_timer as timer
from datetime import timedelta

import util
from reversi import (
    get_valid_moves, 
    enemy_captured_by_move,
    create_standard_board,
)


class CornerReversiState:
    """
    This class defines the mechanics of a Reversi game.
    The task of recasting this game state as a search problem is left to
    the CornerReversiSearchProblem class.
    """

    def __init__(self, board: dict, player1='B', player2='W', cur_player='B', height=8, width=8, ignore_block_cells_in_captures: bool = False):
        "Creates a new CornerReversiState, very similar to Reversi."
        self.board = board
        self.player1 = player1
        self.player2 = player2
        self.blocked_cell_label = 'O'
        self.cur_player = cur_player
        self.height = height
        self.width = width
        self.ignore_block_cells_in_captures = ignore_block_cells_in_captures

    def isGoal(self, min_corners=1):
        """
          Checks to see if any of the players have conquered min_corners in the board.
        """
        corners = [self.board.get((1, 1)), self.board.get((1, self.height)),
                   self.board.get((self.width, 1)), self.board.get((self.width, self.height))]
        return corners.count(self.player1) + corners.count(self.player2) >= min_corners

    def legalMoves(self):
        """
          Returns a list of legal moves from the current state.
        """
        next_player = self.player2 if self.cur_player == self.player1 else self.player1
        return get_valid_moves(
            self.board, 
            self.height, 
            self.width, 
            self.cur_player, 
            next_player, 
            self.blocked_cell_label, 
            self.ignore_block_cells_in_captures
        )

    def result(self, move):
        """
          Returns a new board with the current state updated based on the provided move.
        """
        result_board = self.board.copy()
        adversary = self.player2 if self.cur_player == self.player1 else self.player1
        result_board[move] = self.cur_player
        for enemy in enemy_captured_by_move(self.board, move, self.cur_player, adversary, self.blocked_cell_label, self.ignore_block_cells_in_captures):
            result_board[enemy] = self.cur_player

        return CornerReversiState(
            board=result_board,
            player1=self.player1,
            player2=self.player2,
            cur_player=adversary,
            height=self.height,
            width=self.width,
            ignore_block_cells_in_captures=self.ignore_block_cells_in_captures
        )

    def __eq__(self, other):
        if not isinstance(other, CornerReversiState):
            return False
        return self.board == other.board and self.cur_player == other.cur_player

    def __hash__(self):
        return hash(self.__getAsciiString())

    def __getAsciiString(self):
        adversary = self.player2 if self.cur_player == self.player1 else self.player1
        moves = get_valid_moves(self.board, self.height, self.width, self.cur_player, adversary, self.blocked_cell_label, self.ignore_block_cells_in_captures)
        lines = []
        for y in range(0, self.height + 1):
            rowLine = ''
            for x in range(0, self.width + 1):
                if x > 0 and y > 0:
                    if (x, y) in moves:
                        rowLine = rowLine + self.board.get((x, y), '_',)
                    else:
                        rowLine = rowLine + self.board.get((x, y), '.',)
                if x == 0:
                    if y > 0:
                        rowLine = rowLine + str(y) + ' '
                if y == 0:
                    rowLine = rowLine + (chr(x+96) if x > 0 else '  ')
            lines.append(rowLine)
        return '\n'.join(lines)

    def __str__(self):
        return self.__getAsciiString()


class SearchProblem:
    def getStartState(self):
        raise NotImplementedError

    def isGoalState(self, state):
        raise NotImplementedError

    def getSuccessors(self, state):
        raise NotImplementedError


class CornerReversiSearchProblem(SearchProblem):
    def __init__(self, reversi_state: CornerReversiState):
        self.state = reversi_state

    def getStartState(self):
        return self.state

    def isGoalState(self, state):
        return state.isGoal(1)

    def getSuccessors(self, state):
        succ = []
        for a in state.legalMoves():
            succ.append((state.result(a), a))
        return succ


class AllCornersReversiSearchProblem(CornerReversiSearchProblem):
    def isGoalState(self, state):
        return state.isGoal(4)


def build_game_tree(search_problem, max_depth):
    """
    Creates a game tree from a search problem until max_depth.
    """
    stats = {
        "nodes": 0,
        "leaves": 0,
        "max_depth": 0,
        "branching_sum": 0,
        "internal_nodes": 0,
    }

    class TreeNode:
        def __init__(self, state, depth):
            self.state = state
            self.depth = depth
            self.children = []

    def recursive_build(current_state, current_depth):
        stats["nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], current_depth)
        node = TreeNode(current_state, current_depth)

        if current_depth == max_depth or search_problem.isGoalState(current_state):
            stats["leaves"] += 1
            return node

        successors = search_problem.getSuccessors(current_state)
        if not successors:
            stats["leaves"] += 1
            return node

        stats["internal_nodes"] += 1
        stats["branching_sum"] += len(successors)

        for succ_state, _ in successors:
            child_node = recursive_build(succ_state, current_depth + 1)
            node.children.append(child_node)

        return node

    root = recursive_build(search_problem.getStartState(), 0)
    
    if stats["internal_nodes"] > 0:
        stats["avg_branching_factor"] = stats["branching_sum"] / stats["internal_nodes"]
    else:
        stats["avg_branching_factor"] = 0.0

    return root, stats


def depthFirstSearch(search_problem):
    """
    Búsqueda en profundidad (BP / DFS) como búsqueda en grafo.
    """
    num_visited = 0
    structure = util.Stack()
    start_state = search_problem.getStartState()
    structure.push((start_state, []))
    visited = set()

    while not structure.isEmpty():
        current_state, path = structure.pop()

        if search_problem.isGoalState(current_state):
            return num_visited, path

        if current_state not in visited:
            visited.add(current_state)
            num_visited += 1

            for succ_state, action in search_problem.getSuccessors(current_state):
                if succ_state not in visited:
                    structure.push((succ_state, path + [action]))

    return num_visited, None


def breadthFirstSearch(search_problem):
    """
    Búsqueda en anchura (BA / BFS) como búsqueda en grafo.
    """
    num_visited = 0
    structure = util.Queue()
    start_state = search_problem.getStartState()
    structure.push((start_state, []))
    visited = set()
    visited.add(start_state)

    while not structure.isEmpty():
        current_state, path = structure.pop()
        num_visited += 1

        if search_problem.isGoalState(current_state):
            return num_visited, path

        for succ_state, action in search_problem.getSuccessors(current_state):
            if succ_state not in visited:
                visited.add(succ_state)
                structure.push((succ_state, path + [action]))

    return num_visited, None


def nullHeuristic(state, search_problem=None):
    return 0


def simpleHeuristic(state, search_problem=None):
    return len(state.board)


def heuristic1(state, search_problem=None):
    corners = [(1, 1), (1, state.height), (state.width, 1), (state.width, state.height)]
    free_corners = [c for c in corners if c not in state.board]
    
    if not free_corners:
        return 0

    my_positions = [pos for pos, player in state.board.items() if player == state.cur_player]
    if not my_positions:
        return 8

    min_dist = float('inf')
    for pos in my_positions:
        for corner in free_corners:
            dist = max(abs(pos[0] - corner[0]), abs(pos[1] - corner[1]))
            if dist < min_dist:
                min_dist = dist

    return min_dist


def heuristic2(state, search_problem=None):
    h_val = heuristic1(state, search_problem)
    
    corners = [(1, 1), (1, state.height), (state.width, 1), (state.width, state.height)]
    bad_cells = []
    for cx, cy in corners:
        if (cx, cy) not in state.board:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if 1 <= nx <= state.width and 1 <= ny <= state.height and (nx, ny) != (cx, cy):
                        bad_cells.append((nx, ny))

    for cell in bad_cells:
        if state.board.get(cell) == state.cur_player:
            h_val += 3

    return h_val


def heuristic3(state, search_problem=None):
    corners = [(1, 1), (1, state.height), (state.width, 1), (state.width, state.height)]
    legal_moves = state.legalMoves()
    
    for c in corners:
        if c in legal_moves:
            return 1

    h_val = heuristic2(state, search_problem)
    h_val -= 0.2 * len(legal_moves)
    return max(1.0, float(h_val))


def aStarSearch(search_problem, heuristic=nullHeuristic):
    """
    Búsqueda A* con util.PriorityQueue.
    """
    num_visited = 0
    frontier = util.PriorityQueue()
    start_state = search_problem.getStartState()
    frontier.push((start_state, [], 0), 0 + heuristic(start_state, search_problem))
    best_cost = {start_state: 0}

    while not frontier.isEmpty():
        current_state, path, g_cost = frontier.pop()

        if search_problem.isGoalState(current_state):
            return num_visited, path

        if g_cost > best_cost.get(current_state, float('inf')):
            continue

        num_visited += 1

        for succ_state, action in search_problem.getSuccessors(current_state):
            new_g = g_cost + 1
            if succ_state not in best_cost or new_g < best_cost[succ_state]:
                best_cost[succ_state] = new_g
                new_path = path + [action]
                f_cost = new_g + heuristic(succ_state, search_problem)
                frontier.push((succ_state, new_path, new_g), f_cost)

    return num_visited, None


def createRandomReversiGeneralState(moves, h, w):
    puzzle = CornerReversiState(create_standard_board(h, w, 'B', 'W'), height=h, width=w)
    for _ in range(moves):
        moves_list = puzzle.legalMoves()
        if not moves_list:
            break
        puzzle = puzzle.result(random.sample(moves_list, 1)[0])
    return puzzle


def createSmallRandomReversiState(moves=10):
    return createRandomReversiGeneralState(moves, 6, 6)


def createRandomReversiState(moves=100):
    return createRandomReversiGeneralState(moves, 8, 8)


if __name__ == '__main__':
    iterations_initial_state = 0
    create_small_board = True
    print_steps = False

    if create_small_board:
        reversi_state = createSmallRandomReversiState(iterations_initial_state)
    else:
        reversi_state = createRandomReversiState(iterations_initial_state)

    print('Initial state after %d iterations:' % (iterations_initial_state))
    print(reversi_state)

    problem = CornerReversiSearchProblem(reversi_state)
    print('Problem to be solved: %s' % (problem.__class__))

    for depth in [1, 2, 3, 4]:
        root, stats = build_game_tree(problem, depth)
        print('Depth', depth, '\t', stats)

    for method in ['DFS', 'BFS']:
        start = timer()
        if method == 'DFS':
            visited, path = depthFirstSearch(problem)
        elif method == 'BFS':
            visited, path = breadthFirstSearch(problem)
        end = timer()

        if path is not None:
            print_path = [(chr(x+96), y) for x, y in path]
            print('%s found a path of %d moves visiting %d nodes in %s: %s' %
                  (method, len(path), visited, str(timedelta(seconds=end - start)), str(print_path)))
        else:
            print('%s found no path in %s' % (method, str(timedelta(seconds=end - start))))

    for h in [simpleHeuristic, heuristic1, heuristic2, heuristic3]:
        method = 'A* with ' + h.__name__
        start = timer()
        visited, path = aStarSearch(problem, h)
        end = timer()

        if path is not None:
            print_path = [(chr(x+96), y) for x, y in path]
            print('%s found a path of %d moves visiting %d nodes in %s: %s' %
                    (method, len(path), visited, str(timedelta(seconds=end - start)), str(print_path)))
        else:
            print('%s found no path in %s' % (method, str(timedelta(seconds=end - start))))