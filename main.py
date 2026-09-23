"""Student implementations for CS 4341 Assignment 2."""
from __future__ import annotations

from cylindrical_connect_four import WINDOWS, CylindricalConnectFour

# Import utilities
try:
    from .adversarial_search import (
        ActionT,
        AdversarialSearchProblem,
        PlayerT,
        StateT,
    )
except ImportError:
    from adversarial_search import (
        ActionT,
        AdversarialSearchProblem,
        PlayerT,
        StateT,
    )


# Replace this with the name your group wants displayed in the tournament.
GROUP_NAME = "CAChE"


# def heuristic(state: StateT, player: PlayerT) -> float:
#     return 0.0

def heuristic(
    problem: AdversarialSearchProblem[StateT, ActionT, PlayerT],
    state: StateT,
    maximizing_player: PlayerT
) -> float:
    from cylindrical_connect_four import CylindricalConnectFour as CCF
    ccf = CCF()
    score = 0
    if ccf.is_terminal(state):
        if (problem.utility(state, maximizing_player)>1):
            score += 5000
        elif (problem.utility(state, maximizing_player)<1):
            score -= 5000
        return score
    
    for window in WINDOWS:
        window_score = 0
        pcount = 0
        ocount = 0

        for cell in window:
            if cell == maximizing_player:
                pcount+=1
            elif cell != maximizing_player and cell is not None:
                ocount+=1

        if pcount > 0 and ocount == 0:
            if pcount == 1:
                window_score += 1
            elif pcount == 2:
                window_score += 10
            elif pcount == 3:
                window_score += 100
        elif ocount > 0 and pcount == 0:
            if ocount == 1:
                window_score -= 1
            elif ocount == 2:
                window_score -= 10
            elif ocount == 3:
                window_score -= 100

        score += window_score
    return score

def adversarial_search(
    problem: AdversarialSearchProblem[StateT, ActionT, PlayerT],
    state: StateT,
) -> ActionT | None:
    """
    Choose an action for the current game state.

    How your agent makes this choice is up to you.  You may add any helpers
    you find useful to this file, such as a heuristic, quiescence search,
    rollout policy, move ordering, or a cache.  None of those helpers is a
    required part of the submission interface.

    The ``problem`` object provides the game rules through methods including
    ``actions``, ``result``, ``to_move``, ``is_terminal``, and ``utility``.
    Your agent is responsible for choosing and managing its own search limits.

    Args:
        problem:
            The adversarial search problem.

        state:
            The current game state.

    Returns:
        The selected action: a zero-based column number from
        ``problem.actions(state)``.
        None if state is terminal or has no legal actions.

    Performance:
        Every call must return in less than 10 seconds.  Choose an internal
        search budget with enough margin to satisfy that hard limit.
    """

    # Return none when the state is a terminal position
    # Terminal positions can't make any further moves
    if problem.is_terminal(state):
        return None

    legal_actions = tuple(problem.actions(state))

    # Return None when there are no legal actions
    if not legal_actions:
        return None

    # Evaluate each position from the perspective of the player making the decision at the root of the search
    max_player = problem.to_move(state)

    # Set initial search depth to 4 (can be adjusted with context of heursitic)
    depth_limit = 4

    # Return minimax value of current state for MAX (player)
    def minimax(current_state: StateT, depth: int) -> float:

        # Use exact outcome if game is over
        if problem.is_terminal(current_state):
            return problem.utility(current_state, max_player)

        # Estimate position if we reach search horizon
        if depth == 0:
            return heuristic(problem, current_state, max_player)

        actions = tuple(problem.actions(current_state))

        # MAX (player) chooses child with the largest value
        if problem.to_move(current_state) == max_player:
            value = float("-inf")

            for action in actions:
                child = problem.result(current_state, action)
                child_value = minimax(child, depth - 1)
                value = max(value, child_value)

            return value

        # MIN (agent) chooses child with the smallest value
        value = float("inf")

        for action in actions:
            child = problem.result(current_state, action)
            child_value = minimax(child, depth - 1)
            value = min(value, child_value)

        return value

    # Choose legal move with the greatest minimax value since root belongs to MAX
    best_action = legal_actions[0]
    best_value = float("-inf")

    for action in legal_actions:
        child = problem.result(state, action)
        value = minimax(child, depth_limit - 1)

        if value > best_value:
            best_value = value
            best_action = action

    return best_action
