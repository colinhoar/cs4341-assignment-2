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
    raise NotImplementedError

def heuristic(
    problem: AdversarialSearchProblem[StateT, ActionT, PlayerT],
    state: StateT,
    maximizing_player: PlayerT,
    minimizing_player: PlayerT
) -> float:
    from cylindrical_connect_four import all as ccf
    if ccf.is_terminal(state):
        return problem.utility(state, maximizing_player)
    score = 0

    for window in WINDOWS(state):
        window_score = 0
        pcount = 0
        ocount = 0

        for cell in window:
            if cell == maximizing_player:
                pcount+=1
            elif cell == minimizing_player:
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