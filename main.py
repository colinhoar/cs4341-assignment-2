"""Student implementations for CS 4341 Assignment 2."""
from __future__ import annotations

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


def heuristic(state: StateT, player: PlayerT) -> float:
    return 0.0

def heuristic(
    problem: AdversarialSearchProblem[StateT, ActionT, PlayerT],
    state: StateT,
    maximizing_player: PlayerT,
) -> float:

    try:
        from .cylindrical_connect_four import WINDOWS, RED, YELLOW
    except ImportError:
        from cylindrical_connect_four import WINDOWS, RED, YELLOW

    if maximizing_player == RED:
        opponent = YELLOW
    else:
        opponent = RED

    score = 0.0

    player_threats = _count_winning_threats(
        state,
        maximizing_player,
        WINDOWS,
    )
    opponent_threats = _count_winning_threats(
        state,
        opponent,
        WINDOWS,
    )

    for window in WINDOWS:
        player_count = 0
        opponent_count = 0
        empty_cells = []
        for column, row in window:
            cell = state.cell(column, row)
            if cell == maximizing_player:
                player_count += 1
            elif cell == opponent:
                opponent_count += 1
            else:
                empty_cells.append((column, row))
        if player_count > 0 and opponent_count > 0:
            continue
        if player_count == 4:
            score += 1000
        elif player_count == 3:
            if _is_playable(state, empty_cells[0]):
                score += 100
            else:
                score += 25
        elif player_count == 2:
            score += 10
        elif player_count == 1:
            score += 1
        elif opponent_count == 4:
            score -= 1000
        elif opponent_count == 3:
            if _is_playable(state, empty_cells[0]):
                score -= 100
            else:
                score -= 25
        elif opponent_count == 2:
            score -= 10
        elif opponent_count == 1:
            score -= 1
    if player_threats >= 2:
        score += 250
    if opponent_threats >= 2:
        score -= 250

    return score


def _is_playable(
    state: StateT,
    position: tuple[int, int],
) -> bool:

    column, row = position

    if row == 0:
        return True

    return state.cell(column, row - 1) is not None


def _count_winning_threats(
    state: StateT,
    player: PlayerT,
    windows,
) -> int:

    threats = set()

    if player == "red":
        opponent = "yellow"
    else:
        opponent = "red"

    for window in windows:
        player_count = 0
        opponent_count = 0
        empty_cells = []
        for column, row in window:
            cell = state.cell(column, row)
            if cell == player:
                player_count += 1
            elif cell == opponent:
                opponent_count += 1
            else:
                empty_cells.append((column, row))

        if player_count == 3 and opponent_count == 0 and len(empty_cells) == 1 and _is_playable(state, empty_cells[0]):
            threats.add(empty_cells[0])

    return len(threats)

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

    # Return none when the state is a terminal positions
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

    def minimax(current_state: StateT, depth: int) -> float:
        """Return the minimax value of current_state for max_player."""

        # Use exact outcome if game is over
        if problem.is_terminal(current_state):
            return problem.utility(current_state, max_player)

        # Estimate position if we reach search horizon
        if depth == 0:
            return heuristic(current_state, max_player)

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
