import pytest
from quoridor.env import QuoridorEnv

def test_action_to_wall():
    env = QuoridorEnv()
    assert env._action_to_wall(12) == (0, 0, 0)
    assert env._action_to_wall(12 + 8 + 1) == (1, 1, 0) # 21
    assert env._action_to_wall(12 + 64) == (0, 0, 1) # 76
    assert env._action_to_wall(139) == (7, 7, 1)

def test_wall_overlap_horizontal():
    env = QuoridorEnv()
    env.reset()
    env.state.walls_placed.add((4, 4, 0))
    assert not env._is_valid_wall(4, 4, 0) # Exact match
    assert not env._is_valid_wall(4, 3, 0) # Overlap left
    assert not env._is_valid_wall(4, 5, 0) # Overlap right
    assert env._is_valid_wall(4, 2, 0) # Valid

def test_wall_cross():
    env = QuoridorEnv()
    env.reset()
    env.state.walls_placed.add((4, 4, 0))
    assert not env._is_valid_wall(4, 4, 1) # Cross
    assert env._is_valid_wall(4, 3, 1) # Valid

def test_has_path():
    env = QuoridorEnv()
    env.reset()
    assert env._has_path(0)
    assert env._has_path(1)
    
    # Block P0 horizontally entirely
    for c in range(8):
        env.state.walls_placed.add((7, c, 0))
        
    assert not env._has_path(0) # P0 blocked
    assert not env._has_path(1) # P1 blocked
