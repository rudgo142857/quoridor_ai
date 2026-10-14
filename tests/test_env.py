"""
Phase 0 · Step A — 이동/점프/렌더링 테스트
"""
import pytest
from quoridor.env import (
    QuoridorState, QuoridorEnv,
    BOARD_SIZE, NUM_WALLS, ACTION_MOVE,
    MOVE_DELTA, JUMP_DELTA, DIAG_DELTA,
)


# ════════════════════════════════════════════════════════════
# 1. 게임 상태 초기화
# ════════════════════════════════════════════════════════════
class TestQuoridorState:

    def test_initial_positions_9x9(self):
        """9×9 보드: P0 맨 아래 중앙, P1 맨 위 중앙"""
        state = QuoridorState()
        assert state.pos[0] == (8, 4)
        assert state.pos[1] == (0, 4)

    def test_initial_positions_5x5(self):
        """5×5 보드: P0 (4,2), P1 (0,2)"""
        state = QuoridorState(board_size=5, num_walls=3)
        assert state.pos[0] == (4, 2)
        assert state.pos[1] == (0, 2)
        assert state.walls_remaining == [3, 3]

    def test_initial_walls(self):
        state = QuoridorState()
        assert state.walls_remaining == [10, 10]
        assert len(state.walls_placed) == 0

    def test_goal_rows(self):
        state = QuoridorState()
        assert state.goal_rows == [0, 8]

    def test_copy_independence(self):
        """copy()한 사본 변경이 원본에 영향 없어야 함"""
        state = QuoridorState()
        copied = state.copy()
        copied.pos[0] = (7, 4)
        copied.walls_placed.add((3, 3, 0))
        copied.walls_remaining[0] = 9
        assert state.pos[0] == (8, 4)
        assert len(state.walls_placed) == 0
        assert state.walls_remaining[0] == 10


# ════════════════════════════════════════════════════════════
# 2. 기본 이동
# ════════════════════════════════════════════════════════════
class TestBasicMoves:

    def test_p0_move_north(self):
        env = QuoridorEnv()
        env.reset()
        state, reward, done = env.step(0)       # P0: N
        assert state.pos[0] == (7, 4)
        assert reward == 0.0
        assert not done
        assert state.current_player == 1        # 턴 전환

    def test_p1_move_south(self):
        env = QuoridorEnv()
        env.reset()
        env.step(0)                              # P0 이동
        state, reward, done = env.step(1)        # P1: S
        assert state.pos[1] == (1, 4)
        assert state.current_player == 0

    def test_move_west(self):
        env = QuoridorEnv()
        env.reset()
        env.step(2)                              # P0: W
        assert env.state.pos[0] == (8, 3)

    def test_move_east(self):
        env = QuoridorEnv()
        env.reset()
        env.step(3)                              # P0: E
        assert env.state.pos[0] == (8, 5)

    def test_turn_alternation(self):
        env = QuoridorEnv()
        env.reset()
        assert env.state.current_player == 0
        env.step(0)
        assert env.state.current_player == 1
        env.step(1)
        assert env.state.current_player == 0


# ════════════════════════════════════════════════════════════
# 3. 합법 이동 판정
# ════════════════════════════════════════════════════════════
class TestLegalMoves:

    def test_corner_has_two_moves(self):
        """모서리 칸: 2방향만 합법"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (0, 0)
        env.state.pos[1] = (8, 8)               # 멀리 이동
        legal = env._get_legal_moves()
        assert 0 not in legal                    # N 불가 (밖)
        assert 2 not in legal                    # W 불가 (밖)
        assert 1 in legal                        # S 가능
        assert 3 in legal                        # E 가능
        assert len([a for a in legal if a < 4]) == 2

    def test_edge_has_three_moves(self):
        """모서리가 아닌 보드 가장자리: 3방향"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (0, 4)
        env.state.pos[1] = (8, 8)
        legal = env._get_legal_moves()
        basic = [a for a in legal if a < 4]
        assert 0 not in basic                    # N 불가
        assert len(basic) == 3

    def test_opponent_blocks_walk(self):
        """상대가 인접 → 그 방향 기본 이동 불가"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (4, 4)
        legal = env._get_legal_moves()
        assert 0 not in legal                    # N 직접 이동 불가

    def test_hwall_blocks_vertical_move(self):
        """수평 벽이 수직 이동을 차단"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (0, 4)               # 방해 안 되게
        env.state.walls_placed.add((4, 4, 0))    # row 4↔5 사이 벽
        legal = env._get_legal_moves()
        assert 0 not in legal                    # N 불가
        assert 1 in legal                        # S 가능

    def test_hwall_adjacent_col_blocks(self):
        """수평 벽은 2칸 걸치므로, 왼쪽 열 벽도 차단"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (0, 0)
        env.state.walls_placed.add((4, 3, 0))    # col 3~4 걸침, row 4↔5
        legal = env._get_legal_moves()
        assert 0 not in legal                    # N 불가

    def test_vwall_blocks_horizontal_move(self):
        """수직 벽이 수평 이동을 차단"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (0, 0)
        env.state.walls_placed.add((5, 4, 1))    # col 4↔5 사이 벽
        legal = env._get_legal_moves()
        assert 3 not in legal                    # E 불가
        assert 2 in legal                        # W 가능


# ════════════════════════════════════════════════════════════
# 4. 점프 로직
# ════════════════════════════════════════════════════════════
class TestJumps:

    def test_straight_jump_north(self):
        """직선 점프: 상대 뒤가 비어있으면 뛰어넘기"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (4, 4)
        legal = env._get_legal_moves()
        assert 4 in legal                        # Jump N
        env.step(4)
        assert env.state.pos[0] == (3, 4)

    def test_straight_jump_east(self):
        """수평 직선 점프"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (4, 3)
        env.state.pos[1] = (4, 4)
        legal = env._get_legal_moves()
        assert 7 in legal                        # Jump E
        env.step(7)
        assert env.state.pos[0] == (4, 5)

    def test_wall_blocks_straight_jump_allows_diag(self):
        """직선 점프 불가(벽) → 대각 점프 가능"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (4, 4)
        env.state.walls_placed.add((3, 4, 0))    # 상대 뒤에 벽
        legal = env._get_legal_moves()
        assert 4 not in legal                    # 직선 Jump N 불가
        assert 8 in legal                        # NW 대각 가능
        assert 9 in legal                        # NE 대각 가능

    def test_edge_blocks_straight_jump_allows_diag(self):
        """상대가 보드 끝 → 직선 점프 불가 → 대각"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (1, 4)
        env.state.pos[1] = (0, 4)
        legal = env._get_legal_moves()
        assert 4 not in legal                    # (-1,4)는 보드 밖
        assert 8 in legal                        # NW
        assert 9 in legal                        # NE

    def test_diagonal_jump_execution(self):
        """대각 점프 실행 후 위치 확인"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (1, 4)
        env.state.pos[1] = (0, 4)
        env.step(9)                              # NE
        assert env.state.pos[0] == (0, 5)

    def test_wall_between_blocks_all_jumps(self):
        """나↔상대 사이에 벽 → 직선·대각 모든 점프 불가"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (4, 4)
        env.state.walls_placed.add((4, 4, 0))    # 나(5,4)↔상대(4,4) 사이
        legal = env._get_legal_moves()
        for a in range(4, 12):
            assert a not in legal, f"액션 {a} 불가해야 함"

    def test_diag_blocked_by_side_wall(self):
        """대각 점프 방향에도 벽이 있으면 불가"""
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (5, 4)
        env.state.pos[1] = (4, 4)
        env.state.walls_placed.add((3, 4, 0))    # 직선 점프 차단
        env.state.walls_placed.add((4, 4, 1))    # 상대 우측 차단 (NE 불가)
        legal = env._get_legal_moves()
        assert 9 not in legal                    # NE 불가
        assert 8 in legal                        # NW는 가능


# ════════════════════════════════════════════════════════════
# 5. 승리 조건
# ════════════════════════════════════════════════════════════
class TestWinCondition:

    def test_p0_wins_at_row0(self):
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (1, 4)
        env.state.pos[1] = (8, 0)
        _, reward, done = env.step(0)            # N → (0,4)
        assert done is True
        assert reward == 1.0
        assert env.state.pos[0] == (0, 4)

    def test_p1_wins_at_row8(self):
        env = QuoridorEnv()
        env.reset()
        env.step(0)                              # P0 턴 소비
        env.state.pos[1] = (7, 4)
        env.state.pos[0] = (0, 0)
        _, reward, done = env.step(1)            # P1: S → (8,4)
        assert done is True
        assert reward == 1.0

    def test_no_turn_switch_on_win(self):
        env = QuoridorEnv()
        env.reset()
        env.state.pos[0] = (1, 4)
        env.state.pos[1] = (8, 0)
        state, _, done = env.step(0)
        assert done
        assert state.current_player == 0         # 턴 유지


# ════════════════════════════════════════════════════════════
# 6. 렌더링
# ════════════════════════════════════════════════════════════
class TestRender:

    def test_render_shows_players(self):
        env = QuoridorEnv()
        env.reset()
        output = env.render()
        assert "①" in output
        assert "②" in output

    def test_render_shows_wall(self):
        env = QuoridorEnv()
        env.reset()
        env.state.walls_placed.add((4, 4, 0))    # 수평 벽
        output = env.render()
        assert ("═" in output or "━" in output)

    def test_render_5x5(self):
        """5×5 보드도 정상 렌더링"""
        env = QuoridorEnv(board_size=5, num_walls=3)
        env.reset()
        output = env.render()
        assert "①" in output
        assert "②" in output
        board_lines = [l for l in output.strip().split("\n") if l.strip() and "남은 벽" not in l]
        assert len(board_lines) == 10            # 5셀 + 4구분 + 1헤더


class TestEncodeState:
    def test_encode_state_shape(self):
        env = QuoridorEnv()
        env.reset()
        s = env.encode_state()
        assert s.shape == (6, 9, 9)

    def test_encode_state_p0(self):
        env = QuoridorEnv()
        env.reset()
        s = env.encode_state()
        assert s[0, 8, 4] == 1.0
        assert s[1, 0, 4] == 1.0

    def test_encode_state_p1_flip(self):
        env = QuoridorEnv()
        env.reset()
        env.state.current_player = 1
        s = env.encode_state()
        # P1 관점에서 회전: (0, 4) -> (8, 4)
        assert s[0, 8, 4] == 1.0
        # P0 관점에서 회전: (8, 4) -> (0, 4)
        assert s[1, 0, 4] == 1.0
