import numpy as np
from typing import Tuple, List, Set
from collections import deque
import heapq  # 맨 위에 추가
# ════════════════════════════════════════════════════════════
# 상수 (Constants)
# ════════════════════════════════════════════════════════════
BOARD_SIZE   = 9
WALL_SIZE    = 8
NUM_WALLS    = 10

ACTION_MOVE  = 12
ACTION_WALL  = 128
ACTION_TOTAL = 140

# TODO: 행동 매핑 딕셔너리를 정의하세요.
MOVE_DELTA = {0: (-1, 0), 1: (+1, 0), 2: (0, -1), 3: (0, 1)}        # 0~3: 상,하,좌,우 (dr, dc)
JUMP_DELTA = {4: (-2, 0), 5: (2, 0), 6: (0, -2), 7: (0, 2)}        # 4~7: 직선 점프 (상대를 뛰어넘음)
DIAG_DELTA = {8: (-1, -1), 9: (-1, +1), 10: (+1, -1), 11: (+1, +1)}        # 8~11: 대각선 점프

JUMP_TO_MOVE = {4: 0, 5: 1, 6: 2, 7: 3}      # 직선 점프(4~7) -> 대응하는 기본 이동(0~3) 매핑
DIAG_COMPONENTS = {8: (0, 2), 9: (0, 3), 10: (1, 2), 11: (1, 3)}   # 대각 점프(8~11) -> 구성하는 두 기본 방향(0~3) 매핑

# ════════════════════════════════════════════════════════════
# 게임 상태 (Game State)
# ════════════════════════════════════════════════════════════
class QuoridorState:
    """
    좌표계: row 0 = 맨 위, row 8 = 맨 아래 (BOARD_SIZE=9 기준)
    P0: 아래에서 출발, 목표는 row=0
    P1: 위에서 출발, 목표는 row=8
    """

    def __init__(self, board_size: int = BOARD_SIZE, num_walls: int = NUM_WALLS):
        # TODO: 속성 초기화 (board_size, pos, walls_remaining, walls_placed, current_player, goal_rows)
        self.board_size = board_size
        self.pos = [(board_size-1, (board_size-1)//2),
                        (0, (board_size-1)//2)]
        self.walls_remaining = [num_walls, num_walls]
        self.walls_placed = set()
        self.current_player = 0
        self.goal_rows = [0, board_size-1]

        pass

    def copy(self) -> 'QuoridorState':
        new_state = QuoridorState(self.board_size)

        new_state.pos = list(self.pos)
        new_state.walls_remaining = list(self.walls_remaining)
        new_state.walls_placed = set(self.walls_placed)
        new_state.goal_rows = list(self.goal_rows)

        # 3. 단순 숫자는 그냥 복사해도 됨
        new_state.current_player = self.current_player

        # 4. 새 객체 반환
        return new_state
        pass

# ════════════════════════════════════════════════════════════
# 게임 환경 (Game Environment)
# ════════════════════════════════════════════════════════════
class QuoridorEnv:

    def __init__(self, board_size: int = BOARD_SIZE, num_walls: int = NUM_WALLS):
        self.board_size = board_size
        self.num_walls  = num_walls
        self.state: QuoridorState = None

    def reset(self) -> QuoridorState:
        # TODO: 초기화 후 상태 반환
        self.state = QuoridorState(self.board_size, self.num_walls)
        return self.state
        pass

    def _in_bounds(self, r: int, c: int) -> bool:
        # TODO: (r, c)가 보드 내부인지 반환
        if (0 <= r <= self.board_size-1) and (0 <= c <= self.board_size-1):
            return True
        return False
        pass

    def _is_blocked(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        walls = self.state.walls_placed
        dr = r2 - r1

        if dr != 0: # 위아래(수직) 이동 -> 수평 벽(0) 차단 확인
            eval_r = min(r1, r2)
            # 내 머리 위(eval_r)에 벽이 가로막고 있는지 확인.
            # 벽은 2칸을 차지하므로, 현재 열(c1)과 왼쪽 열(c1-1)에서 시작하는 벽 모두 확인
            if (eval_r, c1, 0) in walls or (eval_r, c1-1, 0) in walls:
                return True

        else: # 좌우(수평) 이동 -> 수직 벽(1) 차단 확인
            eval_c = min(c1, c2)
            # 내 옆(eval_c)에 수직 벽이 있는지 확인.
            # 현재 행(r1)과 위쪽 행(r1-1)에서 시작하는 벽 모두 확인
            if (r1, eval_c, 1) in walls or (r1-1, eval_c, 1) in walls:
                return True

        return False

        pass

    # ════════════════════════════════════════════════════════════
    # NEW STEP 0-B FUNCTIONS (벽 좌표 변환, 충돌 검사, BFS 검사)
    # ════════════════════════════════════════════════════════════
    def _action_to_wall(self, action: int) -> Tuple[int, int, int]:
        """12~139번 액션을 (r, c, orientation)으로 변환"""
        # TODO: action 변환 로직 (orientation 0은 가로, 1은 세로)
        action_idx = action - 12
        orientation = action_idx // 64
        r = (action_idx % 64) // 8
        c = (action_idx % 64) % 8
        return (r, c, orientation)

        pass

    def _is_valid_wall(self, r: int, c: int, orientation: int) -> bool:
        """벽이 보드 안에 있고 기존 벽과 겹치거나 교차하지 않는지 확인"""
        # TODO: 경계 검사, 겹침(overlap) 검사, 십자 교차(intersect) 검사
        if not (0<=r<=7) or not (0<=c<=7):
            return False

        if orientation ==0:
            if (r, c, orientation) in self.state.walls_placed or (r, c-1, orientation) in self.state.walls_placed or (r, c+1, orientation) in self.state.walls_placed:
                return False
        elif orientation == 1:
            if (r, c, orientation) in self.state.walls_placed or (r-1, c, orientation) in self.state.walls_placed or (r+1, c, orientation) in self.state.walls_placed:
                return False

        if  (r, c, 1-orientation) in self.state.walls_placed:
            return False

        return True
        pass

    # def _has_path(self, player: int) -> bool:
    #     """BFS로 해당 플레이어가 목표 행에 도달할 수 있는지 확인"""
    #     # TODO: deque를 사용한 너비 우선 탐색 구현
    #     queue = deque([self.state.pos[player]])
    #     visited = set([self.state.pos[player]])
    #     while queue:
    #         r, c = queue.popleft()
    #         if r==self.state.goal_rows[player]:
    #             return True
    #         for dr, dc in MOVE_DELTA.values():
    #             nr, nc = r+dr, c+dc
    #             if self._in_bounds(nr, nc) and not self._is_blocked(r,c,nr,nc) and not (nr, nc) in visited:
    #                 visited.add((nr, nc))
    #                 queue.append((nr, nc))
    #     return False
    #     pass


    def _has_path(self, player: int) -> bool:
        """Greedy Best-First Search로 더 빠르게 길 찾기"""
        start_r, start_c = self.state.pos[player]
        goal_row = self.state.goal_rows[player]

        # 우선순위 큐: (목표까지의 거리, 행, 열)
        # 거리가 짧은 것부터 먼저 팝(pop) 됩니다!
        queue = []
        heapq.heappush(queue, (abs(goal_row - start_r), start_r, start_c))
        visited = set([(start_r, start_c)])

        while queue:
            dist, r, c = heapq.heappop(queue)

            if r == goal_row:
                return True

            for dr, dc in MOVE_DELTA.values():
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and not self._is_blocked(r, c, nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    # 휴리스틱: 목표 행까지의 남은 수직 거리
                    h = abs(goal_row - nr)
                    heapq.heappush(queue, (h, nr, nc))

        return False

    # ════════════════════════════════════════════════════════════

    def _get_legal_moves(self) -> List[int]:
        """
        현재 플레이어의 합법 이동/점프 액션 리스트 반환.
        1. 인접한 칸으로 한 칸 이동
        2. 상대가 있을 경우 뛰어넘어 직선 점프
        3. 직선 점프가 막혔을 경우 상대를 기준으로 대각 점프
        """
        # TODO: 합법적 이동 액션 필터링 구현
        current_player = self.state.current_player
        opponent_player = 1-current_player
        pos_r, pos_c = self.state.pos[current_player]
        opr, opc = self.state.pos[opponent_player]
        legal_moves = []

        for action, (dr, dc) in MOVE_DELTA.items():
            new_r = pos_r + dr
            new_c = pos_c +dc
            if not self._in_bounds(new_r, new_c):
                continue
            if self._is_blocked(pos_r, pos_c, new_r, new_c):
                continue
            if (new_r, new_c) == (opr, opc):
                continue
            legal_moves.append(action)


        for action, (dr, dc) in JUMP_DELTA.items():

            base: int = JUMP_TO_MOVE[action]
            potent_dr, potent_dc = MOVE_DELTA[base]
            new_r = pos_r + dr
            new_c = pos_c +dc

            if (pos_r +potent_dr, pos_c+potent_dc) != (opr, opc):
                continue
            if self._is_blocked(pos_r, pos_c, opr, opc) :
                continue
            if self._in_bounds(new_r, new_c) and not self._is_blocked(pos_r +potent_dr, pos_c+potent_dc, new_r, new_c):
                legal_moves.append(action)
            else:
                for diag_action, (c1, c2) in DIAG_COMPONENTS.items():
                    if base in (c1, c2):
                        other = c2 if c1 == base else c1
                        odr, odc = MOVE_DELTA[other]
                        new_r = opr+ odr
                        new_c = opc + odc
                        if not self._in_bounds(new_r, new_c):
                            continue
                        if self._is_blocked(opr, opc, new_r, new_c):
                            continue
                        if (new_r, new_c) != (pos_r, pos_c):
                            legal_moves.append(diag_action)

        # [STEP 0-B 추가] 벽 설치 액션 검사 로직
        # TODO: 만약 현재 플레이어의 남은 벽이 0보다 크면:
        # TODO: 12부터 139까지 반복하며 _action_to_wall -> _is_valid_wall 확인
        # TODO: 유효하다면 벽을 임시로 세우고 _has_path(0) and _has_path(1) 확인!
        # TODO: 통과하면 legal_moves.append(a), 끝나면 벽 임시 세운 것 취소(remove)
        if self.state.walls_remaining[current_player] > 0:
            for action in range(12, ACTION_TOTAL):
                wall = self._action_to_wall(action)
                wr,wc,d = wall
                if not self._is_valid_wall(wr,wc,d):
                    continue
                self.state.walls_placed.add(wall)
                if not self._has_path(0) or not self._has_path(1):
                    self.state.walls_placed.remove(wall)
                    continue
                legal_moves.append(action)
                self.state.walls_placed.remove(wall)

        return legal_moves
        pass

    def _check_win(self, player: int) -> bool:
        # TODO: 승리 여부 반환
        return self.state.pos[player][0] == self.state.goal_rows[player]
        pass

    def step(self, action: int) -> Tuple[QuoridorState, float, bool]:
        # TODO: 액션 실행, 상태 업데이트(턴 전환 포함), reward, done 반환
        p = self.state.current_player
        if action <ACTION_MOVE:
            if 0 <= action <=3:
                dr, dc = MOVE_DELTA[action]
            elif 4 <= action <= 7:
                dr, dc = JUMP_DELTA[action]
            elif 8 <= action <= 11:
                dr, dc = DIAG_DELTA[action]

            self.state.pos[p] = (self.state.pos[p][0] +dr, self.state.pos[p][1] +dc)
        else:
            # [STEP 0-B 추가] 벽 설치 액션 처리
            # TODO: _action_to_wall을 호출해 좌표를 얻고 state.walls_placed에 추가
            # TODO: state.walls_remaining[p]를 1 감소
            self.state.walls_placed.add(self._action_to_wall(action))
            self.state.walls_remaining[p] -=1

        done = self._check_win(p)
        reward = 1.0 if done else 0

        if not done:
            self.state.current_player = 1 - p

        return (self.state, reward, done)

        pass

    def render(self) -> str:
        s  = self.state
        bs = s.board_size
        lines = []

        # 🎨 색상 정의 (ANSI Escape Codes)
        RESET = "\033[0m"
        P0_COLOR = "\033[1;31m"  # 진한 빨강 (P0)
        P1_COLOR = "\033[1;34m"  # 진한 파랑 (P1)
        WALL_COLOR = "\033[1;33m" # 진한 노랑 (벽)
        DIM = "\033[90m"         # 어두운 회색 (빈 공간/그리드)

        # 상단 좌표계 (열 번호)
        header = "    " + "   ".join(f"{DIM}{c}{RESET}" for c in range(bs))
        lines.append(header)

        for r in range(bs):
            # 좌측 좌표계 (행 번호)
            row_str = f"{DIM} {r} {RESET}"

            for c in range(bs):
                # 1. 플레이어 및 빈칸 렌더링
                if (r, c) == s.pos[0]:
                    cell = f"{P0_COLOR} ① {RESET}"
                elif (r, c) == s.pos[1]:
                    cell = f"{P1_COLOR} ② {RESET}"
                else:
                    cell = f"{DIM} · {RESET}"
                row_str += cell

                # 2. 세로 벽 렌더링
                if c < bs - 1:
                    vwall = any((wr, c, 1) in s.walls_placed for wr in (r, r - 1) if 0 <= wr <= bs - 2)
                    if vwall:
                        row_str += f"{WALL_COLOR}┃{RESET}"
                    else:
                        row_str += f"{DIM}│{RESET}"
            lines.append(row_str)

            # 3. 가로 벽 및 십자 교차점 렌더링
            if r < bs - 1:
                sep = "   "
                for c in range(bs):
                    # 가로 벽 확인
                    hwall = any((r, wc, 0) in s.walls_placed for wc in (c, c - 1) if 0 <= wc <= bs - 2)
                    if hwall:
                        sep += f"{WALL_COLOR}━━━{RESET}"
                    else:
                        sep += f"{DIM}───{RESET}"

                    # 십자 교차점 확인 (가로/세로 벽이 교차점을 지나가는지)
                    if c < bs - 1:
                        center_v = (r, c, 1) in s.walls_placed
                        center_h = (r, c, 0) in s.walls_placed
                        if center_v:
                            sep += f"{WALL_COLOR}┃{RESET}"
                        elif center_h:
                            sep += f"{WALL_COLOR}━{RESET}"
                        else:
                            sep += f"{DIM}┼{RESET}"
                lines.append(sep)

        # 4. 남은 벽 대시보드 추가
        footer = (
            f"\n  {P0_COLOR}[P0 ①] 남은 벽: {s.walls_remaining[0]}개{RESET}  |  "
            f"{P1_COLOR}[P1 ②] 남은 벽: {s.walls_remaining[1]}개{RESET}\n"
        )
        lines.append(footer)

        return "\n".join(lines)
        pass

    def encode_state(self) -> np.ndarray:
        """
        현재 상태를 6채널 Numpy 배열로 변환합니다.
        (채널 0: 내 위치, 1: 상대 위치, 2: 가로 벽, 3: 세로 벽, 4: 내 남은 벽, 5: 상대 남은 벽)
        self.state.current_player == 1 이라면 180도 회전하여 P0의 관점으로 인코딩합니다.
        """
        p = self.state.current_player
        opp = 1 - p
        bs = self.board_size

        state_idx = np.zeros((6, bs, bs), dtype=np.float32)

        pr, pc = self.state.pos[p]
        opr, opc = self.state.pos[opp]

        if p == 1:
            pr, pc = (bs - 1) - pr, (bs - 1) - pc
            opr, opc = (bs - 1) - opr, (bs - 1) - opc

        state_idx[0, pr, pc] = 1.0
        state_idx[1, opr, opc] = 1.0

        for wr, wc, wd in self.state.walls_placed:
            if p == 1:
                wr, wc = (bs - 2) - wr, (bs - 2) - wc

            if wd == 0:
                state_idx[2, wr, wc] = 1.0
            elif wd == 1:
                state_idx[3, wr, wc] = 1.0

        max_walls = max(self.num_walls, 1)
        state_idx[4, :, :] = self.state.walls_remaining[p] / max_walls
        state_idx[5, :, :] = self.state.walls_remaining[opp] / max_walls

        return state_idx
