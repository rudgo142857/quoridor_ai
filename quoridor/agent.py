"""
Quoridor Agents & Bots
"""
import math
import random
from collections import deque
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from quoridor.env import QuoridorEnv, QuoridorState, MOVE_DELTA


def flip_action(action: int) -> int:
    """
    P1 관점(180도 회전)에서의 액션을 실제 보드 액션으로, 또는 그 반대로 변환합니다.
    자체 역함수(Involution)이므로 두 번 적용하면 원래 액션으로 복원됩니다.
    flip_action(flip_action(a)) == a
    """
    if action < 12:
        mapping = {
            0: 1, 1: 0, 2: 3, 3: 2,          # 상하좌우
            4: 5, 5: 4, 6: 7, 7: 6,          # 직선 점프
            8: 11, 11: 8, 9: 10, 10: 9       # 대각 점프
        }
        return mapping[action]
    else:
        idx = action - 12
        orientation = idx // 64
        r = (idx % 64) // 8
        c = (idx % 64) % 8
        new_r = 7 - r
        new_c = 7 - c
        return 12 + orientation * 64 + new_r * 8 + new_c


class Agent:
    """모든 쿼리도 에이전트의 기본 추상 클래스"""
    def act(self, env: QuoridorEnv, legal_moves: List[int]) -> int:
        raise NotImplementedError


class RandomBot(Agent):
    """합법적인 수 중 무작위로 하나를 선택하는 베이스라인 봇"""
    def act(self, env: QuoridorEnv, legal_moves: List[int]) -> int:
        return random.choice(legal_moves)


class ShortestPathBot(Agent):
    """BFS를 이용해 항상 목표선까지의 최단 경로로 전진하는 봇"""
    def __init__(self, player: int = 0):
        self.player = player

    def get_shortest_path_length(self, env: QuoridorEnv, state: QuoridorState, player: int) -> float:
        r, c = state.pos[player]
        queue = deque([(0, r, c)])
        visited = set([(r, c)])
        goal_row = state.goal_rows[player]

        while queue:
            dist, cr, cc = queue.popleft()
            if cr == goal_row:
                return dist
            for dr, dc in MOVE_DELTA.values():
                nr, nc = cr + dr, cc + dc
                if (nr, nc) not in visited and env._in_bounds(nr, nc) and not env._is_blocked(cr, cc, nr, nc):
                    visited.add((nr, nc))
                    queue.append((dist + 1, nr, nc))
        return math.inf

    def act(self, env: QuoridorEnv, legal_moves: List[int]) -> int:
        move_actions = [a for a in legal_moves if a < 12]
        if not move_actions:
            return random.choice(legal_moves)

        best_action = None
        min_dist = math.inf
        p = env.state.current_player

        for action in move_actions:
            temp_env = QuoridorEnv(env.board_size, env.num_walls)
            temp_env.state = env.state.copy()
            temp_env.step(action)

            dist = self.get_shortest_path_length(temp_env, temp_env.state, p)
            if dist < min_dist:
                min_dist = dist
                best_action = action

        return best_action if best_action is not None else random.choice(move_actions)


class GreedyWallBot(Agent):
    """
    모든 합법 액션을 시뮬레이션하여
    (상대방 목표 최단거리 - 내 목표 최단거리)가 최대가 되는 최선의 수를 두는 방해꾼 봇
    """
    def __init__(self, player: int = 1):
        self.player = player

    def get_shortest_path_length(self, env: QuoridorEnv, player: int) -> float:
        r, c = env.state.pos[player]
        queue = deque([(0, r, c)])
        visited = set([(r, c)])
        goal_row = env.state.goal_rows[player]

        while queue:
            dist, cr, cc = queue.popleft()
            if cr == goal_row:
                return dist
            for dr, dc in MOVE_DELTA.values():
                nr, nc = cr + dr, cc + dc
                if env._in_bounds(nr, nc) and not env._is_blocked(cr, cc, nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((dist + 1, nr, nc))
        return math.inf

    def evaluate_state(self, env: QuoridorEnv) -> float:
        p = env.state.current_player
        opp = 1 - p
        my_dist = self.get_shortest_path_length(env, p)
        opp_dist = self.get_shortest_path_length(env, opp)

        if my_dist == math.inf: return -math.inf
        if opp_dist == math.inf: return math.inf
        return opp_dist - my_dist

    def act(self, env: QuoridorEnv, legal_moves: List[int]) -> int:
        best_action = None
        max_score = -math.inf

        for action in legal_moves:
            temp_env = QuoridorEnv(env.board_size, env.num_walls)
            temp_env.state = env.state.copy()
            temp_env.step(action)

            score = self.evaluate_state(temp_env)
            if action >= 12:
                score += 0.1  # 동점일 때 벽 우선 보너스
            if score > max_score:
                max_score = score
                best_action = action

        return best_action if best_action is not None else random.choice(legal_moves)


class LinearQBot(Agent):
    """수제 특징 5개와 선형 결합(Linear Function Approximation)을 사용하는 Q 에이전트"""
    def __init__(self, alpha: float = 0.002, gamma: float = 0.9, epsilon: float = 0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = np.zeros(5, dtype=np.float32)

    def extract_features(self, env: QuoridorEnv, action: int) -> np.ndarray:
        temp_env = QuoridorEnv(env.board_size, env.num_walls)
        temp_env.state = env.state.copy()
        p = temp_env.state.current_player
        opp = 1 - p
        temp_env.step(action)

        bs = float(env.board_size)
        max_w = float(max(env.num_walls, 1))

        # 거리 계산
        bot = ShortestPathBot()
        my_dist = bot.get_shortest_path_length(temp_env, temp_env.state, p)
        opp_dist = bot.get_shortest_path_length(temp_env, temp_env.state, opp)

        if my_dist == math.inf: my_dist = bs * 2
        if opp_dist == math.inf: opp_dist = bs * 2

        f0 = 1.0
        f1 = (opp_dist - my_dist) / bs
        f2 = -my_dist / bs
        f3 = opp_dist / bs
        f4 = float(temp_env.state.walls_remaining[p] - temp_env.state.walls_remaining[opp]) / max_w
        return np.array([f0, f1, f2, f3, f4], dtype=np.float32)

    def get_q_value(self, features: np.ndarray) -> float:
        return float(np.dot(self.weights, features))

    def act(self, env: QuoridorEnv, legal_moves: List[int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        best_a = legal_moves[0]
        best_q = -math.inf
        for a in legal_moves:
            feats = self.extract_features(env, a)
            q = self.get_q_value(feats)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a


class DQNAgent(Agent):
    """
    관점 정규화(Perspective Normalization) 및 액션 마스킹이 통합된
    풀 사이즈 9x9 Quoridor Double DQN 에이전트
    """
    def __init__(
        self,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        buffer,
        device: torch.device,
        gamma: float = 0.99
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.buffer = buffer
        self.device = device
        self.gamma = gamma

    def act(self, env: QuoridorEnv, legal_moves: List[int], epsilon: float = 0.0) -> int:
        """관점 정규화가 적용된 Epsilon-Greedy 행동 선택 (실제 보드 액션 반환)"""
        curr_p = env.state.current_player

        # P1 턴이라면 합법 행동들을 신경망 관점(180도 회전)으로 변환
        norm_legal_moves = [flip_action(a) for a in legal_moves] if curr_p == 1 else legal_moves

        if random.random() < epsilon:
            norm_action = random.choice(norm_legal_moves)
        else:
            curr_state = torch.tensor(env.encode_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.zeros((1, 140), dtype=torch.bool, device=self.device)
            mask[0, norm_legal_moves] = True

            with torch.no_grad():
                norm_action = self.policy_net(curr_state, mask).argmax(dim=1).item()

        # 신경망이 고른 행동(norm_action)을 실제 게임판 액션으로 복원
        real_action = flip_action(norm_action) if curr_p == 1 else norm_action
        return real_action

    def update(self, batch_size: int) -> float:
        """리플레이 버퍼에서 미니배치를 추출하여 1스텝 학습"""
        if len(self.buffer) < batch_size:
            return 0.0

        from quoridor.buffer import compute_double_dqn_loss
        batch = self.buffer.sample(batch_size, self.device)
        loss = compute_double_dqn_loss(self.policy_net, self.target_net, batch, gamma=self.gamma)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """타깃 네트워크 동기화"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
