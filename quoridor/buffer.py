"""
Experience Replay Buffer & Double DQN Zero-Sum TD Loss
"""
import random
from collections import deque
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """
    경험 튜플 (state, action, reward, next_state, done, next_mask)을 저장하고
    학습 시 무작위 미니배치를 샘플링하여 파이토치 텐서로 반환하는 버퍼입니다.
    """
    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray
    ):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """미니배치를 무작위로 추출하여 파이토치 텐서 튜플로 반환합니다."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        b_states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        b_actions = torch.tensor(actions, dtype=torch.long, device=device)
        b_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        b_next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        b_dones = torch.tensor(dones, dtype=torch.float32, device=device)
        b_next_masks = torch.tensor(np.array(next_masks), dtype=torch.bool, device=device)

        return b_states, b_actions, b_rewards, b_next_states, b_dones, b_next_masks

    def __len__(self) -> int:
        return len(self.buffer)


def compute_double_dqn_loss(
    policy_net: nn.Module,
    target_net: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    gamma: float = 0.99
) -> torch.Tensor:
    """
    더블 DQN (Double DQN) 기반의 제로섬 벨만 TD 손실(Loss)을 계산합니다.
    1. 행동 선택: Policy Net이 next_states에서 최고 Q값의 최적 행동(a*)을 선택 (next_masks 적용)
    2. 가치 평가: Target Net이 그 행동(a*)의 Q값을 독립적으로 평가
    3. 제로섬 타깃 계산: Target = reward - (1 - dones) * gamma * Target_Q(s', a*)
    4. 손실 계산: Smooth L1 Loss (Huber Loss) 반환
    """
    states, actions, rewards, next_states, dones, next_masks = batch

    # 1. 현재 상태에서의 예측 Q값 추출
    q_all = policy_net(states)
    current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 2. 다음 상태에서 Policy Net이 최적 행동 선택
    next_actions = policy_net(next_states, next_masks).argmax(dim=1)

    # 3. Target Net을 이용한 해당 행동의 가치 평가 (Overestimation 방지)
    with torch.no_grad():
        target_q_all = target_net(next_states)
        next_q = target_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)

    # 4. 제로섬 Bellman 타깃 계산 (다음 턴은 상대방 차례이므로 마이너스 부호 적용)
    target = rewards - (1.0 - dones) * gamma * next_q

    # 5. Smooth L1 손실 계산 후 반환
    loss = F.smooth_l1_loss(current_q, target)
    return loss
