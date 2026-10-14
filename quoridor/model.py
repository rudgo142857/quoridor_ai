"""
Quoridor DQN Neural Network Architecture
"""
import torch
import torch.nn as nn


class QuoridorDQN(nn.Module):
    """
    쿼리도 9x9 풀 사이즈 심층 Q-네트워크 (CNN + Action Masking)

    입력:
        x: (Batch_Size, 6, 9, 9) 상태 텐서
           - 채널 0: 내 말 위치 (1.0)
           - 채널 1: 상대 말 위치 (1.0)
           - 채널 2: 가로 벽 위치 (1.0)
           - 채널 3: 세로 벽 위치 (1.0)
           - 채널 4: 내 남은 벽 수 (0.0 ~ 1.0)
           - 채널 5: 상대 남은 벽 수 (0.0 ~ 1.0)
        mask: (Batch_Size, 140) 불리언 텐서 (True: 합법 행동, False: 불법 행동)

    출력:
        q_values: (Batch_Size, 140) 각 행동에 대한 Q-value
    """
    def __init__(self, in_channels: int = 6, num_actions: int = 140):
        super(QuoridorDQN, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        순전파 및 액션 마스킹 연산
        불법 행동(mask == False)의 Q값을 -1e9로 마스킹하여 선택되지 않도록 방지합니다.
        """
        q_values = self.layer(x)
        if mask is not None:
            q_values = q_values.masked_fill(~mask, -1e9)
        return q_values
