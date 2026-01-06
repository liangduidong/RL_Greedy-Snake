import pygame
import random
import sys
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# 初始化 pygame
pygame.init()

# ====== 基本参数 ======
WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20
FPS = 30  # 提高训练速度

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 创建窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("强化学习贪吃蛇 - AI训练中")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)

# ====== 深度Q网络 ======
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ====== 强化学习代理 ======
class SnakeAgent:
    def __init__(self):
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
        # 状态空间: 11个特征
        # 动作空间: 3个动作 (直行, 左转, 右转)
        self.state_size = 11
        self.action_size = 3
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_size, 256, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, 256, self.action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # epsilon-greedy策略
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(self.device)
        
        # 当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 损失和优化
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ====== 游戏环境 ======
class SnakeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.direction = 1  # 0:上, 1:右, 2:下, 3:左
        self.snake_list = [[self.x, self.y]]
        self.snake_length = 1
        self.food_x = random.randrange(0, WIDTH - BLOCK_SIZE, BLOCK_SIZE)
        self.food_y = random.randrange(0, HEIGHT - BLOCK_SIZE, BLOCK_SIZE)
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()
    
    def get_state(self):
        head = self.snake_list[-1]
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]
        
        dir_l = self.direction == 3
        dir_r = self.direction == 1
        dir_u = self.direction == 0
        dir_d = self.direction == 2
        
        state = [
            # 危险在前方
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),
            
            # 危险在右边
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),
            
            # 危险在左边
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # 移动方向
            dir_l, dir_r, dir_u, dir_d,
            
            # 食物位置
            self.food_x < head[0],  # 食物在左边
            self.food_x > head[0],  # 食物在右边
            self.food_y < head[1],  # 食物在上边
            self.food_y > head[1]   # 食物在下边
        ]
        
        return np.array(state, dtype=int)
    
    def is_collision(self, point=None):
        if point is None:
            point = self.snake_list[-1]
        
        # 撞墙
        if point[0] < 0 or point[0] >= WIDTH or point[1] < 0 or point[1] >= HEIGHT:
            return True
        
        # 撞自己
        if point in self.snake_list[:-1]:
            return True
        
        return False
    
    def step(self, action):
        self.frame_iteration += 1
        
        # 动作: 0=直行, 1=右转, 2=左转
        clock_wise = [0, 1, 2, 3]  # 上右下左
        idx = clock_wise.index(self.direction)
        
        if action == 1:  # 右转
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:  # 左转
            new_dir = clock_wise[(idx - 1) % 4]
        else:  # 直行
            new_dir = self.direction
        
        self.direction = new_dir
        
        # 移动蛇
        if self.direction == 0:  # 上
            self.y -= BLOCK_SIZE
        elif self.direction == 1:  # 右
            self.x += BLOCK_SIZE
        elif self.direction == 2:  # 下
            self.y += BLOCK_SIZE
        elif self.direction == 3:  # 左
            self.x -= BLOCK_SIZE
        
        # 更新蛇身
        snake_head = [self.x, self.y]
        self.snake_list.append(snake_head)
        
        # 计算奖励
        reward = 0
        game_over = False
        
        # 检查游戏结束条件
        if self.is_collision() or self.frame_iteration > 100 * self.snake_length:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 吃到食物
        if self.x == self.food_x and self.y == self.food_y:
            self.score += 10
            reward = 10
            self.snake_length += 1
            self.food_x = random.randrange(0, WIDTH - BLOCK_SIZE, BLOCK_SIZE)
            self.food_y = random.randrange(0, HEIGHT - BLOCK_SIZE, BLOCK_SIZE)
        else:
            del self.snake_list[0]
        
        return reward, game_over, self.score
    
    def render(self, episode, total_score, record):
        screen.fill(BLACK)
        
        # 绘制蛇
        for x, y in self.snake_list:
            pygame.draw.rect(screen, GREEN, [x, y, BLOCK_SIZE, BLOCK_SIZE])
        
        # 绘制食物
        pygame.draw.rect(screen, RED, [self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE])
        
        # 显示信息
        text1 = font.render(f"Episode: {episode}", True, WHITE)
        text2 = font.render(f"Score: {self.score}", True, WHITE)
        text3 = font.render(f"Record: {record}", True, WHITE)
        text4 = font.render(f"Avg Score: {total_score/max(1, episode):.1f}", True, WHITE)
        
        screen.blit(text1, [10, 10])
        screen.blit(text2, [10, 40])
        screen.blit(text3, [10, 70])
        screen.blit(text4, [10, 100])
        
        pygame.display.update()

# ====== 训练函数 ======
def train():
    agent = SnakeAgent()
    game = SnakeGame()
    
    episode = 0
    total_score = 0
    record = 0
    
    print("开始训练AI...")
    print("按Q键退出训练")
    
    while True:
        # 检查事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print(f"\n训练结束!")
                    print(f"总回合数: {episode}")
                    print(f"最高分数: {record}")
                    print(f"平均分数: {total_score/max(1, episode):.2f}")
                    pygame.quit()
                    sys.exit()
        
        # 获取当前状态
        state_old = game.get_state()
        
        # 获取动作
        action = agent.act(state_old)
        
        # 执行动作
        reward, done, score = game.step(action)
        state_new = game.get_state()
        
        # 记忆
        agent.remember(state_old, action, reward, state_new, done)
        
        # 训练
        agent.replay()
        
        # 渲染
        game.render(episode, total_score, record)
        
        if done:
            episode += 1
            total_score += score
            
            if score > record:
                record = score
                print(f"新纪录! Episode {episode}: Score = {score}")
            
            # 每10回合更新目标网络
            if episode % 10 == 0:
                agent.update_target_model()
                print(f"Episode {episode}, Avg Score: {total_score/episode:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # 重置游戏
            game.reset()
        
        clock.tick(FPS)

if __name__ == "__main__":
    train()