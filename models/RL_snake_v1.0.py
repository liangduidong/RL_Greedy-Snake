import pygame
import random
import sys
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime

# 初始化 pygame
pygame.init()

# ====== 基本参数 ======
WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20
FPS = 30

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
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ====== 强化学习代理 ======
class SnakeAgent:
    def __init__(self, load_model=None):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005  # 降低学习率提高稳定性
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
        self.state_size = 11
        self.action_size = 3
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_size, 256, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, 256, self.action_size).to(self.device)
        
        # 加载模型
        if load_model and os.path.exists(load_model):
            self.load(load_model)
            print(f"已加载模型: {load_model}")
        else:
            self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """保存模型"""
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filename):
        """加载模型"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.memory = deque(checkpoint['memory'], maxlen=100000)
            return True
        return False

# ====== 游戏环境 ======
class SnakeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.direction = 1
        self.snake_list = [[self.x, self.y]]
        self.snake_length = 1
        self.food_x = random.randrange(0, WIDTH - BLOCK_SIZE, BLOCK_SIZE)
        self.food_y = random.randrange(0, HEIGHT - BLOCK_SIZE, BLOCK_SIZE)
        self.score = 0
        self.frame_iteration = 0
        self.prev_distance = self.get_distance_to_food()
        return self.get_state()
    
    def get_distance_to_food(self):
        """计算到食物的曼哈顿距离"""
        head = self.snake_list[-1]
        return abs(head[0] - self.food_x) + abs(head[1] - self.food_y)
    
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
            self.food_x < head[0],
            self.food_x > head[0],
            self.food_y < head[1],
            self.food_y > head[1]
        ]
        
        return np.array(state, dtype=int)
    
    def is_collision(self, point=None):
        if point is None:
            point = self.snake_list[-1]
        
        if point[0] < 0 or point[0] >= WIDTH or point[1] < 0 or point[1] >= HEIGHT:
            return True
        
        if point in self.snake_list[:-1]:
            return True
        
        return False
    
    def step(self, action):
        self.frame_iteration += 1
        
        clock_wise = [0, 1, 2, 3]
        idx = clock_wise.index(self.direction)
        
        if action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = self.direction
        
        self.direction = new_dir
        
        if self.direction == 0:
            self.y -= BLOCK_SIZE
        elif self.direction == 1:
            self.x += BLOCK_SIZE
        elif self.direction == 2:
            self.y += BLOCK_SIZE
        elif self.direction == 3:
            self.x -= BLOCK_SIZE
        
        snake_head = [self.x, self.y]
        self.snake_list.append(snake_head)
        
        reward = 0
        game_over = False
        
        # 更严格的超时惩罚
        if self.is_collision() or self.frame_iteration > 50 * self.snake_length:
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
            self.prev_distance = self.get_distance_to_food()
        else:
            del self.snake_list[0]
            
            # 距离奖励：靠近食物给小奖励,远离给小惩罚
            current_distance = self.get_distance_to_food()
            if current_distance < self.prev_distance:
                reward = 0.1
            else:
                reward = -0.1
            self.prev_distance = current_distance
        
        return reward, game_over, self.score
    
    def render(self, episode, total_score, record, epsilon):
        screen.fill(BLACK)
        
        for x, y in self.snake_list:
            pygame.draw.rect(screen, GREEN, [x, y, BLOCK_SIZE, BLOCK_SIZE])
        
        pygame.draw.rect(screen, RED, [self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE])
        
        text1 = font.render(f"Episode: {episode}", True, WHITE)
        text2 = font.render(f"Score: {self.score}", True, WHITE)
        text3 = font.render(f"Record: {record}", True, WHITE)
        text4 = font.render(f"Avg: {total_score/max(1, episode):.1f}", True, WHITE)
        text5 = font.render(f"Epsilon: {epsilon:.3f}", True, WHITE)
        
        screen.blit(text1, [10, 10])
        screen.blit(text2, [10, 40])
        screen.blit(text3, [10, 70])
        screen.blit(text4, [10, 100])
        screen.blit(text5, [10, 130])
        
        pygame.display.update()

# ====== 训练函数 ======
def train(load_model=None):
    agent = SnakeAgent(load_model=load_model)
    game = SnakeGame()
    
    episode = 0
    total_score = 0
    record = 0
    
    print("=" * 50)
    print("强化学习贪吃蛇 AI 训练")
    print("=" * 50)
    print("按键说明:")
    print("  Q - 退出训练")
    print("  S - 保存模型")
    print("=" * 50)
    
    if load_model:
        print(f"继续训练模型: {load_model}")
    else:
        print("开始新训练...")
    print()
    
    last_save_episode = 0
    
    while True:
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
                    
                    # 自动保存
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    agent.save(f"snake_model_final_{timestamp}.pth")
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_s:
                    # 手动保存
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    agent.save(f"snake_model_manual_{timestamp}.pth")
        
        state_old = game.get_state()
        action = agent.act(state_old)
        reward, done, score = game.step(action)
        state_new = game.get_state()
        
        agent.remember(state_old, action, reward, state_new, done)
        agent.replay()
        
        game.render(episode, total_score, record, agent.epsilon)
        
        if done:
            episode += 1
            total_score += score
            
            if score > record:
                record = score
                print(f"🎉 新纪录! Episode {episode}: Score = {score}")
                # 破纪录时自动保存
                agent.save(f"snake_model_record_{score}.pth")
            
            if episode % 10 == 0:
                agent.update_target_model()
                avg_score = total_score / episode
                print(f"Episode {episode}, Avg: {avg_score:.2f}, Record: {record}, ε: {agent.epsilon:.3f}")
            
            # 每100回合自动保存
            if episode - last_save_episode >= 100:
                agent.save(f"snake_model_ep{episode}.pth")
                last_save_episode = episode
            
            game.reset()
        
        clock.tick(FPS)

if __name__ == "__main__":
    # 使用示例:
    # 1. 新训练: python snake_rl.py
    # 2. 继续训练: 取消下面的注释并指定模型文件
    # train(load_model='models/snake_model_ep100.pth')
    
    train()  # 新训练
    # train(load_model='models/snake_model_record_480.pth')  # 继续训练