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
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = 20
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE
FPS = 1000  # 训练时极快，不渲染

RENDER_MODE = False  # 训练时关闭渲染
RENDER_INTERVAL = 50  # 每50个episode渲染一次

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)

# 创建窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT + 100))
pygame.display.set_caption("CNN-DQN 贪吃蛇")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)

# ====== 卷积神经网络 DQN ======
class ConvDQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ConvDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        conv_output_size = GRID_WIDTH * GRID_HEIGHT * 64
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ====== CNN-DQN 代理 ======
class CNNAgent:
    def __init__(self, load_model=None):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # 保持10%探索
        self.epsilon_decay = 0.99995  # 超慢衰减：10000 ep → 0.6
        self.learning_rate = 0.0001
        self.memory = deque(maxlen=100000)
        self.batch_size = 64  # 增大batch size
        
        self.frame_stack = 4
        self.action_size = 4  # 上右下左
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.model = ConvDQN(self.frame_stack, self.action_size).to(self.device)
        self.target_model = ConvDQN(self.frame_stack, self.action_size).to(self.device)
        
        # 先初始化 optimizer/criterion，以便 load() 可以恢复 optimizer 状态
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 训练统计信息（用于断点重训）
        self.training_stats = {
            'episodes': 0,
            'total_score': 0,
            'record': 0,
            'scores_history': []
        }
        
        if load_model and os.path.exists(load_model):
            self.load(load_model)
            print(f"✓ 已加载模型: {load_model}")
        else:
            self.update_target_model()
        
        self.frame_buffer = deque(maxlen=self.frame_stack)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filename):
        # 在保存前更新 training_stats（若外部未更新）
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
        }, filepath)
        print(f"✓ 模型已保存: {filepath}")
    
    def load(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint.get('model_state_dict', {}))
            self.target_model.load_state_dict(checkpoint.get('target_model_state_dict', {}))
            # 确保 optimizer 存在
            if not hasattr(self, 'optimizer') or self.optimizer is None:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print(f"⚠️ 无法加载 optimizer 状态: {e}")
            # 使用 checkpoint 中的 epsilon（若存在）
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            # 恢复 training_stats（若存在）
            if 'training_stats' in checkpoint:
                self.training_stats = checkpoint['training_stats']
            return True
        return False

# ====== 游戏环境（修复版） ======
class SnakeGameCNN:
    def __init__(self, episode=0):
        self.episode = episode
        self.reset()
    
    def reset(self):
        self.head = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        self.snake_list = [self.head.copy()]
        self.direction = 1  # 初始向右
        self.length = 1
        self.place_food()
        self.score = 0
        self.frame_iteration = 0
        return self.get_frame()
    
    def place_food(self):
        while True:
            self.food = [random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)]
            if self.food not in self.snake_list:
                break
    
    def get_frame(self):
        frame = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        
        # 蛇身
        for segment in self.snake_list[:-1]:
            if 0 <= segment[0] < GRID_WIDTH and 0 <= segment[1] < GRID_HEIGHT:
                frame[segment[1], segment[0]] = 1
        
        # 蛇头
        head = self.snake_list[-1]
        if 0 <= head[0] < GRID_WIDTH and 0 <= head[1] < GRID_HEIGHT:
            frame[head[1], head[0]] = 2
        
        # 食物
        frame[self.food[1], self.food[0]] = 3
        
        return frame
    
    def is_collision(self, point=None):
        if point is None:
            point = self.head
        
        if point[0] < 0 or point[0] >= GRID_WIDTH or point[1] < 0 or point[1] >= GRID_HEIGHT:
            return True
        
        if point in self.snake_list[:-1]:
            return True
        
        return False
    
    def step(self, action):
        """
        ✅ 修复1：禁止180度掉头
        """
        # 禁止掉头
        opposite = {0: 2, 2: 0, 1: 3, 3: 1}
        if action == opposite.get(self.direction, -1):
            action = self.direction  # 保持原方向
        
        self.frame_iteration += 1
        prev_distance = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
        
        self.direction = action
        
        # 移动
        new_head = self.head.copy()
        if action == 0:  # 上
            new_head[1] -= 1
        elif action == 1:  # 右
            new_head[0] += 1
        elif action == 2:  # 下
            new_head[1] += 1
        elif action == 3:  # 左
            new_head[0] -= 1
        
        self.head = new_head
        self.snake_list.append(self.head.copy())
        
        reward = 0
        done = False
        
        # 检查碰撞
        if self.is_collision():
            done = True
            if self.head in self.snake_list[:-1]:
                reward = -15  # 撞自己
            else:
                reward = -10  # 撞墙
            return reward, done, self.score, self.get_frame()
        
        # ✅ 修复2：大幅放宽超时限制
        # 前500 episode：超宽松
        if self.episode < 500:
            timeout_limit = 500 * max(self.length, 1)
        # 500-2000 episode：逐步收紧
        elif self.episode < 2000:
            timeout_limit = 300 * max(self.length, 1)
        # 2000+ episode：正常限制
        else:
            timeout_limit = 200 * max(self.length, 1)
        
        if self.frame_iteration > timeout_limit:
            done = True
            reward = -5  # 轻微惩罚
            return reward, done, self.score, self.get_frame()
        
        # 吃到食物
        if self.head == self.food:
            self.score += 1
            reward = 50  # ✅ 修复4：大幅增加食物奖励
            self.length += 1
            self.place_food()
        else:
            self.snake_list.pop(0)
            
            # ✅ 修复3：前期友好的距离奖励
            curr_distance = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            if self.length < 5:
                # 短蛇：只奖励靠近，不惩罚远离
                if curr_distance < prev_distance:
                    reward = 1.0
                else:
                    reward = 0
            else:
                # 长蛇：正常奖惩
                if curr_distance < prev_distance:
                    reward = 0.5
                else:
                    reward = -0.3
        
        return reward, done, self.score, self.get_frame()
    
    def render(self, episode=0, total_score=0, record=0, epsilon=0, loss=0, mode="train"):
        screen.fill(BLACK)
        
        # 绘制游戏区域
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                
                if [x, y] in self.snake_list[:-1]:
                    color = GREEN if mode == "train" else BLUE
                    pygame.draw.rect(screen, color, rect)
                elif [x, y] == self.head:
                    color = (255, 255, 0) if mode == "train" else CYAN
                    pygame.draw.rect(screen, color, rect)
                elif [x, y] == self.food:
                    pygame.draw.rect(screen, RED, rect)
        
        # 网格线
        for x in range(0, WIDTH, BLOCK_SIZE):
            pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, BLOCK_SIZE):
            pygame.draw.line(screen, (50, 50, 50), (0, y), (WIDTH, y))
        
        # 显示信息
        info_y = HEIGHT + 10
        if mode == "train":
            text1 = font.render(f"[CNN-已修复] Ep: {episode}", True, WHITE)
            text2 = font.render(f"Score: {self.score} | Record: {record}", True, WHITE)
            text3 = font.render(f"Avg: {total_score/max(1, episode):.2f} | ε: {epsilon:.3f}", True, WHITE)
            text4 = font.render(f"Loss: {loss:.4f} | Steps: {self.frame_iteration}", True, WHITE)
            
            screen.blit(text1, [10, info_y])
            screen.blit(text2, [10, info_y + 25])
            screen.blit(text3, [10, info_y + 50])
            screen.blit(text4, [10, info_y + 75])
        else:
            text1 = font.render(f"[演示] Score: {self.score}", True, CYAN)
            text2 = font.render(f"Best: {record} | ESC退出", True, CYAN)
            
            screen.blit(text1, [10, info_y])
            screen.blit(text2, [10, info_y + 25])
        
        pygame.display.update()

# ====== 演示函数 ======
def demo(model_path):
    global RENDER_MODE
    RENDER_MODE = True
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    agent = CNNAgent(load_model=model_path)
    agent.epsilon = 0
    game = SnakeGameCNN()
    
    print("=" * 50)
    print("CNN-DQN 演示模式")
    print("=" * 50)
    
    best_score = 0
    game_count = 0
    
    frame = game.reset()
    for _ in range(4):
        agent.frame_buffer.append(frame)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print(f"\n演示结束! 游戏: {game_count}, 最佳: {best_score}")
                    pygame.quit()
                    sys.exit()
        
        state = np.array(agent.frame_buffer)
        action = agent.act(state, training=False)
        reward, done, score, next_frame = game.step(action)
        
        agent.frame_buffer.append(next_frame)
        
        game.render(record=best_score, mode="demo")
        
        if done:
            game_count += 1
            if score > best_score:
                best_score = score
                print(f"🎉 新纪录! 游戏 {game_count}: {score} 分")
            else:
                print(f"游戏 {game_count}: {score} 分")
            
            frame = game.reset()
            for _ in range(4):
                agent.frame_buffer.append(frame)
        
        clock.tick(10)

# ====== 训练函数（完全修复版） ======
def train(load_model=None):
    agent = CNNAgent(load_model=load_model)
    
    # 如果从 checkpoint 加载，恢复训练统计
    episode = agent.training_stats.get('episodes', 0)
    total_score = agent.training_stats.get('total_score', 0)
    record = agent.training_stats.get('record', 0)
    last_loss = 0
    
    print("=" * 60)
    print("CNN-DQN 贪吃蛇训练 [已应用所有修复]")
    print("=" * 60)
    print("✅ 修复1: 禁止180度掉头")
    print("✅ 修复2: 大幅放宽超时限制（前500ep: 500×length）")
    print("✅ 修复3: 前期只奖励靠近，不惩罚远离")
    print("✅ 修复4: 吃食物奖励 ×5 (10→50)")
    print("✅ 修复5: 训练时不渲染，每50ep显示一次")
    print("=" * 60)
    print("预期：500-1000 ep 开始明显学习")
    print("按键: Q-退出 | S-保存 | R-切换渲染")
    print("=" * 60)
    print()
    
    game = SnakeGameCNN(episode=episode)
    frame = game.reset()
    for _ in range(4):
        agent.frame_buffer.append(frame)
    
    last_save_episode = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 在保存前更新训练统计
                    agent.training_stats['episodes'] = episode
                    agent.training_stats['total_score'] = total_score
                    agent.training_stats['record'] = record
                    agent.training_stats.setdefault('scores_history', [])
                    agent.save(f"cnn_snake_fixed_{timestamp}.pth")
                    print(f"\n✅ 训练结束!")
                    print(f"Ep: {episode} | 记录: {record} | 平均: {total_score/max(1,episode):.2f}")
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_s:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    agent.training_stats['episodes'] = episode
                    agent.training_stats['total_score'] = total_score
                    agent.training_stats['record'] = record
                    agent.training_stats.setdefault('scores_history', [])
                    agent.save(f"cnn_snake_manual_{timestamp}.pth")
                elif event.key == pygame.K_r:
                    global RENDER_MODE
                    RENDER_MODE = not RENDER_MODE
                    print(f"渲染模式: {'开启' if RENDER_MODE else '关闭'}")
        
        state = np.array(agent.frame_buffer)
        action = agent.act(state)
        
        game.episode = episode  # 更新episode用于动态超时
        reward, done, score, next_frame = game.step(action)
        
        agent.frame_buffer.append(next_frame)
        next_state = np.array(agent.frame_buffer)
        
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()
        if loss > 0:
            last_loss = loss
        
        # ✅ 修复5：大幅减少渲染
        should_render = RENDER_MODE and (episode % RENDER_INTERVAL == 0 or episode < 10)
        if should_render:
            game.render(episode, total_score, record, agent.epsilon, last_loss, mode="train")
            clock.tick(FPS)
        
        if done:
            episode += 1
            total_score += score
            
            # 更新 training_stats
            agent.training_stats['episodes'] = episode
            agent.training_stats['total_score'] = total_score
            agent.training_stats['record'] = record
            agent.training_stats.setdefault('scores_history', []).append(score)
            # 限制历史长度
            if len(agent.training_stats['scores_history']) > 1000:
                agent.training_stats['scores_history'] = agent.training_stats['scores_history'][-1000:]
            
            if score > record:
                record = score
                agent.training_stats['record'] = record
                print(f"🎉 新纪录! Ep {episode}: {score} 分")
                agent.save(f"cnn_snake_record_{score}.pth")
            
            if episode % 50 == 0:
                agent.update_target_model()
                avg = total_score / episode
                print(f"Ep {episode:5d} | Avg: {avg:5.2f} | Record: {record:3d} | ε: {agent.epsilon:.3f} | Loss: {last_loss:.4f}")
                
                # 阶段提示
                if episode == 100 and avg < 0.5:
                    print("    💡 前100轮探索期，avg<0.5正常")
                elif episode == 500:
                    print(f"    {'✅' if avg >= 1 else '⚠️'} 500轮检查点: avg={avg:.2f} (期望≥1)")
                elif episode == 1000:
                    print(f"    {'✅' if avg >= 3 else '⚠️'} 1000轮检查点: avg={avg:.2f} (期望≥3)")
                elif episode == 2000:
                    print(f"    {'✅' if avg >= 8 else '⚠️'} 2000轮检查点: avg={avg:.2f} (期望≥8)")
            
            if episode - last_save_episode >= 500:
                # 保存时确保 training_stats 已更新
            for _ in range(4):
                agent.frame_buffer.append(frame)

# ====== 主菜单 ======
def main_menu():
    print("=" * 60)
    print("CNN-DQN 贪吃蛇 [完全修复版]")
    print("=" * 60)
    print("1. 开始新训练 (已应用所有专业建议)")
    print("2. 继续训练")
    print("3. 演示模式")
    print("4. 退出")
    print("=" * 60)
    
    choice = input("选择 (1-4): ").strip()
    
    if choice == "1":
        train()
    elif choice == "2":
        model_file = input("模型路径: ").strip()
        train(load_model=model_file)
    elif choice == "3":
        model_file = input("模型路径: ").strip()
        demo(model_file)
    elif choice == "4":
        sys.exit()
    else:
        print("无效选择!")
        main_menu()

if __name__ == "__main__":
    main_menu()