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
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

# 创建窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("改进版强化学习贪吃蛇 - 势能函数")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)

# ====== 空域评估工具 ======
class SpaceAnalyzer:
    """Snake 的势能分析器"""
    
    @staticmethod
    def flood_fill(start, snake_body, width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE):
        """计算从起点可达的空格数量"""
        visited = set()
        queue = deque([tuple(start)])
        visited.add(tuple(start))
        snake_set = set(map(tuple, snake_body[:-1]))  # 不包括头部
        
        while queue:
            x, y = queue.popleft()
            
            # 四个方向
            for dx, dy in [(0, -block_size), (0, block_size), 
                          (-block_size, 0), (block_size, 0)]:
                nx, ny = x + dx, y + dy
                
                # 边界检查
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                
                # 已访问或是蛇身
                if (nx, ny) in visited or (nx, ny) in snake_set:
                    continue
                
                visited.add((nx, ny))
                queue.append((nx, ny))
        
        return len(visited)
    
    @staticmethod
    def bfs_distance(start, target, snake_body, width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE):
        """计算从头到尾的最短距离"""
        if start == target:
            return 0
        
        visited = set()
        queue = deque([(tuple(start), 0)])
        visited.add(tuple(start))
        snake_set = set(map(tuple, snake_body[:-1]))  # 不包括头部
        
        while queue:
            (x, y), dist = queue.popleft()
            
            # 四个方向
            for dx, dy in [(0, -block_size), (0, block_size), 
                          (-block_size, 0), (block_size, 0)]:
                nx, ny = x + dx, y + dy
                
                if (nx, ny) == tuple(target):
                    return dist + 1
                
                # 边界检查
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                
                # 已访问或是蛇身
                if (nx, ny) in visited or (nx, ny) in snake_set:
                    continue
                
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        
        return -1  # 不可达

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
        self.learning_rate = 0.0005
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
        # 扩展状态空间：增加势能特征
        self.state_size = 18  # 原15 + 3个势能特征
        self.action_size = 3
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_size, 256, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, 256, self.action_size).to(self.device)
        
        # === Bug Fix: 先初始化optimizer再加载模型 ===
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 记录加载时的元信息（episode / record / total_score / last_save_episode）
        self.loaded_episode = 0
        self.loaded_record = 0
        self.loaded_total_score = 0
        self.loaded_last_save_episode = 0

        if load_model and os.path.exists(load_model):
            metadata = self.load(load_model)
            if metadata:
                self.loaded_episode = metadata.get('episode') or 0
                self.loaded_record = metadata.get('record') or 0
                self.loaded_total_score = metadata.get('total_score') or 0
                self.loaded_last_save_episode = metadata.get('last_save_episode') or 0
            print(f"✓ 已加载模型: {load_model} (episode={self.loaded_episode}, record={self.loaded_record})")
        else:
            self.update_target_model()
        
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
            return
        
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
    
    def save(self, filename, episode=None, record=None, total_score=None, last_save_episode=None):
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),
            'episode': episode,
            'record': record,
            'total_score': total_score,
            'last_save_episode': last_save_episode
        }, filepath)
        print(f"✓ 模型已保存: {filepath}")
    
    def load(self, filename):
        import re
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.memory = deque(checkpoint.get('memory', []), maxlen=100000)

            metadata = {
                'episode': checkpoint.get('episode'),
                'record': checkpoint.get('record'),
                'total_score': checkpoint.get('total_score'),
                'last_save_episode': checkpoint.get('last_save_episode')
            }

            # 向后兼容：从文件名尝试解析 episode 或 record
            basename = os.path.basename(filename)
            if metadata['episode'] is None:
                m = re.search(r'ep(\d+)', basename)
                if m:
                    metadata['episode'] = int(m.group(1))
            if metadata['record'] is None:
                m = re.search(r'record[_\-]?(\d+)', basename)
                if m:
                    metadata['record'] = int(m.group(1))

            return metadata
        return None

# ====== 游戏环境 ======
class SnakeGame:
    def __init__(self):
        self.analyzer = SpaceAnalyzer()
        self.reset()
    
    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.direction = 1
        self.snake_list = [[self.x, self.y]]
        self.snake_length = 1
        self.generate_food()  # 使用新的生成食物函数
        self.score = 0
        self.frame_iteration = 0
        self.prev_distance = self.get_distance_to_food()
        
        # 记录势能信息（用于可视化）
        self.last_reachable = 0
        self.last_tail_dist = 0
        
        return self.get_state()
    
    def generate_food(self):
        """生成食物，确保不在蛇身上"""
        while True:
            self.food_x = random.randrange(0, WIDTH - BLOCK_SIZE, BLOCK_SIZE)
            self.food_y = random.randrange(0, HEIGHT - BLOCK_SIZE, BLOCK_SIZE)
            # 检查食物是否在蛇身上
            if [self.food_x, self.food_y] not in self.snake_list:
                break
    
    def get_distance_to_food(self):
        head = self.snake_list[-1]
        return abs(head[0] - self.food_x) + abs(head[1] - self.food_y)
    
    def check_body_nearby(self, point, distance=2):
        """检查指定点附近是否有蛇身体"""
        for segment in self.snake_list[:-1]:
            if abs(segment[0] - point[0]) <= distance * BLOCK_SIZE and \
               abs(segment[1] - point[1]) <= distance * BLOCK_SIZE:
                return True
        return False
    
    def get_state(self):
        head = self.snake_list[-1]
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]
        
        point_l2 = [head[0] - 2*BLOCK_SIZE, head[1]]
        point_r2 = [head[0] + 2*BLOCK_SIZE, head[1]]
        point_u2 = [head[0], head[1] - 2*BLOCK_SIZE]
        point_d2 = [head[0], head[1] + 2*BLOCK_SIZE]
        
        dir_l = self.direction == 3
        dir_r = self.direction == 1
        dir_u = self.direction == 0
        dir_d = self.direction == 2
        
        # === 计算势能特征 ===
        total_cells = (WIDTH // BLOCK_SIZE) * (HEIGHT // BLOCK_SIZE)
        
        # 1. 可达空格数量（flood-fill）
        reachable_cells = self.analyzer.flood_fill(head, self.snake_list)
        self.last_reachable = reachable_cells
        free_ratio = reachable_cells / total_cells
        
        # 2. 头到尾距离
        tail = self.snake_list[0] if len(self.snake_list) > 1 else head
        tail_distance = self.analyzer.bfs_distance(head, tail, self.snake_list)
        self.last_tail_dist = tail_distance
        
        # 归一化距离
        max_possible_dist = (WIDTH + HEIGHT) // BLOCK_SIZE
        tail_dist_normalized = tail_distance / max_possible_dist if tail_distance > 0 else 0
        
        # 3. 狭窄度检测
        is_narrow = 1 if reachable_cells < self.snake_length * 2 else 0
        
        state = [
            # 原有特征：危险检测
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),
            
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),
            
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # 身体接近检测
            (dir_r and self.check_body_nearby(point_r2)) or 
            (dir_l and self.check_body_nearby(point_l2)) or 
            (dir_u and self.check_body_nearby(point_u2)) or 
            (dir_d and self.check_body_nearby(point_d2)),
            
            (dir_u and self.check_body_nearby(point_r2)) or 
            (dir_d and self.check_body_nearby(point_l2)) or 
            (dir_l and self.check_body_nearby(point_u2)) or 
            (dir_r and self.check_body_nearby(point_d2)),
            
            (dir_d and self.check_body_nearby(point_r2)) or 
            (dir_u and self.check_body_nearby(point_l2)) or 
            (dir_r and self.check_body_nearby(point_u2)) or 
            (dir_l and self.check_body_nearby(point_d2)),
            
            # 移动方向
            dir_l, dir_r, dir_u, dir_d,
            
            # 食物位置
            self.food_x < head[0],
            self.food_x > head[0],
            self.food_y < head[1],
            self.food_y > head[1],
            
            # 蛇长度
            self.snake_length > 5,
            
            # === 新增：势能特征 ===
            free_ratio,              # 可达空间比例
            tail_dist_normalized,    # 头到尾距离（归一化）
            is_narrow,               # 是否在狭窄区域
        ]
        
        return np.array(state, dtype=float)
    
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
        
        # 根据动作确定新方向
        if action == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = self.direction
        
        # ===  Bug Fix 1: 禁止反向移动 ===
        # 检查是否是反向（相反方向差值为2）
        if abs(new_dir - self.direction) != 2:
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
        
        # 检查碰撞
        if self.is_collision():
            game_over = True
            if snake_head in self.snake_list[:-1]:
                reward = -15
            else:
                reward = -10
            return reward, game_over, self.score
        
        # 超时检查
        if self.frame_iteration > 100 * self.snake_length:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 吃到食物
        if self.x == self.food_x and self.y == self.food_y:
            self.score += 10
            reward = 10
            self.snake_length += 1
            self.generate_food()  # === Bug Fix 2: 使用安全的食物生成 ===
            self.prev_distance = self.get_distance_to_food()
        else:
            del self.snake_list[0]
            
            # === 改进的奖励函数：加入势能 ===
            
            # 1. 距离奖励（原有）
            current_distance = self.get_distance_to_food()
            if current_distance < self.prev_distance:
                reward = 0.1
            else:
                reward = -0.15
            self.prev_distance = current_distance
            
            # 2. 可达空间奖励
            total_cells = (WIDTH // BLOCK_SIZE) * (HEIGHT // BLOCK_SIZE)
            reachable_cells = self.analyzer.flood_fill(snake_head, self.snake_list)
            space_reward = 0.02 * (reachable_cells / total_cells)
            
            # 3. 头到尾距离奖励
            tail = self.snake_list[0]
            tail_distance = self.analyzer.bfs_distance(snake_head, tail, self.snake_list)
            if tail_distance > 0:
                tail_reward = 0.01 * min(tail_distance / 10, 1.0)
            else:
                tail_reward = -0.5  # 不可达惩罚
            
            # 4. 狭窄区域惩罚
            narrow_penalty = -0.3 if reachable_cells < self.snake_length * 2 else 0
            
            # 综合奖励
            reward += space_reward + tail_reward + narrow_penalty
        
        return reward, game_over, self.score
    
    def render(self, episode=0, total_score=0, record=0, epsilon=0, mode="train"):
        screen.fill(BLACK)
        
        # 绘制蛇
        for i, (x, y) in enumerate(self.snake_list):
            if mode == "demo":
                color = CYAN if i == len(self.snake_list) - 1 else BLUE
            else:
                color = YELLOW if i == len(self.snake_list) - 1 else GREEN
            pygame.draw.rect(screen, color, [x, y, BLOCK_SIZE, BLOCK_SIZE])
        
        # 绘制食物
        pygame.draw.rect(screen, RED, [self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE])
        
        # 显示信息
        if mode == "train":
            text1 = font.render(f"[训练模式] Episode: {episode}", True, WHITE)
            text2 = font.render(f"Score: {self.score}", True, WHITE)
            text3 = font.render(f"Record: {record}", True, WHITE)
            text4 = font.render(f"Avg: {total_score/max(1, episode):.1f}", True, WHITE)
            text5 = font.render(f"Epsilon: {epsilon:.3f}", True, WHITE)
            
            # 势能信息
            text6 = font.render(f"Space: {self.last_reachable}", True, ORANGE)
            text7 = font.render(f"Tail-Dist: {self.last_tail_dist}", True, ORANGE)
            
            screen.blit(text1, [10, 10])
            screen.blit(text2, [10, 40])
            screen.blit(text3, [10, 70])
            screen.blit(text4, [10, 100])
            screen.blit(text5, [10, 130])
            screen.blit(text6, [10, 160])
            screen.blit(text7, [10, 190])
        else:
            text1 = font.render(f"[演示模式] Score: {self.score}", True, CYAN)
            text2 = font.render(f"Best: {record}", True, CYAN)
            text3 = font.render(f"Space: {self.last_reachable}", True, ORANGE)
            text4 = font.render(f"Tail: {self.last_tail_dist}", True, ORANGE)
            text5 = font.render("Press ESC to exit", True, WHITE)
            
            screen.blit(text1, [10, 10])
            screen.blit(text2, [10, 40])
            screen.blit(text3, [10, 70])
            screen.blit(text4, [10, 100])
            screen.blit(text5, [10, 130])
        
        pygame.display.update()

# ====== 演示函数 ======
def demo(model_path):
    """演示训练好的模型"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    agent = SnakeAgent(load_model=model_path)
    agent.epsilon = 0
    game = SnakeGame()
    
    print("=" * 50)
    print("演示模式 - 观看AI玩贪吃蛇（势能版）")
    print("=" * 50)
    print("按 ESC 退出演示")
    print()
    
    best_score = 0
    game_count = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print(f"\n演示结束!")
                    print(f"游戏场次: {game_count}")
                    print(f"最高分数: {best_score}")
                    pygame.quit()
                    sys.exit()
        
        state = game.get_state()
        action = agent.act(state, training=False)
        reward, done, score = game.step(action)
        
        game.render(record=best_score, mode="demo")
        
        if done:
            game_count += 1
            if score > best_score:
                best_score = score
                print(f"🎉 新纪录! 游戏 {game_count}: {score} 分")
            else:
                print(f"游戏 {game_count}: {score} 分")
            game.reset()
        
        clock.tick(15)

# ====== 训练函数 ======
def train(load_model=None):
    # === Bug Fix 3: 训练模式不显示窗口 ===
    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 设置无头模式
    
    # 如果给的是模型文件名且存在于 models/ 目录，优先使用 models/<file>
    if load_model and not os.path.exists(load_model) and os.path.exists(os.path.join('models', load_model)):
        load_model = os.path.join('models', load_model)

    agent = SnakeAgent(load_model=load_model)
    game = SnakeGame()
    
    # 从加载的模型恢复训练统计（若有）
    episode = int(agent.loaded_episode) if hasattr(agent, 'loaded_episode') else 0
    total_score = float(agent.loaded_total_score) if hasattr(agent, 'loaded_total_score') else 0
    record = int(agent.loaded_record) if hasattr(agent, 'loaded_record') else 0
    last_save_episode = int(agent.loaded_last_save_episode) if hasattr(agent, 'loaded_last_save_episode') else 0
    
    print("=" * 50)
    print("改进版强化学习贪吃蛇 AI 训练（势能函数）")
    print("=" * 50)
    print("改进内容:")
    print("  ✓ flood-fill 空域评估")
    print("  ✓ 头到尾 BFS 距离")
    print("  ✓ 狭窄区域检测")
    print("  ✓ 势能奖励函数")
    print("  ✓ 禁止反向移动")
    print("  ✓ 食物避开蛇身")
    print("=" * 50)
    print("按键说明:")
    print("  Ctrl+C - 退出训练并保存模型")
    print("=" * 50)
    
    if load_model:
        print(f"继续训练: {load_model}")
    else:
        print("开始新训练...")
    print()
    
    last_save_episode = 0
    
    try:
        while True:
            state_old = game.get_state()
            action = agent.act(state_old)
            reward, done, score = game.step(action)
            state_new = game.get_state()
            
            agent.remember(state_old, action, reward, state_new, done)
            agent.replay()
            
            if done:
                episode += 1
                total_score += score
                
                if score > record:
                    record = score
                    print(f"🎉 新纪录! Episode {episode}: {score} 分")
                    agent.save(f"snake_potential_record_{score}.pth", episode=episode, record=record, total_score=total_score, last_save_episode=last_save_episode)
                
                if episode % 10 == 0:
                    agent.update_target_model()
                    avg_score = total_score / episode
                    print(f"Episode {episode}, Avg: {avg_score:.2f}, Record: {record}, ε: {agent.epsilon:.3f}")
                
                if episode - last_save_episode >= 100:
                    agent.save(f"snake_potential_ep{episode}.pth", episode=episode, record=record, total_score=total_score, last_save_episode=episode)
                    last_save_episode = episode
                
                game.reset()
    
    except KeyboardInterrupt:
        print(f"\n\n训练中断!")
        print(f"总回合数: {episode}")
        print(f"最高分数: {record}")
        print(f"平均分数: {total_score/max(1, episode):.2f}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent.save(f"snake_potential_{timestamp}.pth", episode=episode, record=record, total_score=total_score, last_save_episode=last_save_episode)
        print("模型已保存，退出中...")
        sys.exit()

# ====== 主菜单 ======
def main_menu():
    print("=" * 50)
    print("改进版强化学习贪吃蛇（势能函数）")
    print("=" * 50)
    print("1. 开始新训练")
    print("2. 继续训练 (加载模型)")
    print("3. 演示模式 (观看AI)")
    print("4. 退出")
    print("=" * 50)
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice == "1":
        train()
    elif choice == "2":
        model_file = input("输入模型文件路径: ").strip()
        train(load_model=model_file)
    elif choice == "3":
        model_file = input("输入模型文件路径: ").strip()
        demo(model_file)
    elif choice == "4":
        print("再见!")
        sys.exit()
    else:
        print("无效选择!")
        main_menu()

if __name__ == "__main__":
    main_menu()