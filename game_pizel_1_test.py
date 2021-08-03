import pygame
import os
import random
from collections import deque
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as plt


class Deep_q_learning:

    def __init__(self, game,
                 max_memory=50_000, epsilon=0, gamma=0.9, lr=0.001,
                 outputs=3, inputs=11, hidden_layers=[256],
                 batch_size=100,  average_last=1,
                 show_graph=False, plot_every=10, plot_display_time=2,
                 plot_save_every=None, plot_save_path="", save_model_path="Deep_Q_Model1",
                 score_threshold=None, load_model_path=None):

        self.plot_scores = []
        self.plot_scoresx = []
        self.plot_mean_scores = []
        self.model_save_path = save_model_path
        self.total_score = 0
        self.score_threshold = score_threshold
        self.record = 0
        self.outputs=outputs
        self.plot_save_path = plot_save_path
        self.plot_save_every = plot_save_every
        self.average_last = average_last
        self.show_graph = show_graph
        self.agent = Agent(max_memory=max_memory, gamma=gamma, epsilon=epsilon, lr=lr,
                           outputs=outputs, inputs=inputs, hidden_layers=hidden_layers,
                           batch_size=batch_size, load_model_path=load_model_path)
        self.game = game
        self.game.agent = self.agent
        self.valid_game = True
        self.plot_display_time=plot_display_time
        self.plot_every = plot_every
        self.left_score_ptr = 0
        if not(callable(getattr(self.game, 'reset', None))):
            self.valid_game = False
            print("Warning: Game object class does not contain method reset\n" +
                  "reset method should reset/restart the game\n")
        else:
            pass

        if not(callable(getattr(self.game, 'play_step', None))):
            self.valid_game = False
            print("Warning: Game object class does not contain method play_step\n" +
                  "play_step method takes in action and returns following values\n" +
                  "1)Reward\n2)game_over (boolean)\n3)score\n")
        else:
            pass

        if not(callable(getattr(self.game, 'get_state', None))):
            self.valid_game = False
            print("Warning: Game object class does not contain get_state\n" +
                  "play_step method should return an array of states that are given input to the neural nets\n")
        else:
            pass





    def plot_graph(self, n_iters):


        if (n_iters%self.plot_every == 0) or (n_iters%self.plot_save_every==0) and n_iters!=0:
            #print('haello!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', n_iters%self.plot_save_every, self.plot_every)
            plt.style.use("dark_background")
            plt.ion()
            plt.plot(self.plot_scoresx, self.plot_scores, label='Score')
            plt.plot(self.plot_scoresx, self.plot_mean_scores, label= 'Avg Score')
            plt.ylabel("Score", fontdict={'size':10, 'color':'blue'})
            plt.xlabel("Number of Games", fontdict={'size':10, 'color':'blue'})
            plt.legend()
            if n_iters % self.plot_every == 0:
                plt.show()
            if self.plot_save_every and n_iters%self.plot_save_every==0:
                plt.savefig(self.plot_save_path+str(int(n_iters/self.plot_save_every)))
            plt.pause(self.plot_display_time)
            plt.close()




    def play_game(self, end_game_score=None):
        while True:
            final_move = [0]*self.outputs
            state = self.game.get_state()
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.agent.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            reward, game_over, score = game.play_step(final_move)
            if game_over:
                game.reset()
            if end_game_score:
                if score>=end_game_score:
                    print("Score: " + str(score))
                    break



    def train_model(self):
        if self.valid_game:
            n_iters = 0
            while True:
                # get old state
                state_old = self.game.get_state()

                # get move
                final_move = self.agent.get_action(state_old)

                # perform move and get new state
                reward, done, score = self.game.play_step(final_move)
                state_new = self.game.get_state()  # /8

                self.agent.train_short_memory(state_old, final_move, reward, state_new, done)
                self.agent.remember(state_old, final_move, reward, state_new, done)


                if done or score>self.score_threshold:
                    n_iters+=1
                    # train long memory
                    self.game.reset()
                    self.agent.n_games += 1
                    self.agent.train_long_memory()
                    self.score = score

                    if self.score > self.record:
                        self.record = self.score
                        self.agent.model.save(filename=self.model_save_path)
                    print('Game', self.agent.n_games, 'Score', self.score, 'Record:', self.record)
                    self.plot_scores.append(score)
                    self.plot_scoresx.append(n_iters)
                    self.total_score += score
                    if n_iters>self.average_last:
                        self.total_score-=self.plot_scores[self.left_score_ptr]
                        self.left_score_ptr += 1
                    self.plot_mean_scores.append(self.total_score / min(n_iters, self.average_last))


                    self.plot_graph(n_iters)
        else:
            print("The game object class does not contain the required methods\nCheck if class contains reset, train_step, methods ")









class Linear_QNet(nn.Module):

    def __init__(self, input_size=10, hidden_sizes=[256],
                 hidden_activation='relu', output_size=3,
                 dropout=None, input_activation='relu',
                 output_activation='linear'):
        super().__init__()
        temp_size = hidden_sizes[0]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, temp_size))
        if len(hidden_sizes) > 1:
            for i in range(1, len(hidden_sizes)):
                self.layers.append(nn.Linear(temp_size, i))
                temp_size = i

        self.layers.append(nn.Linear(temp_size, output_size))

    def forward(self, x):
        for i in self.layers[:-1]:
            x = f.relu(i(x))

        x = self.layers[-1](x)
        return x

    def save(self, filename='model.pth'):
        model_folder_path = './mode'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        filename = os.path.join(model_folder_path, filename)
        torch.save(self.state_dict(), filename)

    def load_model(self, filepath="model.pth"):
        self.load_state_dict(torch.load(filepath))
        self.eval()


class QTrainer:
        def __init__(self, model, lr, gamma):
            self.lr = lr
            self.gamma = gamma
            self.model = model
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.MINIBATCH_SIZE = 100

        def train_step(self, state, action, reward, next_state, done):

            state = torch.tensor(state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            next_state = torch.tensor(next_state, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)

            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)

            pred = self.model(state)

            target = pred.clone()
            for i in range(len(done)):
                Q_new = reward[i]
                if not done[i]:
                    Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

                target[i][torch.argmax(action).item()] = Q_new

            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()

class Agent:

        def __init__(self, max_memory=50_000, epsilon=0, gamma=0.9, lr=0.001,
                     outputs=3, inputs=11, hidden_layers=[256],
                     batch_size=100, load_model_path=None):
            self.n_games = 0
            self.epsilon = epsilon  # Randomness to agent
            self.memory = deque(maxlen=max_memory)
            self.batch_size = batch_size
            self.gamma = gamma
            self.outputs=outputs

            # model

            self.model = Linear_QNet(inputs, hidden_layers, output_size=outputs)
            if load_model_path != None:
                self.model.load_model("mode/pixel1_deep_q_model")
            self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

        def train_long_memory(self):

            if len(self.memory) > self.batch_size:
                mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

        def train_short_memory(self, state, action, reward, next_state, done):
            self.trainer.train_step(state, action, reward, next_state, done)

        def get_action(self, state):
            # random moves: tradeoff exploration / exploitation
            self.epsilon = 80 - self.n_games
            final_move = [0]*self.outputs
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

            return final_move







































##################################################################Game



SPEED=100
WIDTH = 20
HEIGHT = 20
BLOCK_LENGTH = 10
COLORS = {"blue": (0, 0, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "orange": (255, 165, 0),
          "pink": (255, 192, 203), "yellow": (255, 255, 0)}



class Game1:
    def __init__(self):
        self.block1x = 10
        self.centre = 10
        self.block1y = random.randint(5, self.centre)
        self.block2x = 19
        self.block2y = random.randint(5, self.centre)
        self.playerx = 0
        self.playery = random.randint(7, 13)
        self.score = 0
        self.reward=0
        self.display = pygame.display.set_mode((WIDTH*10, WIDTH*10))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Game1')

    def reset(self):
        self.block1x = 10
        self.block1y = random.randint(5, self.centre)
        self.block2x = 19
        self.block2y = random.randint(5, self.centre)
        self.playerx = 0
        self.playery = random.randint(0, 19)
        self.score = 0


    def move_player(self, dir=1):
        self.playery+=dir*2
        if self.playery<0:
            self.playery=0
        if self.playery>HEIGHT-1:
            self.playery=HEIGHT-1

    def play_step(self, action):
        self.reward=0
        game_over = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #print(self.playerx, self.playery, self.block1x, self.block1y)
        if action[0]==1:
            self.move_player(1)
        elif action[1]==1:
            self.move_player(-1)
        self.move_blocks()
        if self.detect_collision():
            self.reward-=20
            game_over=True
        self.update_game()

        return self.reward, game_over, self.score


    def detect_collision(self):
        if self.playerx==self.block1x:
            if self.playery<=self.block1y:
                return True
        if self.playerx==self.block2x:
            if self.playery>=(HEIGHT-self.block2y):
                return True


    def move_blocks(self):
        self.block1x-=1
        self.block2x-=1
        if self.block1x<0:
            self.score+=1
            self.reward+=10
            self.block1x=WIDTH
            self.block1y=random.randint(5, self.centre)
        if self.block2x<0:
            self.score+=1
            self.reward+=10
            self.block2x=WIDTH
            self.block2y=random.randint(5, self.centre)

    def get_state(self):
        temp_screen = pygame.PixelArray(self.display)
        new_screen = []
        temp_divider =[255]*20
        for j,i in enumerate(temp_screen):
            if j%10==0:
                temp = list(i[0:200:10])
                #print(temp)
                new_screen.append(temp)
        new_screen[self.playerx][self.playery]=-255


        #print(temp_screen)
        #print(np.array(new_screen).shape)
        #print(len(new_screen[0]))
        #print(np.array(new_screen).flatten().shape)
        new_screen = np.array(new_screen)/255
        #print("\n\n\n\n\n")
        #for i in new_screen:
        #        print(i)
        new_screen = new_screen.flatten()

        #print(new_screen)
        return new_screen

    def update_game(self):
        temp_screen = pygame.PixelArray(self.display)
        new_screen = []
        for i in temp_screen:
            new_screen.append(temp_screen[0:200:10])
        #print(new_screen)
        self.display.fill((0, 0, 0))
        #pygame.draw.circle(self.display, COLORS['red'], (self.playerx*10, self.playery*10), 10)
        pygame.draw.rect(self.display, COLORS['blue'], (self.playerx*10, self.playery*10, 10, 10), 0)
        pygame.draw.rect(self.display, COLORS['blue'], (self.block1x*10, 0, 10, self.block1y*10), 0)
        pygame.draw.rect(self.display, COLORS['blue'], (self.block2x * 10, (HEIGHT-self.block2y)*10, 10, self.block2y*10), 0)
        pygame.display.flip()
        self.clock.tick(SPEED)






game= Game1()

ql_model = Deep_q_learning(game=game, plot_save_every=500, plot_every=100, inputs=400, outputs=3, hidden_layers=[256, 256],
                           save_model_path="pixel1_deep_q_model2", score_threshold=1000)
#ql_model.play_game(end_game_score=1000)
ql_model.train_model()
#while True:
    #arr=[0, 0]
    #arr[random.randint(0,1)]=1
    #game.play_step(arr)






