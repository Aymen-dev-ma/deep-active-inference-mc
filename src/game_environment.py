import time
import numpy as np
import torch

class Game:
    def __init__(self, number_of_games=1):
        current_time = time.time()
        dataset = np.load('C:\Users\aymen\OneDrive\Documents\GitHub\deep-active-inference-mc\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
        self.imgs = torch.tensor(dataset['imgs'].reshape(-1, 64, 64, 1), dtype=torch.float32)
        latents_values = dataset['latents_values']
        latents_classes = dataset['latents_classes']
        metadata = dataset['metadata'][()]
        self.s_sizes = metadata['latents_sizes']  # [1 3 6 40 32 32]
        self.s_dim = self.s_sizes.size + 1  # Last dimension is reward!
        self.s_bases = torch.cat((torch.tensor(np.copy(metadata['latents_sizes'][::-1].cumprod()[::-1][1:]), dtype=torch.float32),
                                  torch.tensor([1.], dtype=torch.float32)))  # [737280 245760  40960 1024 32 1]
        self.s_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY', 'reward']
        self.games_no = number_of_games
        self.current_s = torch.zeros((self.games_no, self.s_dim), dtype=torch.float32)
        self.last_r = torch.zeros(self.games_no, dtype=torch.float32)
        self.new_image_all()
        print('Dataset loaded. Time:', time.time() - current_time, 'datapoints:', len(self.imgs), self.s_dim, self.s_bases)

    def sample_s(self):  # Reward is zero after this!
        s = torch.zeros(self.s_dim, dtype=torch.long)
        for s_i, s_size in enumerate(self.s_sizes):
            s[s_i] = torch.randint(s_size, (1,)).item()
        return s

    def sample_s_all(self):  # Reward is zero after this!
        s = torch.zeros((self.games_no, self.s_dim), dtype=torch.long)
        for s_i, s_size in enumerate(self.s_sizes):
            s[:, s_i] = torch.randint(0, s_size, (self.games_no,))
        return s

    def s_to_index(self, s):
        return torch.matmul(s.float(), self.s_bases).long()

    def s_to_o(self, index):
        image_to_return = self.imgs[self.s_to_index(self.current_s[index, :-1])].cpu().numpy().astype(np.float32)

        # Adding the reward encoded to the image..
        if 0.0 <= self.last_r[index] <= 1.0:
            image_to_return[0:3, 0:32] = self.last_r[index]
        elif -1.0 <= self.last_r[index] < 0.0:
            image_to_return[0:3, 32:64] = -self.last_r[index]
        else:
            raise ValueError(f'Error: Reward: {self.last_r[index]}')
        return image_to_return

    def reward_to_rgb(self, reward):
        return torch.tensor([min(1.0, -reward + 1), min(1.0, reward + 1), 1.0 - abs(reward)], dtype=torch.float32)

    def current_frame(self, index):
        return self.s_to_o(index)

    def current_frame_all(self):
        o = torch.zeros((self.games_no, 64, 64, 1), dtype=torch.float32)
        for i in range(self.games_no):
            o[i] = torch.tensor(self.s_to_o(i), dtype=torch.float32).unsqueeze(0)
        return o

    def randomize_environment(self, index):
        self.current_s[index] = self.sample_s()
        self.current_s[index, 6] = -10 + torch.rand(1).item() * 20
        self.last_r[index] = -1.0 + torch.rand(1).item() * 2.0

    def randomize_environment_all(self):
        self.current_s = self.sample_s_all()
        self.current_s[:, 6] = -10 + torch.rand(self.games_no).numpy() * 20
        self.last_r = -1.0 + torch.rand(self.games_no).numpy() * 2.0

    def new_image(self, index):
        reward = self.current_s[index, 6]  # pass reward to the new latent..!
        self.current_s[index] = self.sample_s()
        self.current_s[index, 6] = reward

    def new_image_all(self):
        reward = self.current_s[:, 6]  # pass reward to the new latent..!
        self.current_s = self.sample_s_all()
        self.current_s[:, 6] = reward

    def get_reward(self, index):
        return self.current_s[index, 6]

    # NOTE: Randomness takes values from zero to one.
    def find_move(self, index, randomness):
        right = 0.5 * (1.0 - randomness / 2.0)
        wrong = 0.5 * randomness / 2.0
        if self.current_s[index, 1] < 0.5:  # Square
            Ppi = torch.tensor([right, wrong, wrong, right], dtype=torch.float32)
        else:  # Ellipse or heart
            Ppi = torch.tensor([right, wrong, right, wrong], dtype=torch.float32)
        return Ppi

    def find_move_all(self, randomness):
        return torch.stack([self.find_move(i, randomness) for i in range(self.games_no)], dim=0)

    # NOTE: Randomness takes values from zero to one.
    def auto_play(self, index, randomness=0.4):
        Ppi = self.find_move(index, randomness)
        pi = torch.multinomial(Ppi, 1).item()
        self.pi_to_action(pi, index)
        return pi, Ppi

    def tick(self, index):
        self.last_r[index] *= 0.95

    def tick_all(self):
        self.last_r *= 0.95

    def up(self, index):
        self.tick(index)
        self.current_s[index, 5] += 1.0
        if self.current_s[index, 5] >= 32:
            if self.current_s[index, 1] < 0.5:  # Square
                if self.current_s[index, 4] > 15:
                    self.last_r[index] = float(15.0 - self.current_s[index, 4]) / 16.0
                else:
                    self.last_r[index] = float(16.0 - self.current_s[index, 4]) / 16.0
                self.current_s[index, 6] += self.last_r[index]
            else:  # Ellipse or heart
                if self.current_s[index, 4] > 15:
                    self.last_r[index] = float(self.current_s[index, 4] - 15.0) / 16.0
                else:
                    self.last_r[index] = float(self.current_s[index, 4] - 16.0) / 16.0
                self.current_s[index, 6] += self.last_r[index]
            self.new_image(index)
            return True
        return False

    def down(self, index):
        self.tick(index)
        if self.current_s[index, 5] > 0:
            self.current_s[index, 5] -= 1.0

    def left(self, index):
        self.tick(index)
        if self.current_s[index, 4] < 31:
            self.current_s[index, 4] += 1.0

    def right(self, index):
        self.tick(index)
        if self.current_s[index, 4] > 0:
            self.current_s[index, 4] -= 1.0

    def pi_to_action(self, pi, index, repeats=1):
        round_changed = False
        for i in range(repeats):
            if pi == 0:
                round_changed = self.up(index)
            elif pi == 1:
                self.down(index)
            elif pi == 2:
                self.left(index)
            elif pi == 3:
                self.right(index)
            else:
                raise ValueError('Value error!')
            if round_changed:
                return True
        return False
