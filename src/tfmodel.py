import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from shutil import copyfile


# ModelTop (Policy Network)
class ModelTop(nn.Module):
    def __init__(self, s_dim, pi_dim):
        super(ModelTop, self).__init__()
        self.s_dim = s_dim
        self.pi_dim = pi_dim

        # Define the network layers
        self.qpi_net = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, pi_dim)  # No activation (logits)
        )

    def encode_s(self, s0):
        logits_pi = self.qpi_net(s0)
        q_pi = F.softmax(logits_pi, dim=-1)
        log_q_pi = torch.log(q_pi + 1e-20)
        return logits_pi, q_pi, log_q_pi


# ModelMid (Transition Model)
class ModelMid(nn.Module):
    def __init__(self, s_dim, pi_dim):
        super(ModelMid, self).__init__()
        self.s_dim = s_dim
        self.pi_dim = pi_dim

        # Define the network layers
        self.ps_net = nn.Sequential(
            nn.Linear(pi_dim + s_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2 * s_dim)  # Output mean and logvar for reparameterization
        )

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def transition(self, pi, s0):
        ps_params = self.ps_net(torch.cat([pi, s0], dim=1))
        mean, logvar = torch.split(ps_params, self.s_dim, dim=1)
        return mean, logvar

    def transition_with_sample(self, pi, s0):
        mean, logvar = self.transition(pi, s0)
        ps1 = self.reparameterize(mean, logvar)
        return ps1, mean, logvar


# ModelDown (Observation Model)
class ModelDown(nn.Module):
    def __init__(self, s_dim, pi_dim, colour_channels, resolution):
        super(ModelDown, self).__init__()
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.colour_channels = colour_channels
        self.resolution = resolution

        # Encoder (q(s|o))
        self.qs_net = nn.Sequential(
            nn.Conv2d(colour_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (resolution // 16) * (resolution // 16), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, s_dim * 2)  # Mean and logvar
        )

        # Decoder (p(o|s))
        self.po_net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64 * (resolution // 16) * (resolution // 16)),
            nn.ReLU(),
            nn.Unflatten(1, (64, resolution // 16, resolution // 16)),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, colour_channels, 3, stride=1, padding=1),
            nn.Sigmoid()  # Output probability
        )

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def encoder(self, o):
        params = self.qs_net(o)
        mean, logvar = torch.split(params, self.s_dim, dim=1)
        return mean, logvar

    def decoder(self, s):
        return self.po_net(s)

    def encoder_with_sample(self, o):
        mean, logvar = self.encoder(o)
        s = self.reparameterize(mean, logvar)
        return s, mean, logvar


# ActiveInferenceModel
class ActiveInferenceModel:
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.gamma = gamma
        self.beta_s = beta_s
        self.beta_o = beta_o

        if pi_dim > 0:
            self.model_top = ModelTop(s_dim, pi_dim)
            self.model_mid = ModelMid(s_dim, pi_dim)
        self.model_down = ModelDown(s_dim, pi_dim, colour_channels, resolution)

        # Policy one-hot encodings
        self.pi_one_hot = torch.eye(pi_dim)
        self.pi_one_hot_3 = torch.eye(3)

    def save_weights(self, folder_chp):
        torch.save(self.model_down.state_dict(), os.path.join(folder_chp, 'checkpoint_down.pth'))
        if self.pi_dim > 0:
            torch.save(self.model_top.state_dict(), os.path.join(folder_chp, 'checkpoint_top.pth'))
            torch.save(self.model_mid.state_dict(), os.path.join(folder_chp, 'checkpoint_mid.pth'))

    def load_weights(self, folder_chp):
        self.model_down.load_state_dict(torch.load(os.path.join(folder_chp, 'checkpoint_down.pth')))
        if self.pi_dim > 0:
            self.model_top.load_state_dict(torch.load(os.path.join(folder_chp, 'checkpoint_top.pth')))
            self.model_mid.load_state_dict(torch.load(os.path.join(folder_chp, 'checkpoint_mid.pth')))

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        self.save_weights(folder_chp)
        with open(os.path.join(folder_chp, 'stats.pkl'), 'wb') as f:
            pickle.dump(stats, f)
        with open(os.path.join(folder_chp, 'optimizers.pkl'), 'wb') as f:
            pickle.dump(optimizers, f)
        copyfile('src/tfmodel.py', os.path.join(folder_chp, 'tfmodel.py'))
        copyfile('src/tfloss.py', os.path.join(folder_chp, 'tfloss.py'))
        if script_file:
            copyfile(script_file, os.path.join(folder_chp, script_file))

    def load_all(self, folder_chp):
        self.load_weights(folder_chp)
        with open(os.path.join(folder_chp, 'stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
        try:
            with open(os.path.join(folder_chp, 'optimizers.pkl'), 'rb') as f:
                optimizers = pickle.load(f)
        except FileNotFoundError:
            optimizers = {}
        return stats, optimizers

    def check_reward(self, o):
        # Assuming this depends on some reward computation logic
        if self.model_down.resolution == 64:
            return o.mean(dim=[1, 2, 3]) * 10.0
        elif self.model_down.resolution == 32:
            return o.sum(dim=[1, 2, 3])

    def habitual_net(self, o):
        qs_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs_mean)
        return Qpi
    def imagine_future_from_o(self, o0, pi):
        s0, _, _ = self.model_down.encoder_with_sample(o0)
        ps1, _, _ = self.model_mid.transition_with_sample(pi, s0)
        po1 = self.model_down.decoder(ps1)
        return po1

    def habitual_net(self, o):
        qs_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs_mean)
        return Qpi

    def calculate_G_repeated(self, o, pi, steps=1, calc_mean=False, samples=10):
        # Calculate current s_t
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [torch.zeros(o.size(0)), torch.zeros(o.size(0)), torch.zeros(o.size(0))]
        sum_G = torch.zeros(o.size(0))

        # Predict s_t+1 for various policies
        s0_temp = qs0_mean if calc_mean else qs0

        for _ in range(steps):
            G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, pi, samples=samples)

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_G += G

            s0_temp = ps1_mean if calc_mean else s1

        return sum_G, sum_terms, po1

    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        # Calculate current s_t
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [torch.zeros(4), torch.zeros(4), torch.zeros(4)]
        sum_G = torch.zeros(4)

        s0_temp = qs0_mean if calc_mean else qs0

        for _ in range(steps):
            if calc_mean:
                G, terms, ps1_mean, po1 = self.calculate_G_mean(s0_temp, self.pi_one_hot)
            else:
                G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, self.pi_one_hot, samples=samples)

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_G += G

            s0_temp = ps1_mean if calc_mean else s1

        return sum_G, sum_terms, po1

    def calculate_G(self, s0, pi0, samples=10):
        term0 = torch.zeros(s0.size(0))
        term1 = torch.zeros(s0.size(0))

        for _ in range(samples):
            ps1, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
            po1 = self.model_down.decoder(ps1)
            qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

            logpo1 = self.check_reward(po1)
            term0 += logpo1
            term1 += -torch.sum(ps1_logvar + qs1_logvar, dim=1)

        term0 /= float(samples)
        term1 /= float(samples)

        term2_1 = torch.zeros(s0.size(0))
        term2_2 = torch.zeros(s0.size(0))

        for _ in range(samples):
            # Sampling different thetas
            po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[0])
            term2_1 += torch.sum(torch.sigmoid(po1_temp1), dim=[1, 2, 3])

            # Sampling different s with the same theta
            po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
            term2_2 += torch.sum(torch.sigmoid(po1_temp2), dim=[1, 2, 3])

        term2_1 /= float(samples)
        term2_2 /= float(samples)

        term2 = term2_1 - term2_2

        G = -term0 + term1 + term2
        return G, [term0, term1, term2], ps1, ps1_mean, po1

    def calculate_G_mean(self, s0, pi0):
        _, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
        po1 = self.model_down.decoder(ps1_mean)
        _, qs1_mean, qs1_logvar = self.model_down.encoder_with_sample(po1)

        logpo1 = self.check_reward(po1)
        term0 = logpo1

        term1 = -torch.sum(ps1_logvar + qs1_logvar, dim=1)

        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[1])
        term2_1 = torch.sum(torch.sigmoid(po1_temp1), dim=[1, 2, 3])

        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
        term2_2 = torch.sum(torch.sigmoid(po1_temp2), dim=[1, 2, 3])

        term2 = term2_1 - term2_2

        G = -term0 + term1 + term2
        return G, [term0, term1, term2], ps1_mean, po1

    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        po1 = self.model_down.decoder(ps1_traj)
        qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

        term0 = self.check_reward(po1)

        term1 = -torch.sum(ps1_logvar_traj + qs1_logvar, dim=1)

        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0_traj, s0_traj)[0])
        term2_1 = torch.sum(torch.sigmoid(po1_temp1), dim=[1, 2, 3])

        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean_traj, ps1_logvar_traj))
        term2_2 = torch.sum(torch.sigmoid(po1_temp2), dim=[1, 2, 3])

        term2 = term2_1 - term2_2

        return -term0 + term1 + term2

    def mcts_step_simulate(self, starting_s, depth, use_means=False):
        s0 = torch.zeros((depth, self.s_dim))
        ps1 = torch.zeros((depth, self.s_dim))
        ps1_mean = torch.zeros((depth, self.s_dim))
        ps1_logvar = torch.zeros((depth, self.s_dim))
        pi0 = torch.zeros((depth, self.pi_dim))

        s0[0] = starting_s
        Qpi_t_to_return = self.model_top.encode_s(s0[0].unsqueeze(0))[1]
        pi0[0, torch.multinomial(Qpi_t_to_return.squeeze(), 1)] = 1.0

        for t in range(1, depth):
            ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[t - 1].unsqueeze(0), s0[t - 1].unsqueeze(0))
            ps1[t] = ps1_new
            ps1_mean[t] = ps1_mean_new
            ps1_logvar[t] = ps1_logvar_new

            if t + 1 < depth:
                if use_means:
                    s0[t + 1] = ps1_mean[t]
                else:
                    s0[t + 1] = ps1[t]

        G = self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)
        return G, pi0, Qpi_t_to_return
