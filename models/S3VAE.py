import numpy as np
from itertools import chain
from torch import autograd
from S3VAE.models.models import *
# from flownet2-pytorch.models import FlowNet2


class S3VAE:
    def __init__(self, config, device='cpu'):
        self.device = device
        self._config = config
        self.build_model()

    def build_model(self):
        self._encode_net = Encoder().to(self.device)
        self._decode_net = Decoder(input_size=self._config['model']['zf_size']+
                                              self._config['model']['zt_size']).to(self.device)
        self._static_lstm = LSTMEncoder(input_size=128, z_size=self._config['model']['zf_size'],
                                        static=True).to(self.device)
        self._dynamic_lstm = LSTMEncoder(input_size=128, z_size=self._config['model']['zt_size'],
                                         static=False).to(self.device)
        self._prior_lstm = LSTMEncoder(input_size=self._config['model']['zt_size'],
                                       z_size=self._config['model']['zt_size'],
                                       static=False).to(self.device)

        self._dfp = DynamicFactorPrediction(z_size=self._config['model']['zt_size']).to(self.device)
        # margin = size(z_f) / 2 - to be proportional to dimensionality
        self._triplet_loss = torch.nn.TripletMarginLoss(margin=1)#self._config['model']['zf_size'] / 2)
        self.model = chain(self._encode_net.parameters(),
                           self._decode_net.parameters(),
                           self._static_lstm.parameters(),
                           self._dynamic_lstm.parameters(),
                           self._prior_lstm.parameters(),
                           self._dfp.parameters())
        self._optimizer = torch.optim.Adam(self.model,
                                           lr=float(self._config['model']['lr']),
                                           betas=(float(self._config['model']['beta_0']),
                                                  float(self._config['model']['beta_0'])))

    def save(self, path):
        torch.save(self._encode_net, path + 'encoder.pt')
        torch.save(self._decode_net, path + 'decoder.pt')
        torch.save(self._static_lstm, path + 'static.pt')
        torch.save(self._dynamic_lstm, path + 'dynamic.pt')
        torch.save(self._prior_lstm, path + 'prior.pt')
        torch.save(self._dfp, path + 'dfp.pt')

    def load(self, path):
        self._encode_net.load_state_dict(torch.load(path + 'encoder.pt', map_location=self.device))
        self._encode_net.eval()
        self._decode_net.load_state_dict(torch.load(path + 'decoder.pt', map_location=self.device))
        self._decode_net.eval()
        self._static_lstm.load_state_dict(torch.load(path + 'static.pt', map_location=self.device))
        self._static_lstm.eval()
        self._dynamic_lstm.load_state_dict(torch.load(path + 'dynamic.pt', map_location=self.device))
        self._dynamic_lstm.eval()
        self._prior_lstm.load_state_dict(torch.load(path + 'prior.pt', map_location=self.device))
        self._prior_lstm.eval()
        self._dfp.load_state_dict(torch.load(path + 'dfp.pt', map_location=self.device))
        self._dfp.eval()

    def compute_loss(self, x, y, other):
        def vae_loss(x, x_hat, z_f, z_f_prior, z_t, z_t_prior):
            # # sum of log probs
            # bce_error = -F.binary_cross_entropy(x_hat, x, reduction='sum') / self._config.T
            # # KL for time-invariant representation
            # # https://arxiv.org/abs/1312.6114 (Appendix B)
            # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)  ???
            # log_var = torch.log(sigma_f.pow(2))
            # kl_element_time_invariant = mu_f.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            # kl_time_invariant = torch.sum(kl_element_time_invariant).mul_(0.5)
            # # KL for time-dependant representation
            # kl_time_dependant = 0
            # for t in range(self._config.T):
            #     log_var = torch.log(sigma_t[t].pow(2))
            #     kl_element_time_invariant = mu_t[t].pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            #     kl_time_dependant += torch.sum(kl_element_time_invariant).mul_(0.5)

            img_loss = x_hat.log_prob(x.transpose(0, 1)).sum(-1).sum(-1).sum(-1).mean()
            kl_td = dist.kl_divergence(z_t, z_t_prior).sum(-1).sum(0).mean()
            kl_ti = dist.kl_divergence(z_f, z_f_prior).sum(-1).mean()
            return -img_loss + kl_td + kl_ti

        def static_consistency_constraint(z_f, z_f_pos, z_f_neg):
            # z_f -- time-invariant representation from real data
            # z_f_pos -- time-invariant representation from shuffled real video
            # z_f_neg -- time-invariant representation from  another video
            # return: max(D(z_f, z_f_pos) - D(z_f, z_f_neg) + margin, 0)
            return self._triplet_loss(z_f.rsample(), z_f_pos.sample(), z_f_neg.sample())

        def dynamic_factor_prediction(z_t, y):
            prediction = self._dfp(z_t)
            return F.mse_loss(prediction.rsample().reshape(-1, 1), y.transpose(0, 1).reshape(-1, 1))

        def mutual_information_regularization(z_f, z_t):
            def dist_op(dist1, op):
                return dist.Normal(loc=op(dist1.loc), scale=op(dist1.scale))
            # H_t entropy
            # below shorthand for
            # z_f.loc = z_f.loc.unsqueeze(0)
            # z_f.scale = z_f.scale.unsqueeze(0)
            z_t1 = dist_op(z_t, lambda x: x.unsqueeze(1))
            z_t2 = dist_op(z_t, lambda x: x.unsqueeze(2))
            log_q_t = z_t1.log_prob(z_t2.rsample()).sum(-1)
            # 2 is important here!
            H_t = log_q_t.logsumexp(2).mean(1) - np.log(log_q_t.shape[2])

            z_f1 = dist_op(z_f, lambda x: x.unsqueeze(0))
            z_f2 = dist_op(z_f, lambda x: x.unsqueeze(1))
            log_q_f = z_f1.log_prob(z_f2.rsample()).sum(-1)
            H_f = log_q_f.logsumexp(1).mean(0) - np.log(log_q_t.shape[2])
            H_ft = (log_q_f.unsqueeze(0) + log_q_t).logsumexp(1).mean(1)

            mi_loss = -(H_f + H_t.mean() - H_ft.mean())
            return mi_loss

        # get all images after encoder to feed to static LSTM, shape: (batch_size, 1, 128)
        encoded_tensor = self._encode_net(x).transpose(0, 1).contiguous()

        # static LSTM gives distribution for z_f and input for dynamic LSTM
        z_f = self._static_lstm(encoded_tensor)

        # shuffle batch to get z_f_pos
        shuffle_idx = torch.randperm(encoded_tensor.shape[1])
        shuffled_encoded_tensor = encoded_tensor[:, shuffle_idx].contiguous()
        z_f_p = self._static_lstm(shuffled_encoded_tensor)

        # get z_f_neg from another data
        another_encoded_tensor = self._encode_net(other).transpose(0, 1).contiguous()
        z_f_n = self._static_lstm(another_encoded_tensor)

        # dynamic LSTM gives distribution for z_t on each timestep
        z_t = self._dynamic_lstm(encoded_tensor)
        z_t_sample = z_t.rsample()
        z = torch.cat([z_t_sample,
                       z_f.rsample().unsqueeze(0).repeat(z_t.loc.shape[0], 1, 1)], 2)

        z_f_prior = dist.Normal(loc=z.new_zeros(z_f.loc.shape), scale=z.new_ones(z_f.loc.shape))

        # note: drop gradients with detach
        z_t_prior = self._prior_lstm(z_t_sample.detach())

        # shift time
        loc = z_t_prior.loc[:-1]
        scale = z_t_prior.scale[:-1]
        z_t_prior.loc = torch.cat([z.new_zeros(loc[:1].shape), loc], 0)
        z_t_prior.scale = torch.cat([z.new_ones(scale[:1].shape), scale], 0)

        # decode back to frame seq
        x_hat = self._decode_net(z)
        vae = vae_loss(x=x, x_hat=x_hat, z_f=z_f, z_f_prior=z_f_prior, z_t=z_t, z_t_prior=z_t_prior)
        scc = static_consistency_constraint(z_f, z_f_p, z_f_n)
        dfp = dynamic_factor_prediction(z_t_sample, y)
        mi = mutual_information_regularization(z_f, z_t)
        loss = vae + self._config['model']['lambda_1'] * scc + \
               self._config['model']['lambda_2'] * dfp + self._config['model']['lambda_3'] * mi
        return loss, vae, scc, dfp, mi, x_hat

    def train_step(self, x, y, other):
        loss, vae, scc, dfp, mi, _ = self.compute_loss(x, y, other)
        self._optimizer.zero_grad()
        # with autograd.detect_anomaly():
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model, 1)
        self._optimizer.step()
        return loss, vae, scc, dfp, mi

    def validate_step(self, x, y,  other):
        loss, vae, scc, dfp, mi, x_hat = self.compute_loss(x, y, other)
        return loss, vae, scc, dfp, mi, x_hat.mean.transpose(0, 1).cpu()
