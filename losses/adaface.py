"""
AdaFace Loss Implementation
Reproduces the official AdaFace behavior faithfully from AdaFace/head.py.

Formulas reproduced:
- kernel_norm = l2_norm(kernel, axis=0)
- cosine = torch.mm(embeddings, kernel_norm).clamp(-1+eps, 1-eps)
- EMA batch_mean = mean * t_alpha + (1 - t_alpha) * batch_mean
- EMA batch_std = std * t_alpha + (1 - t_alpha) * batch_std
- margin_scaler = (safe_norms - batch_mean) / (batch_std + eps)
- margin_scaler = torch.clip(margin_scaler * h, -1, 1)
- g_angular = m * margin_scaler * -1
- g_add = m + (m * margin_scaler)
- theta_m = torch.clip(theta + m_arc, eps, pi - eps)
- scaled_cosine_m = (theta_m.cos() - m_cos) * s
"""

import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class AdaFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=0.01,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        # t_alpha default is changed to 0.01 here because the official config.py 
        # uses 0.01 as the default for training, even though head.py defaults to 1.0.
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)

    def forward(self, embeddings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        if getattr(self, 'update_ema', True):
            with torch.no_grad():
                mean = safe_norms.mean().detach()
                std = safe_norms.std().detach()
                self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333, 0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        if getattr(self, 'force_q_zero', False):
            margin_scaler = torch.zeros_like(margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size(0), cosine.size(1), device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine_m = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size(0), cosine.size(1), device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine_m = cosine_m - m_cos

        # scale
        scaled_cosine_m = cosine_m * self.s
        return scaled_cosine_m, margin_scaler
