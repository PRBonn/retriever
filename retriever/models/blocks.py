import torch.nn as nn
import torch.nn.functional as F
import torch


class STNkd(nn.Module):
    def __init__(self, k=64, norm=True):
        super(STNkd, self).__init__()
        self.conv1 = nn.Linear(k, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        # exchanged Batchnorm1d by Layernorm
        self.bn1 = nn.LayerNorm(64) if norm else nn.Identity()
        self.bn2 = nn.LayerNorm(128) if norm else nn.Identity()
        self.bn3 = nn.LayerNorm(1024) if norm else nn.Identity()
        self.bn4 = nn.LayerNorm(512) if norm else nn.Identity()
        self.bn5 = nn.LayerNorm(256) if norm else nn.Identity()

        self.k = k

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, -2, keepdim=True)[0]
        # x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device, dtype=x.dtype)
        shape = x.shape[:-1]+(1,)
        iden = iden.repeat(*shape)
        x = x.view(iden.shape) + iden
        return x


class PointNetFeat(nn.Module):
    def __init__(self, in_dim=3, out_dim=1024, feature_transform=False, norm=True):
        super(PointNetFeat, self).__init__()
        self.stn = STNkd(k=in_dim, norm=norm)
        self.conv1 = nn.Linear(in_dim, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, out_dim)
        self.bn1 = nn.LayerNorm(64) if norm else nn.Identity()
        self.bn2 = nn.LayerNorm(128) if norm else nn.Identity()
        self.bn3 = nn.LayerNorm(out_dim) if norm else nn.Identity()
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64, norm=norm)

    def forward(self, x):
        trans = self.stn(x)
        x = torch.matmul(x, trans)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.matmul(x, trans_feat)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


######################################
######### Perceiver ##################
######################################

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        d_k_sq = q.shape[-1]**0.5
        w = (torch.matmul(q, k.transpose(-1, -2))/d_k_sq).softmax(dim=-1)
        f = torch.matmul(w, v)
        return f, w


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, out_dim=None, dropout=0., norm=False):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PerceiverBlock(nn.Module):
    def __init__(self,  latent_dim, inp_dim=None, num_heads=1, dropout=0):
        super().__init__()
        inp_dim = latent_dim if inp_dim is None else inp_dim
        self.to_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.to_kv = nn.Linear(inp_dim, latent_dim * 2, bias=False)

        self.norm = nn.LayerNorm(latent_dim)
        self.attn = Attention()
        self.ff = FeedForward(latent_dim, dropout=dropout)

    def forward(self, latents, inp=None):
        inp = latents if inp is None else inp
        q = self.to_q(latents)
        k, v = self.to_kv(inp).chunk(2, dim=-1)

        f, w = self.attn(q, k, v)

        latents = self.norm(latents+self.ff(f))
        return latents


class PerceiverVLAD(nn.Module):
    def __init__(self, num_latents, in_dim, out_dim):
        super().__init__()
        self.num_latents = num_latents
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(int(in_dim*num_latents), out_dim)

    def forward(self, x):
        # x = F.normalize(x, dim=-1, p=2)
        x2 = x.reshape(*x.shape[:-2], x.shape[-2]*x.shape[-1])
        # x2 = x.sum(dim=-2)
        out = self.proj(x2)
        # out = F.normalize(out, dim=-1, p=2)
        return out
