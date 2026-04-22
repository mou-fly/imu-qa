import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """单个 VQ 层"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: (batch_size, embedding_dim, L) -> (B, L, D)
        inputs = inputs.permute(0, 2, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 计算距离: d = x^2 + y^2 - 2xy
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # 编码: 寻找最近邻索引
        encoding_indices = torch.min(distances, dim=1)[1].unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # 损失函数 (Commitment loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized.permute(0, 2, 1).contiguous(), loss, encoding_indices

class ResidualVQ(nn.Module):
    """残差量化器：包含多层 VQ"""
    def __init__(self, num_embeddings, embedding_dim, num_quantizers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim) 
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []

        for vq in self.layers:
            quantized, loss, indices = vq(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_indices.append(indices)
            
        return quantized_out, sum(all_losses), all_indices