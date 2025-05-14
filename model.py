import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detr import Transformer, TransformerEncoder, TransformerEncoderLayer, Backbone, Joiner, PositionEmbeddingSine
from torch.autograd import Variable

class ACT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = build_transformer(config)
        self.encoder = build_encoder(config)

        backbones = []
        for _ in config["camera_names"]:
            backbone = build_backbone(config)
            backbones.append(backbone)
        self.backbones = nn.ModuleList(backbones)

        self.num_queries = config["chunk_size"]
        self.camera_names = config["camera_names"]
        self.kl_weight = config["kl_weight"]

        hidden_dim = self.transformer.d_model
        state_dim = config["state_dimension"]
        num_queries = config["chunk_size"]

        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.query_embedding = nn.Embedding(num_queries, hidden_dim)

        self.input_images_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)

        self.input_joint_proj = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embedding = nn.Embedding(1, hidden_dim)  # extra cls token embedding

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding

        self.latent_position = nn.Embedding(1, hidden_dim)
        self.current_joint_position = nn.Embedding(1, hidden_dim)

        if self.kl_weight != 0:
            self.encoder_qpos_chunk_proj = nn.Linear(state_dim, hidden_dim)  # project action to embedding
            self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)  # project hidden state to latent std, var
            self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))  # [CLS], qpos, a_seq

    def forward(self, current_joint, images, qpos_chunk=None):
        is_training = qpos_chunk is not None
        batch_size = current_joint.shape[0]

        # Obtain latent z from action sequence
        if is_training and self.kl_weight != 0:  # hidden_dim输入参数是512
            action_embedding = self.encoder_qpos_chunk_proj(qpos_chunk)  # (bs, seq, hidden_dim)

            current_joint_embedding = self.encoder_joint_proj(current_joint)  # (bs, hidden_dim)
            current_joint_embedding = torch.unsqueeze(current_joint_embedding, axis=1)  # (bs, 1, hidden_dim)

            cls_embedding = self.cls_embedding.weight  # (1, hidden_dim)
            cls_embedding = torch.unsqueeze(cls_embedding, axis=0).repeat(batch_size, 1, 1)  # (bs, 1, hidden_dim)

            encoder_input = torch.cat([cls_embedding, current_joint_embedding, action_embedding], axis=1)  # (bs, seq+1, hidden_dim)

            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)

            # obtain position embedding  合并位置编码
            pos_embedding = self.pos_table.clone().detach()
            pos_embedding = pos_embedding.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            encoder_output = self.encoder(encoder_input, pos=pos_embedding)
            encoder_output = encoder_output[0]  # take cls output only

            # 线性层  hidden_dim扩大到64
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([batch_size, self.latent_dim], dtype=torch.float32).to(current_joint.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_position = []
        for cam_id in range(len(self.camera_names)):
            features, src_position = self.backbones[cam_id](images[:, cam_id])

            features = features[0]  # take the last layer feature
            src_position = src_position[0]

            all_cam_features.append(self.input_images_proj(features))
            all_cam_position.append(src_position)
        all_cam_features = torch.cat(all_cam_features, axis=3)
        all_cam_position = torch.cat(all_cam_position, axis=3)

        # proprioception features
        current_joint_input = self.input_joint_proj(current_joint)
        current_joint_input = torch.unsqueeze(current_joint_input, axis=0)
        latent_input = torch.unsqueeze(latent_input, axis=0)

        hs = self.transformer(
            self.query_embedding.weight,
            all_cam_features,
            all_cam_position,
            None,
            current_joint_input,
            self.current_joint_position.weight,
            latent_input,
            self.latent_position.weight,
        )[0]
        predict_qpos_chunk = self.action_head(hs)

        if is_training:
            loss = F.l1_loss(qpos_chunk, predict_qpos_chunk, reduction="none").mean()

            if self.kl_weight != 0:
                total_kld, _, _ = kl_divergence(mu, logvar)
                loss += total_kld[0] * self.kl_weight

            return loss, predict_qpos_chunk
        else:  # inference time
            return predict_qpos_chunk
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)

def build_encoder(config):
    # 编码器的作用，将K个未来动作（主臂的）和当前的关节参数（应该也是主臂的）以及CLS一起送入Encoder，得到隐向量的平均值和方差。论文里面因为是关节角，就搭配主臂使用，如果是末端执行器，可能需要是从臂的参数了。

    encoder_layer = TransformerEncoderLayer(config["hidden_dimension"], config["head_num"], config["feed_forward_dimension"], config["dropout"], "relu", config["pre_normalization"])
    
    encoder_norm = nn.LayerNorm(config["hidden_dimension"]) if config["pre_normalization"] else None
    
    encoder = TransformerEncoder(encoder_layer, config["encoder_layer_num"], encoder_norm)

    return encoder

def build_transformer(config):
    return Transformer(
        d_model=config["hidden_dimension"],
        dropout=config["dropout"],
        nhead=config["head_num"],
        dim_feedforward=config["feed_forward_dimension"],
        num_encoder_layers=config["encoder_layer_num"],
        num_decoder_layers=config["decoder_layer_num"],
        normalize_before=config["pre_normalization"],
        return_intermediate_dec=True
    )

def build_backbone(config):
    position_embedding = build_position_encoding(config)
    train_backbone = config["backbone_learning_rate"] > 0
    return_interm_layers = config["masks"]
    backbone = Backbone(config["backbone"], train_backbone, return_interm_layers, config["dilation"])
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_position_encoding(config):
    return PositionEmbeddingSine(config["hidden_dimension"] // 2, normalize=True)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
