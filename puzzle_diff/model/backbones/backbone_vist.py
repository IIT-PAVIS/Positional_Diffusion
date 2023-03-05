import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GraphNorm
from transformers import BartModel, BartTokenizer

from .Transformer_GNN import Transformer_GNN


class Eff_GAT_Vist(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, steps, input_channels=1, output_channels=1, model="facebook/bart-large"
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.tokenizer = BartTokenizer.from_pretrained(model)
        self.text_encoder = BartModel.from_pretrained(model)
        self.text_encoder.return_dict = True

        self.trans_features_dim = {
            "facebook/bart-base": 768,
            "facebook/bart-large": 1024,
        }[model]

        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            self.trans_features_dim,
            8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=False,
            device=None,
            dtype=None,
        )
        self.transformer_encode = torch.nn.TransformerEncoder(
            self.transformer_encoder_layer, 2
        )
        self.visual_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, features_only=True
        )
        visual_backbone_feat_dim = 4352 + 13056  # 128x128
        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.combined_features_dim = (
            {"facebook/bart-base": 768, "facebook/bart-large": 1024}[model]
            + visual_backbone_feat_dim
            + 32
            + 32
        )

        # self.gnn_backbone = torch_geometric.nn.models.GAT(
        #     in_channels=self.combined_features_dim,
        #     hidden_channels=256,
        #     num_layers=2,
        #     out_channels=self.combined_features_dim,
        # )

        self.gnn_backbone = Transformer_GNN(
            self.combined_features_dim,
            hidden_dim=32 * 8,
            heads=8,
            output_size=self.combined_features_dim,
        )
        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )
        # self.GN = GraphNorm(self.combined_features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )
        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, xy_pos, time, sentences, frames, edge_index, batch):

        text_feats = self.text_features(frames)
        patch_feats = self.visual_features(sentences)
        final_feats = self.forward_with_feats(
            xy_pos,
            time,
            edge_index,
            frames_feats=patch_feats,
            text_feats=text_feats,
            batch=batch,
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        edge_index: Tensor,
        text_feats: Tensor,
        frames_feats: Tensor,
        batch,
    ):
        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32

        # COMBINE  and transform with MLP
        combined_feats = torch.cat(
            [text_feats, frames_feats, pos_feats, time_feats], -1
        )
        combined_feats = self.mlp(combined_feats)

        # GNN
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)

        # Residual + final transform
        final_feats = self.final_mlp(
            feats + combined_feats
        )  # combined -> (err_x, err_y)

        return final_feats

    def text_features(self, patch_rgb):
        with torch.no_grad():
            phrases = [y for x in patch_rgb for y in x]
            tokens = self.tokenizer(phrases, return_tensors="pt", padding=True).to(
                self.text_encoder.device
            )
            text_emb = self.text_encoder(**tokens)["last_hidden_state"]

        attn_mask = (1 - tokens["attention_mask"]).bool()
        feats = self.transformer_encode(text_emb, src_key_padding_mask=attn_mask)

        return feats[:, 0, :]

    def visual_features(self, frames):
        frames = (frames - self.mean) / self.std

        feats = self.visual_backbone.forward(frames)

        patch_feats = torch.cat(
            [
                feats[2].reshape(frames.shape[0], -1),  # 10240
                feats[3].reshape(frames.shape[0], -1),
            ],
            -1,
        )

        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        return patch_feats
