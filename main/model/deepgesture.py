import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from local_attention.local_attention import LocalAttention


class DeepGesture(nn.Module):
    def __init__(self, modeltype, njoints, nfeats,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False,
                 audio_feat='',
                 text_feat='',
                 n_seed=1, cond_mode='', **kargs):
        super().__init__()
        # ~~~~~~~~~~~~~~~~~ configuration ~~~~~~~~~~~~~~~~~
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0

        # --- Audio
        self.audio_feat = audio_feat
        if audio_feat == 'wav encoder':
            self.audio_feat_dim = 32
        elif audio_feat == 'mfcc':
            self.audio_feat_dim = 13
        elif self.audio_feat == 'wavlm':
            print('USE WAVLM')
            self.audio_feat_dim = 64  # Linear 1024 -> 64
            self.speech_linear_encoder = WavEncoder()

        # --- Text
        self.text_feat = text_feat
        if text_feat == 'word2vec':
            self.text_feat_dim = 64  # Linear 1024 -> 64
            self.text_linear_encoder = TextEncoder()

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.cond_mode = cond_mode
        self.num_head = 8

        # ~~~~~~~~~~~~~ Feature Encode -> Feature Concat -> Feature Correlation -> Feature Decode
        # Feature Encoding
        if 'style2' not in self.cond_mode:
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)

        if self.arch == 'mytrans_enc':
            print("MY TRANS_ENC init")
            from mytransformer import TransformerEncoderLayer, TransformerEncoder

            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(1536, self.latent_dim)

            sequence_trans_encoder_layer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                                   nhead=self.num_heads,
                                                                   dim_feedforward=self.ff_size,
                                                                   dropout=self.dropout,
                                                                   activation=self.activation)
            self.seqTransEncoder = TransformerEncoder(sequence_trans_encoder_layer, num_layers=self.num_layers)

        elif self.arch == 'trans_enc':
            print("TRANS_ENC init")
            sequence_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                      nhead=self.num_heads,
                                                                      dim_feedforward=self.ff_size,
                                                                      dropout=self.dropout,
                                                                      activation=self.activation, batch_first=True)

            self.seqTransEncoder = nn.TransformerEncoder(sequence_trans_encoder_layer, num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            sequence_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                                      nhead=self.num_heads,
                                                                      dim_feedforward=self.ff_size,
                                                                      dropout=self.dropout,
                                                                      activation=activation, batch_first=True)
            self.seqTransDecoder = nn.TransformerDecoder(sequence_trans_decoder_layer, num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=False)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        if 'style1' in self.cond_mode:
            print('EMBED STYLE BEGIN TOKEN')
            if self.n_seed != 0:
                self.style_dim = 64
                self.style_linear_encoder = nn.Linear(6, self.style_dim)
                self.seed_gesture_linear = nn.Linear(self.njoints * n_seed, self.latent_dim - self.style_dim)
            else:
                self.style_dim = self.latent_dim
                self.style_linear_encoder = nn.Linear(6, self.style_dim)

        elif 'style2' in self.cond_mode:
            print('EMBED STYLE ALL FRAMES')
            self.style_dim = 64
            self.style_linear_encoder = nn.Linear(6, self.style_dim)
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.audio_feat_dim + self.gru_emb_dim + self.style_dim,
                                              self.latent_dim)
            if self.n_seed != 0:
                self.seed_gesture_linear = nn.Linear(self.njoints * n_seed, self.latent_dim)
        elif self.n_seed != 0:
            self.seed_gesture_linear = nn.Linear(self.njoints * n_seed, self.latent_dim)

        # Feature Decoding
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        if 'cross_local_attention' in self.cond_mode:
            self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)
            self.cross_local_attention = LocalAttention(
                dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=11,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,  # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2 = nn.Linear(self.latent_dim * 2 + self.audio_feat_dim, self.latent_dim)

        if 'cross_local_attention2' in self.cond_mode:
            print('Cross Local Attention2')
            self.selfAttention = LinearTemporalCrossAttention(seq_len=0, latent_dim=256, text_latent_dim=256, num_head=8, dropout=0.1, time_embed_dim=0)

        if 'cross_local_attention3' in self.cond_mode:
            print('Cross Local Attention3')

        if 'cross_local_attention4' in self.cond_mode:
            print('Cross Local Attention4')

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None, uncond_info=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        y:{
            seed: [batch_size, njoints, nfeats],
            ...
        }
        """

        bs, njoints, nfeats, nframes = x.shape  # 64, 251, 1, 196
        latent_timestep = self.embed_timestep(timesteps)  # [1, batch_size, d], (1, 2, 256)

        # force_mask = y.get('uncond', False)  # False
        force_mask = uncond_info

        # ~~~ Seed Gesture Embedding
        if 'style1' in self.cond_mode:
            embed_style = self.mask_cond(self.style_linear_encoder(y['style']), force_mask=force_mask)  # (batch_size, 64)
            if self.n_seed != 0:
                latent_seed_gesture = self.seed_gesture_linear(self.mask_cond(y['seed'].squeeze(2).reshape(bs, -1), force_mask=force_mask))  # (batch_size, 256-64)
                emb_1 = torch.cat((embed_style, latent_seed_gesture), dim=1)
            else:
                emb_1 = embed_style
        elif self.n_seed != 0:
            emb_1 = self.seed_gesture_linear(self.mask_cond(y['seed'].squeeze(2).reshape(bs, -1), force_mask=force_mask))  # z_tk

        # ~~~ Text Embedding
        if self.text_feat == "word2vec":
            word_embedding = self.text_linear_encoder(y['text']).permute(1, 0, 2)
        else:
            word_embedding = y['text'].permute(1, 0, 2)

        # ~~~ Speech Embedding
        if self.audio_feat == 'wavlm':
            enc_audio = (self.speech_linear_encoder(y['audio'])).permute(1, 0, 2)
            enc_text = enc_audio + word_embedding
        else:
            enc_audio = y['audio'].permute(1, 0, 2)
            enc_text = enc_audio + word_embedding

        # ~~~ Latent Feature Attention
        if 'cross_local_attention' in self.cond_mode:
            if 'cross_local_attention3' in self.cond_mode:
                x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [batch_size, gesture_dim, 1, n_frame]
                # self-attention
                x_ = self.input_process(x)  # [batch_size, 135, 1, 240] -> [240, batch_size, 256]

                # local-cross-attention
                packed_shape = [torch.Size([bs, self.num_head])]
                xseq = torch.cat((x_, enc_text), axis=2)  # [batch_size, d+joints*feat, 1, #frames], (n_frame, 2, 32)

                # all frames
                embed_style_2 = (emb_1 + latent_timestep).repeat(nframes, 1, 1)  # (batch_size, 64) -> (len, batch_size, 64)
                xseq = torch.cat((embed_style_2, xseq), axis=2)  # (seq, batch_size, dim)
                xseq = self.input_process2(xseq)
                xseq = xseq.permute(1, 0, 2)  # (batch_size, len, dim)
                xseq = xseq.view(bs, nframes, self.num_head, -1)
                xseq = xseq.permute(0, 2, 1, 3)  # Need (batch_size, 8, 2048, 64)
                xseq = xseq.reshape(bs * self.num_head, nframes, -1)
                pos_emb = self.rel_pos(xseq)  # (89, 32)
                xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
                                                  mask=y['mask_local'])  # (batch_size, 8, 2048, 64)
                xseq = xseq.permute(0, 2, 1, 3)  # (batch_size, len, 8, 64)
                xseq = xseq.reshape(bs, nframes, -1)
                xseq = xseq.permute(1, 0, 2)

                xseq = torch.cat((emb_1 + latent_timestep, xseq), axis=0)  # [seqlen+1, batch_size, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                xseq = xseq.permute(1, 0, 2)  # (batch_size, len, dim)
                xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
                xseq = xseq.permute(0, 2, 1, 3)  # Need (batch_size, 8, 2048, 64)
                xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
                pos_emb = self.rel_pos(xseq)  # (89, 32)
                xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
                xseq = xseq_rpe.permute(0, 2, 1, 3)  # [seqlen+1, batch_size, d]
                xseq = xseq.view(bs, nframes + 1, -1)
                xseq = xseq.permute(1, 0, 2)
                if 'cross_local_attention2' in self.cond_mode:
                    xseq = (self.selfAttention(xseq).permute(1, 0, 2))[1:]
                else:
                    output = self.seqTransEncoder(xseq)[1:]

            elif 'cross_local_attention5' in self.cond_mode:
                x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [batch_size, 135, 1, n_frame]
                # self-attention
                x_ = self.input_process(x)  # [batch_size, 135, 1, n_frame] -> [n_frame, batch_size, 256]

                # local-cross-attention
                packed_shape = [torch.Size([bs, self.num_head])]
                xseq = torch.cat((x_, enc_text), axis=2)  # [batch_size, d+joints*feat, 1, frames], (240, 2, 32)
                # all frames
                embed_style_2 = (emb_1 + latent_timestep).repeat(nframes, 1, 1)  # (batch_size, 64) -> (len, batch_size, 64)
                xseq = torch.cat((embed_style_2, xseq), axis=2)  # (seq, batch_size, dim)
                xseq = self.input_process2(xseq)
                xseq = xseq.permute(1, 0, 2)  # (batch_size, len, dim)
                xseq = xseq.view(bs, nframes, self.num_head, -1)
                xseq = xseq.permute(0, 2, 1, 3)  # Need (batch_size, 8, 2048, 64)
                xseq = xseq.reshape(bs * self.num_head, nframes, -1)
                pos_emb = self.rel_pos(xseq)  # (89, 32)
                xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
                                                  mask=y['mask_local'])  # (batch_size, 8, 2048, 64)
                xseq = xseq.permute(0, 2, 1, 3)  # (batch_size, len, 8, 64)
                xseq = xseq.reshape(bs, nframes, -1)
                output = xseq.permute(1, 0, 2)

            else:
                x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [batch_size, 135, 1, 240]
                # self-attention
                x_ = self.input_process(x)  # [batch_size, 135, 1, 240] -> [240, 2, 256]
                xseq = torch.cat((emb_1 + latent_timestep, x_), axis=0)  # [seqlen+1, batch_size, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                xseq = xseq.permute(1, 0, 2)  # (batch_size, len, dim)
                xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
                xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
                pos_emb = self.rel_pos(xseq)  # (89, 32)
                xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
                xseq = xseq_rpe.permute(0, 2, 1, 3)  # [seqlen+1, batch_size, d]
                xseq = xseq.view(bs, nframes + 1, -1)
                xseq = xseq.permute(1, 0, 2)
                if 'cross_local_attention2' in self.cond_mode:
                    xseq = (self.selfAttention(xseq).permute(1, 0, 2))[1:]
                else:
                    xseq = self.seqTransEncoder(xseq)[1:]

                # local-cross-attention
                packed_shape = [torch.Size([bs, self.num_head])]
                xseq = torch.cat((xseq, enc_text), axis=2)  # [batch_size, d+joints*feat, 1, #frames], (n_frame, 2, 32)
                # all frames
                embed_style_2 = (emb_1 + latent_timestep).repeat(nframes, 1, 1)  # (batch_size, 64) -> (len, batch_size, 64)
                xseq = torch.cat((embed_style_2, xseq), axis=2)  # (seq, batch_size, dim)
                xseq = self.input_process2(xseq)
                xseq = xseq.permute(1, 0, 2)  # (batch_size, len, dim)
                xseq = xseq.view(bs, nframes, self.num_head, -1)
                xseq = xseq.permute(0, 2, 1, 3)  # Need (batch_size, 8, 2048, 64)
                xseq = xseq.reshape(bs * self.num_head, nframes, -1)
                pos_emb = self.rel_pos(xseq)  # (89, 32)
                xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape, mask=y['mask_local'])  # (batch_size, 8, 2048, 64)
                xseq = xseq.permute(0, 2, 1, 3)  # (batch_size, len, 8, 64)
                xseq = xseq.reshape(bs, nframes, -1)
                output = xseq.permute(1, 0, 2)

        else:
            if self.arch == 'trans_enc' or self.arch == 'trans_dec' or self.arch == 'conformers_enc' or self.arch == 'mytrans_enc':
                x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)  # [2, 135, 1, n_frame]
                enc_text_gru = enc_text.permute(1, 2, 0)  # (n_frame, 2, 32) -> (2, 32, n_frame)
                enc_text_gru = enc_text_gru.reshape(bs, self.audio_feat_dim, 1, nframes)
                x = torch.cat((x_reshaped, enc_text_gru), axis=1)  # [batch_size, d+joints*feat, 1, #frames]
                if 'style2' in self.cond_mode:
                    embed_style = self.mask_cond(self.style_linear_encoder(y['style']), force_mask=force_mask).repeat(nframes, 1, 1)  # (#frames, batch_size, 64)
                    embed_style = embed_style.unsqueeze(2)
                    embed_style = embed_style.permute(1, 3, 2, 0)
                    x = torch.cat((x, embed_style), axis=1)  # [batch_size, d+joints*feat, 1, #frames]

            if self.arch == 'gru':
                x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)  # [2, 135, 1, n_frame]
                emb_gru = enc_text.repeat(nframes, 1, 1)  # [#frames, batch_size, d]

                enc_text_gru = enc_text.permute(1, 2, 0)  # (n_frame, 2, 32) -> (2, 32, n_frame)
                enc_text_gru = enc_text_gru.reshape(bs, self.audio_feat_dim, 1, nframes)

                emb_gru = emb_gru.permute(1, 2, 0)  # [batch_size, d, #frames]
                emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  # [batch_size, d, 1, #frames]
                x = torch.cat((x_reshaped, emb_gru, enc_text_gru), axis=1)  # [batch_size, d+joints*feat, 1, #frames]

            x = self.input_process(x)  # [2, 135, 1, 240] -> [240, 2, 224]

            if self.arch == 'trans_enc':
                # adding the timestep embed
                # x = torch.cat((x, enc_text), axis=2)        # [[n_frame, 2, 224], (n_frame, 2, 32)] -> (n_frame, 2, 256)
                xseq = torch.cat((enc_text, x), axis=0)  # [seqlen+1, batch_size, d]     # [(1, 2, 256), (n_frame, 2, 256)] -> (241, 2, 256)

                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, batch_size, d]
                output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, batch_size, d]      # -> (n_frame, 2, 256)

            elif self.arch == 'trans_dec':
                if self.emb_trans_dec:
                    xseq = torch.cat((enc_text, x), axis=0)
                else:
                    xseq = x
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, batch_size, d]
                if self.emb_trans_dec:
                    output = self.seqTransDecoder(tgt=xseq, memory=enc_text)[1:]  # [seqlen, batch_size, d] # FIXME - maybe add a causal mask
                else:
                    output = self.seqTransDecoder(tgt=xseq, memory=enc_text)

            elif self.arch == 'gru':
                xseq = x
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen, batch_size, d]
                # pdb.set_trace()
                output, _ = self.gru(xseq)

            elif self.arch == 'mytrans_enc':
                # adding the timestep embed
                # x = torch.cat((x, enc_text), axis=2)        # [[240, 2, 224], (240, 2, 32)] -> (240, 2, 256)
                xseq = torch.cat((enc_text, x), axis=0)  # [seqlen+1, batch_size, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)

                sinusoidal_pos = self.embed_positions(xseq.shape[0], 0)[None, None, :, :].chunk(2, dim=-1)
                xseq = self.apply_rotary(xseq.permute(1, 0, 2), sinusoidal_pos).squeeze(0).permute(1, 0, 2)

                output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, batch_size, d]      # -> (240, 2, 256)

        # ~~~ Feature Decoder
        output = self.output_process(output)  # [batch_size, njoints, nfeats, nframes]
        return output

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]

        # For rotating queries/keys: The result of the operation can directly concatenate
        # since a matrix multiplication will sum across the last dimension.
        # e.g., torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
            self, num_positions: int, embedding_dim: int
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, batch_size, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, batch_size, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, batch_size, d]
            vel = x[1:]  # [seqlen-1, batch_size, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, batch_size, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, batch_size, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, batch_size, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, batch_size, d]
            first_pose = self.poseFinal(first_pose)  # [1, batch_size, 150]
            vel = output[1:]  # [seqlen-1, batch_size, d]
            vel = self.velFinal(vel)  # [seqlen-1, batch_size, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, batch_size, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [batch_size, njoints, nfeats, nframes]
        return output


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, xf=None, emb=None):
        """
        x: B, T, D      , [240, 2, 256]
        xf: B, N, L     , [1, 2, 256]
        """
        x = x.permute(1, 0, 2)
        # xf = xf.permute(1, 0, 2)
        B, T, D = x.shape
        # N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(x)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        return y


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_map = nn.Linear(1024, 64)

    def forward(self, wav_feature):
        wav_feature = self.audio_feature_map(wav_feature)
        return wav_feature


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_feature_map = nn.Linear(300, 64)

    def forward(self, text_feat):
        text_feat = self.text_feature_map(text_feat)
        return text_feat


if __name__ == '__main__':
    """
    cd ./main/ohgesture
    python deepgesture.py
    """

    n_frames = 88
    n_seed = 8
    text_dim = 300
    audio_dim = 1024
    batch_size = 64
    joints_feature = 1141
    latent_dim = 256

    device = torch.device('mps')

    # arch=mytrans_enc cross_local_attention5_style1 mfcc mytrans_enc trans_enc
    model = DeepGesture(modeltype='', njoints=joints_feature, nfeats=1,
                        cond_mode='cross_local_attention3_style1', action_emb='tensor',
                        audio_feat='wavlm',
                        text_feat='word2vec',
                        arch='trans_enc', latent_dim=latent_dim,
                        n_seed=n_seed, cond_mask_prob=0.1)

    # batch_size, njoints, nfeats, max_frames
    x = torch.randn(batch_size, joints_feature, 1, n_frames)
    t = torch.randint(low=1, high=1000, size=[batch_size])
    print("time_step: ", t.shape)

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([batch_size, 1, 1, n_frames]) < 1)  # [..., n_seed:]
    # mfcc
    # model_kwargs_['y']['audio'] = torch.randn(batch_size, n_frames, 13)  # [n_seed:, ...]
    # wavlm
    model_kwargs_['y']['audio'] = torch.randn(batch_size, n_frames, audio_dim)  # [n_seed:, ...]
    # model_kwargs_['y']['audio'] = wav2wavlm(args, wavlm_model, model_kwargs_['y']['audio'].transpose(0, 1), device)
    model_kwargs_['y']['style'] = torch.randn(batch_size, 6)
    model_kwargs_['y']['text'] = torch.randn(batch_size, n_frames, text_dim)
    model_kwargs_['y']['mask_local'] = torch.ones(batch_size, n_frames).bool()
    model_kwargs_['y']['seed'] = x[..., 0:n_seed]

    print("x: ", x.shape)
    print("audio: ", model_kwargs_['y']['audio'].shape)
    print("style: ", model_kwargs_['y']['style'].shape)
    print("text: ", model_kwargs_['y']['text'].shape)
    y = model(x, t, model_kwargs_['y'])
    print("y shape: ", y.shape)
