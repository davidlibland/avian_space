//! Network architecture: `NetBlock`, multi-head attention, `RLNet`.

use burn::{
    module::{Initializer, Param},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{
        Tensor,
        activation::{silu, softmax},
    },
};

use crate::rl_obs::{
    CORE_BLOCK_START, CORE_FEAT_SIZE, K_PROJECTILES, N_ENTITY_TYPES, SHIP_IS_HOSTILE,
    SHIP_SHOULD_ENGAGE, SLOT_IS_PRESENT, SLOT_TYPE_ONEHOT, TYPE_BLOCK_SIZE, TYPE_BLOCK_START,
    TYPE_IDX_ASTEROID, TYPE_IDX_SHIP, TYPE_ONEHOT_SIZE,
};

use super::{N_OBJECTS, PROJ_INPUT_DIM, SELF_INPUT_DIM, USE_SKIP};

/// `LayerNorm → Linear(dim→dim, Xavier) → SiLU`, with residual connection.
#[derive(Module, Debug)]
pub struct NetBlock<B: Backend> {
    norm: LayerNorm<B>,
    fc: Linear<B>,
}

impl<B: Backend> NetBlock<B> {
    pub fn new(device: &B::Device, dim: usize) -> Self {
        Self {
            norm: LayerNormConfig::new(dim).init(device),
            fc: LinearConfig::new(dim, dim)
                .with_initializer(Initializer::XavierNormal { gain: 1.52 })
                .init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let residual = x.clone();
        let out = self.norm.forward(x);
        let out = self.fc.forward(out);
        let out = silu(out);
        if USE_SKIP { residual + out } else { out }
    }
}

type Block<B> = NetBlock<B>;

/// Number of attention heads for multi-head attention.
const N_HEADS: usize = 4;

/// Multi-head scaled dot-product attention.
fn multi_head_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_heads: usize,
) -> Tensor<B, 3> {
    let [b, nq, h] = q.dims();
    let [_, nk, _] = k.dims();
    let head_dim = h / n_heads;
    let scale = (head_dim as f64).sqrt() as f32;

    let q = q.reshape([b, nq, n_heads, head_dim]).swap_dims(1, 2);
    let k = k.reshape([b, nk, n_heads, head_dim]).swap_dims(1, 2);
    let v = v.reshape([b, nk, n_heads, head_dim]).swap_dims(1, 2);

    let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

    let scores = if let Some(m) = mask {
        let m = m.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
        let neg_inf = scores.zeros_like().add_scalar(-1e9);
        scores.clone() * m.clone() + neg_inf * (m.ones_like() - m)
    } else {
        scores
    };

    let weights = softmax(scores, 3);
    let out = weights.matmul(v);

    out.swap_dims(1, 2).reshape([b, nq, h])
}

/// Split a `[B, N, SLOT_SIZE]` object-feature tensor into its 4 blocks.
///
/// Returns `(type_onehot, is_present, core_feat, type_feat)`.
pub fn split_obj_feat<B: Backend>(
    obj_feat: Tensor<B, 3>,
) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
    let type_onehot = obj_feat
        .clone()
        .narrow(2, SLOT_TYPE_ONEHOT, TYPE_ONEHOT_SIZE);
    let is_present = obj_feat
        .clone()
        .narrow(2, SLOT_IS_PRESENT, 1)
        .squeeze_dim(2);
    let core_feat = obj_feat.clone().narrow(2, CORE_BLOCK_START, CORE_FEAT_SIZE);
    let type_feat = obj_feat.narrow(2, TYPE_BLOCK_START, TYPE_BLOCK_SIZE);
    (type_onehot, is_present, core_feat, type_feat)
}

/// Shared policy and value network.
#[derive(Module, Debug)]
pub struct RLNet<B: Backend> {
    #[module(skip)]
    hidden_dim: usize,

    core_embed: Linear<B>,
    type_embed_w: Param<Tensor<B, 3>>,
    type_embed_b: Param<Tensor<B, 2>>,
    eblock1: Block<B>,
    eblock2: Block<B>,
    enc_norm: LayerNorm<B>,

    proj_embed: Linear<B>,
    proj_block: Block<B>,
    proj_enc_norm: LayerNorm<B>,

    cross_q_proj: Linear<B>,
    cross_k_proj: Linear<B>,
    cross_v_proj: Linear<B>,
    cross_norm: LayerNorm<B>,

    self_embed: Linear<B>,
    self_fc2: Linear<B>,
    self_enc_norm: LayerNorm<B>,

    ent_q_proj: Linear<B>,
    ent_k_proj: Linear<B>,
    ent_v_proj: Linear<B>,

    pq_proj: Linear<B>,
    pk_proj: Linear<B>,
    pv_proj: Linear<B>,

    merge_proj: Linear<B>,
    merge_norm: LayerNorm<B>,
    dblock1: Block<B>,
    dblock2: Block<B>,
    norm: LayerNorm<B>,
    output: Linear<B>,

    nav_target_q_proj: Linear<B>,
    nav_target_k_proj: Linear<B>,
    nav_target_no_tgt: Linear<B>,
}

impl<B: Backend> RLNet<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, output_dim: usize) -> Self {
        let lin = |i, o| LinearConfig::new(i, o).init(device);
        let lin_xavier = |i, o| {
            LinearConfig::new(i, o)
                .with_initializer(Initializer::XavierNormal { gain: 1.0 })
                .init(device)
        };
        let lin_zero = |i, o| {
            LinearConfig::new(i, o)
                .with_initializer(Initializer::Zeros)
                .init(device)
        };

        let type_std = (2.0 / (TYPE_BLOCK_SIZE + hidden_dim) as f64).sqrt();
        let type_w = Tensor::<B, 3>::random(
            [N_ENTITY_TYPES, hidden_dim, TYPE_BLOCK_SIZE],
            burn::tensor::Distribution::Normal(0.0, type_std),
            device,
        );
        let type_b = Tensor::<B, 2>::zeros([N_ENTITY_TYPES, hidden_dim], device);

        Self {
            hidden_dim,

            core_embed: lin(CORE_FEAT_SIZE, hidden_dim),
            type_embed_w: Param::from_tensor(type_w),
            type_embed_b: Param::from_tensor(type_b),
            eblock1: Block::new(device, hidden_dim),
            eblock2: Block::new(device, hidden_dim),
            enc_norm: LayerNormConfig::new(hidden_dim).init(device),

            proj_embed: lin(PROJ_INPUT_DIM, hidden_dim),
            proj_block: Block::new(device, hidden_dim),
            proj_enc_norm: LayerNormConfig::new(hidden_dim).init(device),

            cross_q_proj: lin_xavier(hidden_dim, hidden_dim),
            cross_k_proj: lin_xavier(hidden_dim, hidden_dim),
            cross_v_proj: lin_xavier(hidden_dim, hidden_dim),
            cross_norm: LayerNormConfig::new(hidden_dim).init(device),

            self_embed: lin(SELF_INPUT_DIM, hidden_dim),
            self_fc2: lin(hidden_dim, hidden_dim),
            self_enc_norm: LayerNormConfig::new(hidden_dim).init(device),

            ent_q_proj: lin_xavier(hidden_dim, hidden_dim),
            ent_k_proj: lin_xavier(hidden_dim, hidden_dim),
            ent_v_proj: lin_xavier(hidden_dim, hidden_dim),

            pq_proj: lin_xavier(hidden_dim, hidden_dim),
            pk_proj: lin_xavier(hidden_dim, hidden_dim),
            pv_proj: lin_xavier(hidden_dim, hidden_dim),

            merge_proj: lin(hidden_dim * 3, hidden_dim),
            merge_norm: LayerNormConfig::new(hidden_dim).init(device),
            dblock1: Block::new(device, hidden_dim),
            dblock2: Block::new(device, hidden_dim),
            norm: LayerNormConfig::new(hidden_dim).init(device),
            output: lin_zero(hidden_dim, output_dim),

            nav_target_q_proj: lin_xavier(hidden_dim, hidden_dim),
            nav_target_k_proj: lin_xavier(hidden_dim, hidden_dim),
            nav_target_no_tgt: lin_zero(hidden_dim, 1),
        }
    }

    pub fn forward(
        &self,
        self_feat: Tensor<B, 2>,
        obj_feat: Tensor<B, 3>,
        proj_feat: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.hidden_dim;
        let (type_onehot, is_present, core_feat, type_feat) = split_obj_feat(obj_feat);

        let core_enc = self.core_embed.forward(core_feat);

        let ts = TYPE_BLOCK_SIZE;
        let w_flat = self.type_embed_w.val().reshape([N_ENTITY_TYPES, h * ts]);
        let w_blended = type_onehot.clone().matmul(w_flat.unsqueeze());
        let [batch_size, n_ents, _] = w_blended.dims();
        let w_blended = w_blended.reshape([batch_size, n_ents, h, ts]);
        let b_blended = type_onehot
            .clone()
            .matmul(self.type_embed_b.val().unsqueeze());
        let type_feat_col = type_feat.clone().unsqueeze_dim::<4>(3);
        let type_enc = w_blended.matmul(type_feat_col).squeeze_dim(3) + b_blended;

        let ent_enc = core_enc + type_enc;
        let ent_enc = self.eblock1.forward(ent_enc);
        let ent_enc = self.eblock2.forward(ent_enc);
        let ent_enc = self.enc_norm.forward(ent_enc);

        let proj_enc = silu(self.proj_embed.forward(proj_feat.clone()));
        let proj_enc = self.proj_block.forward(proj_enc);

        let proj_present: Tensor<B, 2> = proj_feat.clone().narrow(2, 0, 1).squeeze_dim::<2>(2);

        let scale = (h as f64).sqrt() as f32;

        let cross_q = self
            .cross_q_proj
            .forward(self.cross_norm.forward(proj_enc.clone()));
        let cross_k = self.cross_k_proj.forward(ent_enc.clone());
        let cross_v = self.cross_v_proj.forward(ent_enc.clone());

        let scores = cross_q.matmul(cross_k.swap_dims(1, 2)).div_scalar(scale);

        let ent_mask_3d = is_present
            .clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, K_PROJECTILES);
        let neg_inf_cross =
            Tensor::<B, 3>::full([batch_size, K_PROJECTILES, n_ents], -1e9, &scores.device());
        let scores = scores.clone() * ent_mask_3d.clone()
            + neg_inf_cross * (ent_mask_3d.ones_like() - ent_mask_3d);

        let cross_weights = softmax(scores, 2);
        let cross_out = cross_weights.matmul(cross_v);

        let proj_gate = proj_present.clone().unsqueeze_dim::<3>(2);
        let proj_enc = if USE_SKIP {
            proj_enc + cross_out * proj_gate
        } else {
            cross_out * proj_gate
        };
        let proj_enc = self.proj_enc_norm.forward(proj_enc);

        let self_enc = silu(self.self_embed.forward(self_feat));
        let self_enc = silu(self.self_fc2.forward(self_enc));
        let self_enc = self.self_enc_norm.forward(self_enc);

        let sq = self_enc.clone().unsqueeze_dim::<3>(1);
        let ent_q = self.ent_q_proj.forward(sq);
        let ent_k = self.ent_k_proj.forward(ent_enc.clone());
        let ent_v = self.ent_v_proj.forward(ent_enc.clone());
        let ent_attn = multi_head_attention(ent_q, ent_k, ent_v, Some(is_present.clone()), N_HEADS)
            .squeeze_dim(1);

        let pq = self.pq_proj.forward(self_enc.clone().unsqueeze_dim::<3>(1));
        let pk = self.pk_proj.forward(proj_enc.clone());
        let pv = self.pv_proj.forward(proj_enc);
        let proj_attn =
            multi_head_attention(pq, pk, pv, Some(proj_present), N_HEADS).squeeze_dim(1);

        let merged = Tensor::cat(vec![ent_attn, proj_attn, self_enc], 1);
        let x = silu(self.merge_norm.forward(self.merge_proj.forward(merged)));

        let x = self.dblock1.forward(x);
        let x = self.dblock2.forward(x);
        let x = self.norm.forward(x);
        let x = silu(x);
        let action_logits = self.output.forward(x.clone());

        let device = is_present.device();
        let ones = Tensor::<B, 2>::ones([batch_size, 1], &device);
        let mask = Tensor::cat(vec![is_present, ones], 1);
        let neg_inf = Tensor::<B, 2>::full([batch_size, N_OBJECTS + 1], -1e9, &device);

        let pointer_head = |q_proj: &Linear<B>,
                            k_proj: &Linear<B>,
                            no_tgt: &Linear<B>,
                            x_ref: &Tensor<B, 2>,
                            ent_ref: &Tensor<B, 3>|
         -> Tensor<B, 2> {
            let tq = q_proj.forward(x_ref.clone()).unsqueeze_dim::<3>(1);
            let tk = k_proj.forward(ent_ref.clone()).swap_dims(1, 2);
            let entity_logits = tq.matmul(tk).div_scalar(scale).squeeze_dim(1);
            let no_tgt_logit = no_tgt.forward(x_ref.clone());
            let logits = Tensor::cat(vec![entity_logits, no_tgt_logit], 1);
            logits.clone() * mask.clone() + neg_inf.clone() * (mask.ones_like() - mask.clone())
        };

        let nav_target_logits = pointer_head(
            &self.nav_target_q_proj,
            &self.nav_target_k_proj,
            &self.nav_target_no_tgt,
            &x,
            &ent_enc,
        );

        let is_asteroid = type_onehot
            .clone()
            .narrow(2, TYPE_IDX_ASTEROID, 1)
            .squeeze_dim(2);
        let is_ship = type_onehot
            .clone()
            .narrow(2, TYPE_IDX_SHIP, 1)
            .squeeze_dim(2);
        let type_specific_offset = 1;
        let is_hostile = type_feat
            .clone()
            .narrow(2, type_specific_offset + SHIP_IS_HOSTILE, 1)
            .squeeze_dim(2);
        let should_engage = type_feat
            .clone()
            .narrow(2, type_specific_offset + SHIP_SHOULD_ENGAGE, 1)
            .squeeze_dim(2);
        let valid_weapons_entity = is_asteroid
            + is_ship
                * (is_hostile.clamp(0.0, 1.0) + should_engage.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let valid_weapons_target = Tensor::cat(
            vec![valid_weapons_entity, Tensor::<B, 2>::ones([batch_size, 1], &device)],
            1,
        );
        let weapons_target_logits = nav_target_logits.clone() * valid_weapons_target.clone()
            + neg_inf.clone() * (valid_weapons_target.ones_like() - valid_weapons_target);

        (action_logits, nav_target_logits, weapons_target_logits)
    }
}

/// Serialize a net to bytes for cross-thread / cross-backend weight transfer.
#[allow(dead_code)]
pub fn net_to_bytes<B: Backend>(net: RLNet<B>) -> Vec<u8> {
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    Recorder::<B>::record(&recorder, net.into_record(), ())
        .expect("failed to serialize net to bytes")
}
