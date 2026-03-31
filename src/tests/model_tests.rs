use super::*;
use crate::rl_obs::OBS_DIM;

// ── split_obs ─────────────────────────────────────────────────────────────

#[test]
fn test_split_obs_sizes() {
    let obs = vec![0.0_f32; OBS_DIM];
    let (s, o) = split_obs(&obs);
    assert_eq!(s.len(), SELF_INPUT_DIM);
    assert_eq!(o.len(), N_OBJECTS * OBJECT_INPUT_DIM);
}

#[test]
fn test_split_obs_self_features() {
    let mut obs = vec![0.0_f32; OBS_DIM];
    for (i, v) in obs[0..SELF_SIZE].iter_mut().enumerate() {
        *v = (i + 1) as f32 * 0.1;
    }
    let (s, _) = split_obs(&obs);
    assert_eq!(s, &obs[0..SELF_SIZE]);
}

#[test]
fn test_split_obs_entity_features() {
    let mut obs = vec![0.0_f32; OBS_DIM];
    // Write a distinctive pattern into slot 0.
    for (i, v) in obs[SELF_SIZE..SELF_SIZE + SLOT_SIZE].iter_mut().enumerate() {
        *v = (i + 1) as f32 * 0.5;
    }
    let (_, o) = split_obs(&obs);
    assert_eq!(
        &o[0..SLOT_SIZE],
        &obs[SELF_SIZE..SELF_SIZE + SLOT_SIZE],
    );
}

// ── logits_to_discrete_action ────────────────────────────────────────────

#[test]
fn test_logits_to_discrete_action_argmax() {
    // Craft logits so each head has an obvious winner.
    let action_logits = [1.0_f32, 5.0, 0.0,   0.0, 3.0,   2.0, 0.0,   0.0, 4.0];
    // Target logits: slot 3 is the winner.
    let mut target_logits = vec![-1e9_f32; TARGET_OUTPUT_DIM];
    target_logits[3] = 5.0;
    let (turn, thrust, fp, fs, tgt) =
        logits_to_discrete_action(&action_logits, &target_logits);
    assert_eq!(turn, 1, "turn should be straight");
    assert_eq!(thrust, 1, "thrust should be on");
    assert_eq!(fp, 0, "fire_primary should be off");
    assert_eq!(fs, 1, "fire_secondary should be on");
    assert_eq!(tgt, 3, "target should be slot 3");
}

#[test]
fn test_logits_to_discrete_action_all_valid() {
    let action_logits = [0.0_f32; POLICY_OUTPUT_DIM];
    let target_logits = vec![0.0_f32; TARGET_OUTPUT_DIM];
    let (turn, thrust, fp, fs, tgt) =
        logits_to_discrete_action(&action_logits, &target_logits);
    assert!(turn <= 2);
    assert!(thrust <= 1);
    assert!(fp <= 1);
    assert!(fs <= 1);
    assert!((tgt as usize) < TARGET_OUTPUT_DIM);
}

// ── RLNet forward-pass shapes ────────────────────────────────────────────

/// Helper: create zero self + obj tensors for testing.
fn zero_inputs(batch: usize) -> (Tensor<InferBackend, 2>, Tensor<InferBackend, 3>) {
    let device = Default::default();
    let s = Tensor::<InferBackend, 2>::zeros([batch, SELF_INPUT_DIM], &device);
    let o = Tensor::<InferBackend, 3>::zeros([batch, N_OBJECTS, OBJECT_INPUT_DIM], &device);
    (s, o)
}

#[test]
fn test_rlnet_policy_output_shape() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
    let batch = 4usize;
    let (s, o) = zero_inputs(batch);
    let (action_out, target_out) = net.forward(s, o);
    assert_eq!(action_out.shape().dims, [batch, POLICY_OUTPUT_DIM]);
    assert_eq!(target_out.shape().dims, [batch, TARGET_OUTPUT_DIM]);
}

#[test]
fn test_rlnet_value_output_shape() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, VALUE_OUTPUT_DIM);
    let batch = 3usize;
    let (s, o) = zero_inputs(batch);
    let (action_out, target_out) = net.forward(s, o);
    assert_eq!(action_out.shape().dims, [batch, VALUE_OUTPUT_DIM]);
    assert_eq!(target_out.shape().dims, [batch, TARGET_OUTPUT_DIM]);
}

#[test]
fn test_rlnet_output_zero_init() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
    let (s, o) = zero_inputs(1);
    let (action_logits, _) = net.forward(s, o);
    let logits: Vec<f32> = action_logits
        .into_data()
        .into_vec::<f32>()
        .expect("extraction failed");
    for (i, &v) in logits.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "logit[{i}] should be ~0 on zero input, got {v}"
        );
    }
}

#[test]
fn test_rlnet_target_logits_masked() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
    let (s, o) = zero_inputs(1);
    let (_, target_logits) = net.forward(s, o);
    let tl: Vec<f32> = target_logits
        .into_data()
        .into_vec::<f32>()
        .expect("extraction failed");
    for i in 0..N_OBJECTS {
        assert!(
            tl[i] < -1e8,
            "empty slot {i} should be masked, got {}",
            tl[i]
        );
    }
    assert!(
        tl[N_OBJECTS] > -1e8,
        "no-target logit should not be masked, got {}",
        tl[N_OBJECTS]
    );
}

// ── InferenceNet end-to-end ──────────────────────────────────────────────

#[test]
fn test_inference_net_produces_valid_action() {
    let inference = InferenceNet::new();
    let obs = vec![0.0_f32; OBS_DIM];
    let (s, o) = split_obs(&obs);
    let (action_logits, target_logits) =
        inference.run_inference(s.to_vec(), o.to_vec(), 1);

    assert_eq!(action_logits.len(), POLICY_OUTPUT_DIM, "wrong action logit count");
    assert_eq!(target_logits.len(), TARGET_OUTPUT_DIM, "wrong target logit count");
    let (turn, thrust, fp, fs, tgt) =
        logits_to_discrete_action(&action_logits, &target_logits);
    assert!(turn <= 2, "turn_idx out of range: {turn}");
    assert!(thrust <= 1, "thrust_idx out of range: {thrust}");
    assert!(fp <= 1, "fire_primary out of range: {fp}");
    assert!(fs <= 1, "fire_secondary out of range: {fs}");
    assert!((tgt as usize) < TARGET_OUTPUT_DIM, "target_idx out of range: {tgt}");
}

#[test]
fn test_inference_net_batched() {
    let inference = InferenceNet::new();
    let batch_size = 5;
    let obs = vec![0.0_f32; OBS_DIM];
    let (s, o) = split_obs(&obs);
    let self_flat: Vec<f32> = s.iter().cloned().cycle().take(batch_size * s.len()).collect();
    let obj_flat: Vec<f32> = o.iter().cloned().cycle().take(batch_size * o.len()).collect();
    let (action_logits, target_logits) =
        inference.run_inference(self_flat, obj_flat, batch_size);
    assert_eq!(action_logits.len(), batch_size * POLICY_OUTPUT_DIM);
    assert_eq!(target_logits.len(), batch_size * TARGET_OUTPUT_DIM);
}
