//! Game-thread inference wrapper and checkpoint save/load helpers.

use burn::{prelude::*, tensor::{Tensor, TensorData}};

use crate::rl_obs::K_PROJECTILES;

use super::{
    HIDDEN_DIM, InferBackend, N_OBJECTS, OBJECT_INPUT_DIM, POLICY_OUTPUT_DIM, PROJ_INPUT_DIM,
    SELF_INPUT_DIM, TrainBackend,
    net::{RLNet, net_to_bytes},
};

/// Wraps a policy `RLNet` for synchronous game-thread inference.
///
/// Stored inside `RLResource` behind an `Arc<Mutex<_>>` so the training
/// thread can push updated weights while the game thread reads them.
pub struct InferenceNet {
    net: RLNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
}

impl InferenceNet {
    pub fn new() -> Self {
        let device = Default::default();
        Self {
            net: RLNet::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM),
            device,
        }
    }

    /// Batched forward pass. Returns `(action_logits, nav_target_logits, weapons_target_logits)`.
    pub fn run_inference(
        &self,
        self_flat: Vec<f32>,
        obj_flat: Vec<f32>,
        proj_flat: Vec<f32>,
        batch_size: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let self_input = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(self_flat, [batch_size, SELF_INPUT_DIM]),
            &self.device,
        );
        let obj_input = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(obj_flat, [batch_size, N_OBJECTS, OBJECT_INPUT_DIM]),
            &self.device,
        );
        let proj_input = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(proj_flat, [batch_size, K_PROJECTILES, PROJ_INPUT_DIM]),
            &self.device,
        );
        let (action, nav_tgt, wep_tgt) = self.net.forward(self_input, obj_input, proj_input);
        let action_logits = action
            .into_data()
            .into_vec::<f32>()
            .expect("action logit extraction failed");
        let nav_target_logits = nav_tgt
            .into_data()
            .into_vec::<f32>()
            .expect("nav target logit extraction failed");
        let wep_target_logits = wep_tgt
            .into_data()
            .into_vec::<f32>()
            .expect("weapons target logit extraction failed");
        (action_logits, nav_target_logits, wep_target_logits)
    }

    /// Serialize weights to bytes (compatible with [`Self::load_bytes`]).
    pub fn to_bytes(&self) -> Vec<u8> {
        net_to_bytes(self.net.clone())
    }

    /// Replace weights from a byte buffer produced by [`net_to_bytes`].
    pub fn load_bytes(&mut self, bytes: Vec<u8>) {
        use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = Recorder::<InferBackend>::load(&recorder, bytes, &self.device)
            .expect("failed to deserialize weights");
        self.net = RLNet::new(&self.device, HIDDEN_DIM, POLICY_OUTPUT_DIM).load_record(record);
    }
}

/// Load inference network weights from `path`, returning `None` on failure.
pub fn load_inference_net(path: &str) -> Option<InferenceNet> {
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let device: <InferBackend as Backend>::Device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    match RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM)
        .load_file(path, &recorder, &device)
    {
        Ok(net) => {
            println!("[model] Loaded policy net from {path}");
            Some(InferenceNet { net, device })
        }
        Err(e) => {
            eprintln!("[model] Failed to load policy net from {path}: {e}");
            None
        }
    }
}

/// Save a `TrainBackend` net to disk as an inference-compatible checkpoint.
pub fn save_training_net(net: &RLNet<TrainBackend>, path: &str) {
    let bytes = net_to_bytes(net.clone());
    let file = format!("{path}.bin");
    if let Err(e) = std::fs::write(&file, bytes) {
        eprintln!("[model] Failed to save checkpoint to {file}: {e}");
    } else {
        println!("[model] Checkpoint saved to {file}");
    }
}

/// Load a checkpoint from `path` into a `TrainBackend` net.
pub fn load_training_net(
    path: &str,
    device: &<TrainBackend as Backend>::Device,
) -> Option<RLNet<TrainBackend>> {
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let infer = load_inference_net(path)?;
    let bytes = infer.to_bytes();
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::load(&rec, bytes, device) {
        Ok(record) => {
            let net = RLNet::<TrainBackend>::new(device, HIDDEN_DIM, POLICY_OUTPUT_DIM)
                .load_record(record);
            println!("[model] Loaded training net from {path}");
            Some(net)
        }
        Err(e) => {
            eprintln!("[model] Failed to deserialise training net from {path}: {e}");
            None
        }
    }
}

/// Like [`load_training_net`] but with an explicit `output_dim` (e.g. for the value network).
pub fn load_training_net_with_dim(
    path: &str,
    device: &<TrainBackend as Backend>::Device,
    output_dim: usize,
) -> Option<RLNet<TrainBackend>> {
    use burn::record::{BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, Recorder};
    let infer_device: <InferBackend as Backend>::Device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let infer_net = match RLNet::<InferBackend>::new(&infer_device, HIDDEN_DIM, output_dim)
        .load_file(path, &recorder, &infer_device)
    {
        Ok(net) => net,
        Err(e) => {
            eprintln!("[model] Failed to load net (dim={output_dim}) from {path}: {e}");
            return None;
        }
    };
    let bytes = net_to_bytes(infer_net);
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::load(&rec, bytes, device) {
        Ok(record) => {
            let net =
                RLNet::<TrainBackend>::new(device, HIDDEN_DIM, output_dim).load_record(record);
            println!("[model] Loaded training net (dim={output_dim}) from {path}");
            Some(net)
        }
        Err(e) => {
            eprintln!("[model] Failed to deserialise training net from {path}: {e}");
            None
        }
    }
}

/// Serialize a `TrainBackend` net to bytes via `net_to_bytes`.
pub fn training_net_to_bytes(net: &RLNet<TrainBackend>) -> Vec<u8> {
    net_to_bytes(net.clone())
}
