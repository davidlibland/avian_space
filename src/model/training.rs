//! Training-thread state and optimizer save/load.

use burn::{
    optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor},
    tensor::backend::AutodiffBackend,
};

use super::{HIDDEN_DIM, POLICY_OUTPUT_DIM, TrainBackend, VALUE_OUTPUT_DIM, net::RLNet};

/// All mutable training state owned by the background thread.
///
/// Fields are `Option<_>` so they can be temporarily moved out during
/// optimizer step (burn's `Optimizer::step` consumes the module).
#[allow(dead_code)]
pub struct RLInner<B: AutodiffBackend> {
    pub policy_net: Option<RLNet<B>>,
    pub value_net: Option<RLNet<B>>,
    pub policy_optim: OptimizerAdaptor<Adam, RLNet<B>, B>,
    pub value_optim: OptimizerAdaptor<Adam, RLNet<B>, B>,
}

#[allow(dead_code)]
impl<B: AutodiffBackend> RLInner<B> {
    pub fn new(device: &B::Device) -> Self {
        let adam = AdamConfig::new();
        Self {
            policy_net: Some(RLNet::new(device, HIDDEN_DIM, POLICY_OUTPUT_DIM)),
            value_net: Some(RLNet::new(device, HIDDEN_DIM, VALUE_OUTPUT_DIM)),
            policy_optim: adam.init::<B, RLNet<B>>(),
            value_optim: adam.init::<B, RLNet<B>>(),
        }
    }
}

/// Save an optimizer's state to `path` (`.bin` extension appended).
pub fn save_optimizer(
    optim: &OptimizerAdaptor<Adam, RLNet<TrainBackend>, TrainBackend>,
    path: &str,
) {
    use burn::optim::Optimizer;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let record = optim.to_record();
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::record(&recorder, record, ()) {
        Ok(bytes) => {
            let file = format!("{path}.bin");
            if let Err(e) = std::fs::write(&file, bytes) {
                eprintln!("[model] Failed to save optimizer to {file}: {e}");
            } else {
                println!("[model] Optimizer saved to {file}");
            }
        }
        Err(e) => eprintln!("[model] Failed to serialize optimizer: {e}"),
    }
}

/// Load an optimizer's state from `path`. Returns `None` on failure.
pub fn load_optimizer(
    path: &str,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> Option<OptimizerAdaptor<Adam, RLNet<TrainBackend>, TrainBackend>> {
    use burn::optim::Optimizer;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let file = format!("{path}.bin");
    let bytes = std::fs::read(&file).ok()?;
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::load(&recorder, bytes, device) {
        Ok(record) => {
            let optim = AdamConfig::new()
                .init::<TrainBackend, RLNet<TrainBackend>>()
                .load_record(record);
            println!("[model] Optimizer loaded from {file}");
            Some(optim)
        }
        Err(e) => {
            eprintln!("[model] Failed to load optimizer from {file}: {e}");
            None
        }
    }
}
