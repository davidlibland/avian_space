//! PPO (Proximal Policy Optimization) training thread.
//!
//! Receives [`Segment`](crate::rl_collection::Segment)s from the game thread,
//! trains both policy and value networks, and periodically syncs updated
//! policy weights back to the inference net used by the game thread.

mod batch;
mod buffer;
mod loss;
mod persistence;
mod train;

pub use train::spawn_ppo_training_thread;
