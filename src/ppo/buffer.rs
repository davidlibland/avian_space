//! Fixed-capacity, priority-ordered replay buffer for the value-function
//! auxiliary training.

use crate::consts::N_REWARD_TYPES;
use crate::model::{self, N_OBJECTS, OBJECT_INPUT_DIM, SELF_INPUT_DIM};

use super::batch::PpoBatch;

struct ReplayStep {
    self_feat: Vec<f32>,
    obj_feat: Vec<f32>,
    proj_feat: Vec<f32>,
    returns: [f32; N_REWARD_TYPES],
    priority: f32,
}

/// Steps are stored as a binary **min-heap** keyed by `priority` (root = lowest
/// priority). This makes "evict the lowest-priority step when full" an
/// `O(log n)` operation instead of an `O(n)` scan per insert, which matters a
/// lot once the buffer is large (e.g. 200k): a full scan per inserted step
/// would make each update spend many seconds just maintaining the buffer.
///
/// Heap order is irrelevant to `sample`, which draws uniformly at random by
/// index, so we get cheap eviction without affecting sampling semantics.
pub struct ValueReplayBuffer {
    steps: Vec<ReplayStep>,
    capacity: usize,
}

impl ValueReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            steps: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.steps[i].priority < self.steps[parent].priority {
                self.steps.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.steps.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut smallest = i;
            if l < n && self.steps[l].priority < self.steps[smallest].priority {
                smallest = l;
            }
            if r < n && self.steps[r].priority < self.steps[smallest].priority {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            self.steps.swap(i, smallest);
            i = smallest;
        }
    }

    /// Insert high-priority steps from a batch. Priority = max absolute advantage
    /// across heads; when at capacity, evicts the lowest-priority step (heap
    /// root) iff the incoming step ranks higher.
    pub fn insert_from_batch(
        &mut self,
        batch: &PpoBatch,
        head_advantages: &[[f32; N_REWARD_TYPES]],
        head_returns: &[[f32; N_REWARD_TYPES]],
    ) {
        for i in 0..batch.total_steps {
            let priority: f32 = head_advantages[i]
                .iter()
                .map(|a| a.abs())
                .fold(0.0_f32, f32::max);

            if priority < 1e-8 {
                continue;
            }

            // Skip cheaply if the buffer is full and this step can't beat the
            // current minimum — avoids constructing the (heavy) ReplayStep.
            if self.steps.len() >= self.capacity && priority <= self.steps[0].priority {
                continue;
            }

            let s_start = i * SELF_INPUT_DIM;
            let o_start = i * N_OBJECTS * OBJECT_INPUT_DIM;
            let p_start = i * model::N_PROJECTILE_SLOTS * model::PROJ_INPUT_DIM;

            let step = ReplayStep {
                self_feat: batch.self_flat[s_start..s_start + SELF_INPUT_DIM].to_vec(),
                obj_feat: batch.obj_flat[o_start..o_start + N_OBJECTS * OBJECT_INPUT_DIM].to_vec(),
                proj_feat: batch.proj_flat
                    [p_start..p_start + model::N_PROJECTILE_SLOTS * model::PROJ_INPUT_DIM]
                    .to_vec(),
                returns: head_returns[i],
                priority,
            };

            if self.steps.len() < self.capacity {
                self.steps.push(step);
                self.sift_up(self.steps.len() - 1);
            } else {
                // Replace the lowest-priority step (root) and restore heap order.
                self.steps[0] = step;
                self.sift_down(0);
            }
        }
    }

    /// Sample `n` steps uniformly at random, returned as flat contiguous arrays.
    pub fn sample(
        &self,
        n: usize,
        rng: &mut impl rand::Rng,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.steps.is_empty() || n == 0 {
            return None;
        }
        let mut self_flat = Vec::with_capacity(n * SELF_INPUT_DIM);
        let mut obj_flat = Vec::with_capacity(n * N_OBJECTS * OBJECT_INPUT_DIM);
        let mut proj_flat =
            Vec::with_capacity(n * model::N_PROJECTILE_SLOTS * model::PROJ_INPUT_DIM);
        let mut ret_flat = Vec::with_capacity(n * N_REWARD_TYPES);

        for _ in 0..n {
            let idx = rng.gen_range(0..self.steps.len());
            let step = &self.steps[idx];
            self_flat.extend_from_slice(&step.self_feat);
            obj_flat.extend_from_slice(&step.obj_feat);
            proj_flat.extend_from_slice(&step.proj_feat);
            ret_flat.extend_from_slice(&step.returns);
        }

        Some((self_flat, obj_flat, proj_flat, ret_flat))
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }
}
