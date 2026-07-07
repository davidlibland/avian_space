//! Shared character animation for the planet surface.
//!
//! Two sheet layouts share one animation system:
//!
//! - **People** (player + NPCs, composited by `character_compositor`):
//!   18 cols × 4 rows of 32×32 — per facing row: 2 idle (breathing) +
//!   8 walk cycle + 8 run cycle. Gait is chosen from speed.
//! - **Legacy RPG** (fauna sheets): 3 cols (still, w1, w2) × 4 rows with the
//!   classic still→w1→still→w2 ping-pong.
//!
//! [`CharacterAnim`] carries the layout as frame *sequences* (atlas column
//! indices), so [`animate_characters`] is agnostic: velocity → facing →
//! gait → next frame → atlas index. Control logic (keyboard vs pathing)
//! stays in the respective modules; both just set `LinearVelocity`.

use avian2d::prelude::*;
use bevy::prelude::*;

// ── Constants ────────────────────────────────────────────────────────────

/// People sheet: columns per facing row (2 idle + 8 walk + 8 run).
pub const PERSON_COLS: usize = 18;

/// Velocity magnitude below which a character is considered stationary.
pub const MOVE_THRESHOLD: f32 = 5.0;

/// Speed at which a person switches from the walk cycle to the run cycle.
/// (NPC civilians walk at 40; the player walks at 70 and runs at 120 while
/// holding Shift — terrain slowdowns naturally drop a runner back to a walk.)
pub const RUN_THRESHOLD: f32 = 80.0;

// ── Facing ───────────────────────────────────────────────────────────────

/// Direction a character is facing.  Row index in the sprite sheet.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum Facing {
    #[default]
    Down = 0,
    Left = 1,
    Right = 2,
    Up = 3,
}

impl Facing {
    /// Choose facing from a velocity vector.
    pub fn from_velocity(v: Vec2) -> Self {
        if v.x.abs() > v.y.abs() {
            if v.x > 0.0 { Facing::Right } else { Facing::Left }
        } else {
            if v.y > 0.0 { Facing::Up } else { Facing::Down }
        }
    }
}

// ── Gait ─────────────────────────────────────────────────────────────────

/// Which frame sequence is currently playing.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum Gait {
    #[default]
    Idle,
    Walk,
    Run,
}

/// One animation cycle: atlas column indices + seconds per frame.
#[derive(Clone, Debug)]
pub struct FrameSeq {
    pub cols: Vec<usize>,
    pub interval: f32,
    /// Positions within `cols` where a foot hits the ground (for footsteps).
    pub footfalls: Vec<usize>,
}

// ── CharacterAnim component ──────────────────────────────────────────────

/// Shared animation state for any character on the surface.
///
/// Attach alongside `LinearVelocity` and a `Sprite` with a `TextureAtlas`;
/// [`animate_characters`] updates facing, gait, frame, and atlas index.
#[derive(Component)]
pub struct CharacterAnim {
    pub facing: Facing,
    pub gait: Gait,
    /// Index into the active sequence's `cols`.
    pub frame: usize,
    pub timer: Timer,
    pub is_moving: bool,
    /// True on the frame(s) where a footfall lands (consumed by sfx).
    pub just_stepped: bool,
    /// Total columns per facing row in the atlas.
    pub sheet_cols: usize,
    pub idle: FrameSeq,
    pub walk: FrameSeq,
    /// `None` = no run cycle (legacy sheets); walk is used at any speed.
    pub run: Option<FrameSeq>,
}

impl Default for CharacterAnim {
    fn default() -> Self {
        Self::person(0.10)
    }
}

impl CharacterAnim {
    /// Composited people sheet (18 cols): idle breathing + walk + run.
    /// `interval` is seconds per walk frame (run plays 25% faster).
    pub fn person(interval: f32) -> Self {
        Self {
            facing: Facing::Down,
            gait: Gait::Idle,
            frame: 0,
            timer: Timer::from_seconds(0.6, TimerMode::Repeating),
            is_moving: false,
            just_stepped: false,
            sheet_cols: PERSON_COLS,
            idle: FrameSeq { cols: vec![0, 1], interval: 0.6, footfalls: vec![] },
            // LPC 8-frame walk: feet plant around frames 1 and 5 of the cycle.
            walk: FrameSeq {
                cols: (2..10).collect(),
                interval,
                footfalls: vec![1, 5],
            },
            run: Some(FrameSeq {
                cols: (10..18).collect(),
                interval: interval * 0.75,
                footfalls: vec![1, 5],
            }),
        }
    }

    /// Legacy 3-col RPG sheet (fauna): still, w1, w2 with the classic
    /// still→w1→still→w2 ping-pong at `interval` seconds per phase.
    pub fn legacy_rpg(interval: f32) -> Self {
        Self {
            facing: Facing::Down,
            gait: Gait::Idle,
            frame: 0,
            timer: Timer::from_seconds(interval, TimerMode::Repeating),
            is_moving: false,
            just_stepped: false,
            sheet_cols: 3,
            idle: FrameSeq { cols: vec![0], interval, footfalls: vec![] },
            walk: FrameSeq {
                cols: vec![0, 1, 0, 2],
                interval,
                footfalls: vec![1, 3],
            },
            run: None,
        }
    }

    fn seq(&self) -> &FrameSeq {
        match self.gait {
            Gait::Idle => &self.idle,
            Gait::Walk => &self.walk,
            Gait::Run => self.run.as_ref().unwrap_or(&self.walk),
        }
    }

    /// Atlas index for the current facing + frame.
    pub fn atlas_index(&self) -> usize {
        let seq = self.seq();
        self.facing as usize * self.sheet_cols + seq.cols[self.frame.min(seq.cols.len() - 1)]
    }
}

// ── System ───────────────────────────────────────────────────────────────

/// Update facing, gait, frame, and atlas index for all entities with
/// `CharacterAnim` + `LinearVelocity` + `Sprite`.
pub fn animate_characters(
    time: Res<Time>,
    mut query: Query<(&LinearVelocity, &mut CharacterAnim, &mut Sprite)>,
) {
    for (vel, mut anim, mut sprite) in &mut query {
        let speed = vel.0.length();
        anim.just_stepped = false;

        let target_gait = if speed <= MOVE_THRESHOLD {
            Gait::Idle
        } else if speed >= RUN_THRESHOLD && anim.run.is_some() {
            Gait::Run
        } else {
            Gait::Walk
        };
        anim.is_moving = target_gait != Gait::Idle;
        if anim.is_moving {
            anim.facing = Facing::from_velocity(vel.0);
        }

        if target_gait != anim.gait {
            anim.gait = target_gait;
            anim.frame = 0;
            let interval = anim.seq().interval;
            anim.timer = Timer::from_seconds(interval, TimerMode::Repeating);
        } else {
            anim.timer.tick(time.delta());
            if anim.timer.just_finished() {
                let seq_len = anim.seq().cols.len();
                anim.frame = (anim.frame + 1) % seq_len;
                if anim.seq().footfalls.contains(&anim.frame) {
                    anim.just_stepped = true;
                }
            }
        }

        if let Some(atlas) = &mut sprite.texture_atlas {
            atlas.index = anim.atlas_index();
        }
    }
}
