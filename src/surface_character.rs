//! Shared character animation for the planet surface.
//!
//! Both the player's walker and civilian NPCs use the same RPG Maker VX
//! sprite sheet layout (3 cols × 4 rows, 16×16px) and the same
//! still→w1→still→w2 walk cycle.  This module provides:
//!
//! - [`CharacterAnim`] component: facing, walk phase, timer
//! - [`animate_characters`] system: velocity → facing → walk phase → atlas index
//!
//! Control logic (keyboard input vs path-following) stays in the
//! respective modules (`surface.rs` for the walker, `surface_civilians.rs`
//! for NPCs).  Both just set `LinearVelocity`; this module handles the rest.

use avian2d::prelude::*;
use bevy::prelude::*;

// ── Constants ────────────────────────────────────────────────────────────

/// Sprite sheet: 3 columns (still, w1, w2) × 4 rows (down, left, right, up).
pub const SPRITE_COLS: usize = 3;

/// Velocity magnitude below which a character is considered stationary.
pub const MOVE_THRESHOLD: f32 = 5.0;

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

// ── Walk frame ───────────────────────────────────────────────────────────

/// Which animation frame to display.  Column index in the sprite sheet.
#[derive(Clone, Copy, Default, Debug)]
pub enum WalkFrame {
    #[default]
    Still = 0,
    W1 = 1,
    W2 = 2,
}

/// Compute the texture atlas index from facing direction and walk frame.
#[inline]
pub fn sprite_index(facing: Facing, frame: WalkFrame) -> usize {
    facing as usize * SPRITE_COLS + frame as usize
}

// ── CharacterAnim component ──────────────────────────────────────────────

/// Shared animation state for any character on the surface (walker or NPC).
///
/// Attach this alongside `LinearVelocity` and a `Sprite` with a
/// `TextureAtlas`.  The [`animate_characters`] system will automatically
/// update facing, walk phase, and atlas index based on the velocity.
#[derive(Component)]
pub struct CharacterAnim {
    pub facing: Facing,
    /// Walk cycle phase: 0=still, 1=w1, 2=still, 3=w2.
    pub walk_phase: u8,
    pub walk_timer: Timer,
    pub is_moving: bool,
}

impl Default for CharacterAnim {
    fn default() -> Self {
        Self {
            facing: Facing::Down,
            walk_phase: 0,
            walk_timer: Timer::from_seconds(0.15, TimerMode::Repeating),
            is_moving: false,
        }
    }
}

impl CharacterAnim {
    /// Create with a custom timer interval (for variation between NPCs).
    pub fn with_interval(interval: f32) -> Self {
        Self {
            walk_timer: Timer::from_seconds(interval, TimerMode::Repeating),
            ..default()
        }
    }

    /// Current walk frame from the phase.
    pub fn current_frame(&self) -> WalkFrame {
        if !self.is_moving {
            WalkFrame::Still
        } else {
            match self.walk_phase {
                1 => WalkFrame::W1,
                3 => WalkFrame::W2,
                _ => WalkFrame::Still,
            }
        }
    }

    /// Atlas index for the current facing + frame.
    pub fn atlas_index(&self) -> usize {
        sprite_index(self.facing, self.current_frame())
    }
}

// ── System ───────────────────────────────────────────────────────────────

/// Update facing, walk phase, and atlas index for all entities that have
/// `CharacterAnim` + `LinearVelocity` + `Sprite`.
///
/// This is the single animation system for both the player walker and
/// civilian NPCs.  Control systems only need to set `LinearVelocity`;
/// this system handles the visual side.
pub fn animate_characters(
    time: Res<Time>,
    mut query: Query<(&LinearVelocity, &mut CharacterAnim, &mut Sprite)>,
) {
    for (vel, mut anim, mut sprite) in &mut query {
        let speed = vel.0.length();

        if speed > MOVE_THRESHOLD {
            anim.is_moving = true;
            anim.facing = Facing::from_velocity(vel.0);

            anim.walk_timer.tick(time.delta());
            if anim.walk_timer.just_finished() {
                anim.walk_phase = (anim.walk_phase + 1) % 4;
            }
        } else {
            anim.is_moving = false;
            anim.walk_phase = 0;
            anim.walk_timer.reset();
        }

        if let Some(atlas) = &mut sprite.texture_atlas {
            atlas.index = anim.atlas_index();
        }
    }
}
