//! Reusable ship scale animations.
//!
//! Provides timer-based grow/shrink components used by the carrier bay system
//! (escort launch) and the planet landing/takeoff system.

use bevy::prelude::*;

use crate::PlayState;

/// Starting / ending visual scale for scale animations (fraction of full size).
pub const ANIM_MIN_SCALE: f32 = 0.15;

/// Default duration for planet landing / takeoff animations (seconds).
pub const PLANET_ANIM_DURATION: f32 = 0.8;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Ship sprite is growing from [`ANIM_MIN_SCALE`] to full size.
/// When the timer finishes the component is removed, `custom_size` is cleared,
/// and a [`ScaleUpFinished`] message is emitted.
#[derive(Component)]
pub struct ScalingUp {
    pub timer: Timer,
    pub full_size: Vec2,
}

/// Ship sprite is shrinking from full size to [`ANIM_MIN_SCALE`].
/// When the timer finishes the component is removed and a
/// [`ScaleDownFinished`] message is emitted.
#[derive(Component)]
pub struct ScalingDown {
    pub timer: Timer,
    pub full_size: Vec2,
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Event, Message, Clone, Debug)]
pub struct ScaleUpFinished {
    pub entity: Entity,
}

#[derive(Event, Message, Clone, Debug)]
pub struct ScaleDownFinished {
    pub entity: Entity,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the natural pixel size of a sprite from its image asset.
/// Ignores `custom_size` because that may be a mid-animation value.
pub fn image_size(sprite: &Sprite, images: &Assets<Image>) -> Vec2 {
    if let Some(img) = images.get(&sprite.image) {
        let s = img.size();
        return Vec2::new(s.x as f32, s.y as f32);
    }
    Vec2::splat(32.0)
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

fn animate_scale_up(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut ScalingUp, &mut Sprite)>,
    mut finished: MessageWriter<ScaleUpFinished>,
) {
    for (entity, mut su, mut sprite) in &mut query {
        su.timer.tick(time.delta());
        let t = su.timer.fraction();
        let scale = ANIM_MIN_SCALE + (1.0 - ANIM_MIN_SCALE) * t;
        sprite.custom_size = Some(su.full_size * scale);

        if su.timer.is_finished() {
            sprite.custom_size = Some(su.full_size);
            commands.entity(entity).remove::<ScalingUp>();
            finished.write(ScaleUpFinished { entity });
        }
    }
}

fn animate_scale_down(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut ScalingDown, &mut Sprite)>,
    mut finished: MessageWriter<ScaleDownFinished>,
) {
    for (entity, mut sd, mut sprite) in &mut query {
        sd.timer.tick(time.delta());
        let t = sd.timer.fraction();
        let scale = 1.0 - (1.0 - ANIM_MIN_SCALE) * t;
        sprite.custom_size = Some(sd.full_size * scale);

        if sd.timer.is_finished() {
            commands.entity(entity).remove::<ScalingDown>();
            finished.write(ScaleDownFinished { entity });
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub fn ship_anim_plugin(app: &mut App) {
    app.add_message::<ScaleUpFinished>()
        .add_message::<ScaleDownFinished>()
        .add_systems(
            Update,
            (animate_scale_up, animate_scale_down).run_if(in_state(PlayState::Flying)),
        );
}
