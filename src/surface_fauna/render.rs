//! Depth sorting for fauna, and the ground shadow under a flier.

use bevy::prelude::*;

use super::{FLY_Z, Fauna, FlierShadow};
use crate::surface_objects::{DepthShear, depth_z_xy};

/// Depth-sort fauna each frame, and park each flier's shadow on the ground
/// beneath it.
///
/// Roamers y-sort with the ground (feet below the sprite centre).  Fliers sort
/// at a fixed high z, over ground and player.
///
/// The shadow is what actually communicates altitude: a sprite drawn on top of
/// everything reads as "in front of", and only the gap down to a blob on the
/// ground turns that into "above".  Which way "down" points depends on where we
/// are — on the planet surface it is straight down, but inside a cabinet-sheared
/// interior a thing at height h sits up AND right of its floor position, so the
/// shadow has to go down-and-left by the same shear.  Both cases fall out of
/// [`DepthShear`], which is 0 outside and `SHX/SHY` inside.
pub fn depth_sort_fauna(
    shear: Option<Res<DepthShear>>,
    mut fauna: Query<(&mut Transform, &Fauna)>,
    mut shadows: Query<(&mut Transform, &ChildOf), (With<FlierShadow>, Without<Fauna>)>,
) {
    let k = shear.map_or(0.0, |s| s.0);
    for (mut tf, fauna) in &mut fauna {
        tf.translation.z = if fauna.flier {
            FLY_Z
        } else {
            depth_z_xy(tf.translation.x, tf.translation.y - fauna.foot_off, k)
        };
    }
    for (mut tf, child_of) in &mut shadows {
        let Ok((parent, fauna)) = fauna.get(child_of.parent()) else {
            continue;
        };
        // Local offset from the flier down to its floor position.
        let drop = Vec2::new(-k * fauna.altitude, -fauna.altitude);
        let ground = parent.translation.truncate() + drop;
        tf.translation.x = drop.x;
        tf.translation.y = drop.y;
        // The shadow is a CHILD, so its local z has to cancel the parent's
        // before adding the ground value.
        tf.translation.z = depth_z_xy(ground.x, ground.y, k) - FLY_Z;
    }
}
