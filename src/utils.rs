use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use rand::Rng;
use std::f32::consts::PI;

pub fn polygon_mesh(verts: &[Vec2]) -> Mesh {
    // use bevy::render::mesh::{Indices, PrimitiveTopology};
    // use bevy::render::render_asset::RenderAssetUsages;

    let positions: Vec<[f32; 3]> = verts.iter().map(|v| [v.x, v.y, 0.0]).collect();
    let n = verts.len();
    // Fan triangulation from vertex 0
    let mut indices: Vec<u32> = Vec::new();
    for i in 1..(n as u32 - 1) {
        indices.push(0);
        indices.push(i);
        indices.push(i + 1);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

pub fn random_velocity(speed: f32) -> Vec2 {
    let mut rng = rand::thread_rng();
    let angle = rng.gen_range(0.0_f32..(2.0 * std::f32::consts::PI));
    Vec2::new(angle.cos(), angle.sin()) * speed
}

pub fn angle_to_hit(proj_vel: f32, obj_pos: &Vec2, obj_vel: &Vec2) -> Option<f32> {
    let a = obj_vel.length_squared() - proj_vel.powi(2);
    if a == 0.0 {
        return None;
    }
    let b = 2.0 * (obj_pos.x * obj_vel.x + obj_pos.y * obj_vel.y);
    let c = obj_pos.length_squared();
    let disc_sq = b.powi(2) - 4.0 * a * c;
    if disc_sq < 0.0 {
        return None;
    }
    let disc = disc_sq.sqrt();
    let t1 = (-b + disc) / (2.0 * a);
    let t2 = (-b - disc) / (2.0 * a);
    if t1 < 0.0 && t2 < 0.0 {
        return None;
    }
    let t = if t1 < 0.0 {
        t2
    } else if t2 < 0.0 {
        t1
    } else if t1 < t2 {
        t1
    } else {
        t2
    };
    let contact_pos = obj_pos + t * obj_vel;
    return Some(contact_pos.y.atan2(contact_pos.x));
}

pub fn angle_indicator(maybe_angle: Option<f32>) -> f32 {
    maybe_angle
        .map(|a| {
            // Wrap to (-π, π] so that e.g. 2π is treated the same as 0.
            ((a + PI).rem_euclid(2.0 * PI) - PI).abs()
        })
        .map(|angle_error| {
            if angle_error < (PI / 2.0) {
                4.0 * (angle_error - PI / 2.0).powi(2) / PI.powi(2)
            } else {
                0.0
            }
        })
        .unwrap_or(0.0)
}
