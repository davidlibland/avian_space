use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use rand::Rng;

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
