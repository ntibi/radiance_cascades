use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat,
};
use bevy::render::texture::ImageSampler;
use bevy::sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle, Mesh2dHandle};
use bevy::window::WindowResized;
use bevy_inspector_egui::quick::ResourceInspectorPlugin;
use bevy_inspector_egui::{inspector_options::std_options::NumberDisplay, prelude::*};
use parry2d::{
    math::Isometry,
    na::{Point2, Vector2},
    query::{Ray, RayCast},
    shape::Cuboid,
};

pub struct LightPlugin;
pub use parry2d;
use utils::{Mouse, Viewport};

#[derive(Component)]
pub struct LightMaterial {
    pub shape: Cuboid,
    pub color: Color,
}

impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                mouse_emitter,
                update_probes,
                (
                    debug_texture,
                    debug_mouse_rays.run_if(run_if_debug_mouse_rays),
                    (recreate_texture, write_to_texture, force_texture_reload).chain(),
                ),
            )
                .chain(),
        )
        .register_type::<RadianceCascadeConfig>()
        .register_type::<RadianceCascadeDebug>()
        .insert_resource(RadianceCascadeConfig {
            cascades: 1,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 4,
            cascade_zero_ray_length: 20,
        })
        .init_resource::<RadianceCascadeDebug>()
        .add_plugins(ResourceInspectorPlugin::<RadianceCascadeConfig>::default())
        .add_plugins(ResourceInspectorPlugin::<RadianceCascadeDebug>::default())
        .add_plugins(Material2dPlugin::<Material>::default())
        .init_resource::<RadianceCascade>();
    }
}

#[derive(Reflect, Resource, Default, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct RadianceCascadeConfig {
    #[inspector(min = 1, max = 5, display = NumberDisplay::Slider)]
    /// number of cascades
    pub cascades: usize,
    /// space between the center of two probes on cascade zero
    pub cascade_zero_spacing: usize,
    /// number of rays per probe on cascade zero
    pub cascade_zero_rays: usize,
    /// length of one ray on cascade zero
    pub cascade_zero_ray_length: usize,
}

impl RadianceCascadeConfig {
    /// gets a list of the 4 probes (xy indices) that are closest to the given position
    /// and their weights for bilinear interpolation
    pub fn get_bilinear(
        &self,
        viewport: Vec2,
        pos: Vec2,
        cascade_index: usize,
    ) -> [(UVec2, f32); 4] {
        let cascade_zero_x_axis_probes =
            next_power_of_two(viewport.x as u32 / self.cascade_zero_spacing as u32) as usize;
        let cascade_zero_y_axis_probes =
            next_power_of_two(viewport.y as u32 / self.cascade_zero_spacing as u32) as usize;
        let x_axis_probes = cascade_zero_x_axis_probes / 2_usize.pow(cascade_index as u32);
        let y_axis_probes = cascade_zero_y_axis_probes / 2_usize.pow(cascade_index as u32);

        let cascade_zero_x_spacing = viewport.x / cascade_zero_x_axis_probes as f32;
        let cascade_zero_y_spacing = viewport.y / cascade_zero_y_axis_probes as f32;
        let spacing = Vec2::new(
            cascade_zero_x_spacing * 2_usize.pow(cascade_index as u32) as f32,
            cascade_zero_y_spacing * 2_usize.pow(cascade_index as u32) as f32,
        );

        let bottom_left = ((pos - spacing / 2.) / spacing).floor().as_ivec2();
        let bottom_right = bottom_left + IVec2::new(1, 0);
        let top_left = bottom_left + IVec2::new(0, 1);
        let top_right = bottom_left + IVec2::new(1, 1);

        let fract = ((pos - spacing / 2.) / spacing).fract_gl();
        let min = IVec2::ZERO;
        let max = IVec2::new(x_axis_probes as i32 - 1, y_axis_probes as i32 - 1);

        [
            (
                bottom_left.clamp(min, max).as_uvec2(),
                fract.distance(Vec2::ZERO),
            ),
            (
                bottom_right.clamp(min, max).as_uvec2(),
                fract.distance(Vec2::new(0., 1.)),
            ),
            (
                top_left.clamp(min, max).as_uvec2(),
                fract.distance(Vec2::new(1., 0.)),
            ),
            (
                top_right.clamp(min, max).as_uvec2(),
                fract.distance(Vec2::new(1., 1.)),
            ),
        ]
    }

    /// returns the indices of the 4 rays that are closest to the given angle
    /// * `angle` - the angle to interpolate in radians
    /// * `cascade_index` - the index of the cascade
    fn get_interpolated_angle_indices(&self, angle: f32, cascade_index: usize) -> [u32; 4] {
        let rays_per_probe = self.cascade_zero_rays * 4_usize.pow(cascade_index as u32);
        let angle_offset = std::f32::consts::TAU / rays_per_probe as f32 / 2.;
        let interval = std::f32::consts::TAU / rays_per_probe as f32;

        let lower_neighbour = ((angle - angle_offset) / interval).floor() as u32;
        let upper_neighbour = lower_neighbour + 1;

        [
            (lower_neighbour as i32 - 1).rem_euclid(rays_per_probe as i32) as u32,
            lower_neighbour,
            upper_neighbour,
            (upper_neighbour + 1).rem_euclid(rays_per_probe as u32) as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_zero() {
        let c = RadianceCascadeConfig {
            cascades: 1,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 4,
            cascade_zero_ray_length: 20,
        };

        let offset = std::f32::consts::PI / 4.;
        let mut r = c.get_interpolated_angle_indices(offset, 0).to_vec();
        r.sort();
        assert_eq!(r, vec![0, 1, 2, 3]);
    }

    #[test]
    fn cascade_zero_8rays() {
        let c = RadianceCascadeConfig {
            cascades: 1,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 8,
            cascade_zero_ray_length: 20,
        };

        let offset = std::f32::consts::PI / 8.;
        let mut r = c.get_interpolated_angle_indices(offset, 0).to_vec();
        r.sort();
        assert_eq!(r, vec![0, 1, 2, 7]);
    }

    #[test]
    fn cascade_one() {
        let c = RadianceCascadeConfig {
            cascades: 2,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 4,
            cascade_zero_ray_length: 20,
        };

        let offset = std::f32::consts::PI / 16.;
        let mut r = c.get_interpolated_angle_indices(offset, 1).to_vec();
        r.sort();
        assert_eq!(r, vec![0, 1, 2, 15]);
    }

    #[test]
    fn cascade_one_half_pi() {
        let c = RadianceCascadeConfig {
            cascades: 2,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 4,
            cascade_zero_ray_length: 20,
        };

        let offset = std::f32::consts::PI / 16.;
        let mut r = c
            .get_interpolated_angle_indices(std::f32::consts::PI / 2. + offset, 1)
            .to_vec();
        r.sort();
        assert_eq!(r, vec![3, 4, 5, 6]);
    }

    #[test]
    fn cascade_one_pi() {
        let c = RadianceCascadeConfig {
            cascades: 2,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 4,
            cascade_zero_ray_length: 20,
        };

        let offset = std::f32::consts::PI / 16.;
        let mut r = c
            .get_interpolated_angle_indices(std::f32::consts::PI + offset, 1)
            .to_vec();
        r.sort();
        assert_eq!(r, vec![7, 8, 9, 10]);
    }
}

#[derive(Reflect, Resource, Default, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct RadianceCascadeDebug {
    pub rays: bool,
    pub mouse_rays: bool,
    pub mouse_emitter: bool,
    pub cascade_view: bool,
}

fn run_if_debug_mouse_rays(debug: Res<RadianceCascadeDebug>) -> bool {
    debug.mouse_rays
}

#[derive(Resource, Default)]
struct RadianceCascade {
    pub data: Vec<Color>,
}

//fn _test_raycast(
//query: Query<(&LightMaterial, &Transform)>,
//mut gizmos: Gizmos,
//time: Res<Time>,
//mouse: Res<Mouse>,
//) {
//let len = 1000.;
//let elapsed = time.elapsed_seconds();

//let start = Vec2::new(0.0, 0.0);
//let end = if let Some(world_position) = mouse.0 {
//world_position
//} else {
//len * Vec2::new(elapsed.cos(), elapsed.sin())
//};

//cast_ray(
//start,
//end,
//&query
//.iter()
//.map(|(light, transform)| (light, transform.translation.xy()))
//.collect(),
//&mut gizmos,
//);
//}

fn cast_ray<'m>(
    start: Vec2,
    end: Vec2,
    emitters: &Vec<(&'m LightMaterial, Vec2)>,
) -> Option<(&'m LightMaterial, f32)> {
    let mut smallest_toi = f32::MAX;
    let mut closest_emitter = None;

    for (i, (emitter, pos)) in emitters.iter().enumerate() {
        if let Some(toi) = emitter.shape.cast_ray(
            &Isometry::translation(pos.x, pos.y),
            &Ray::new(
                Point2::new(start.x, start.y),
                Vector2::new(end.x, end.y) - Vector2::new(start.x, start.y),
            ),
            1.,
            true,
        ) {
            if toi < smallest_toi {
                smallest_toi = toi;
                closest_emitter = Some(i);
            }
        }
    }

    if let Some(i) = closest_emitter {
        Some((emitters[i].0, smallest_toi * start.distance(end)))
    } else {
        None
    }
}

fn next_power_of_two(v: u32) -> u32 {
    let mut v = v;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}

fn update_probes(
    conf: Res<RadianceCascadeConfig>,
    debug: Res<RadianceCascadeDebug>,
    mut cascade: ResMut<RadianceCascade>,
    viewport: Res<Viewport>,
    emitters: Query<(&LightMaterial, &Transform)>,
    mut gizmos: Gizmos,
    camera: Query<&Transform, With<Camera>>,
) {
    let camera = camera.single();
    let camera_bottom_left = camera.translation.truncate() - viewport.world / 2.;

    let colors = vec![
        Color::srgb(0.2, 1.0, 0.),
        Color::srgb(0.0, 0.0, 1.0),
        Color::srgb(1.0, 0.2, 0.2),
        Color::srgb(0.0, 1.0, 1.0),
        Color::srgb(1.0, 1.0, 0.0),
    ];

    let emitters = emitters
        .iter()
        .map(|(light, transform)| (light, transform.translation.xy()))
        .collect();

    let cascade_zero_x_axis_probes =
        next_power_of_two(viewport.world.x as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_y_axis_probes =
        next_power_of_two(viewport.world.y as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_x_spacing = viewport.world.x / cascade_zero_x_axis_probes as f32;
    let cascade_zero_y_spacing = viewport.world.y / cascade_zero_y_axis_probes as f32;
    let rays_per_cascade =
        cascade_zero_x_axis_probes * cascade_zero_y_axis_probes * conf.cascade_zero_rays;
    let rays_count = rays_per_cascade * conf.cascades;

    cascade
        .data
        .resize(rays_count, Color::srgba(0., 0., 0., 0.));

    for ray in 0..rays_count {
        let cascade_index = ray / rays_per_cascade;
        let x_axis_probes = cascade_zero_x_axis_probes / 2_usize.pow(cascade_index as u32);
        let x_spacing = cascade_zero_x_spacing * 2_usize.pow(cascade_index as u32) as f32;
        let y_spacing = cascade_zero_y_spacing * 2_usize.pow(cascade_index as u32) as f32;

        let rays_per_probe = conf.cascade_zero_rays * 4_usize.pow(cascade_index as u32);
        let probe_index = (ray - rays_per_cascade * cascade_index) / rays_per_probe;
        let ray_index = (ray - rays_per_cascade * cascade_index) % rays_per_probe;
        let ray_length = conf.cascade_zero_ray_length * 4_usize.pow(cascade_index as u32);

        let probe_x =
            (probe_index % x_axis_probes) as f32 * x_spacing as f32 + x_spacing as f32 / 2.;
        let probe_y =
            (probe_index / x_axis_probes) as f32 * y_spacing as f32 + y_spacing as f32 / 2.;

        let angle_offset = std::f32::consts::TAU / rays_per_probe as f32 / 2.; // TODO
        let ray_angle =
            (ray_index as f32 / rays_per_probe as f32) * std::f32::consts::TAU + angle_offset;

        let start = camera_bottom_left + Vec2::new(probe_x, probe_y);
        let end = start + Vec2::new(ray_angle.cos(), ray_angle.sin()) * ray_length as f32;

        cascade.data[ray] = if let Some((hit, toi)) = cast_ray(start, end, &emitters) {
            if debug.rays {
                gizmos.line_2d(
                    start,
                    start.lerp(end, toi / start.distance(end)),
                    colors[cascade_index],
                );
            }
            hit.color.with_alpha(1.)
        } else {
            if debug.rays {
                gizmos.line_2d(start, end, colors[cascade_index]);
            }
            Color::srgba(0., 0., 0., 0.)
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct Material {
    #[texture(0)]
    #[sampler(1)]
    texture: Option<Handle<Image>>,
}

impl Material2d for Material {
    fn fragment_shader() -> ShaderRef {
        "shaders/material.wgsl".into()
    }
}

#[derive(Component)]
struct LightTexture;

fn recreate_texture(
    conf: Res<RadianceCascadeConfig>,
    viewport: Res<Viewport>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<Material>>,
    mut resize: EventReader<WindowResized>,
    light_textures: Query<Entity, With<LightTexture>>,
) {
    if resize.read().next().is_none() {
        return;
    }

    let cascade_zero_x_axis_probes =
        next_power_of_two(viewport.world.x as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_y_axis_probes =
        next_power_of_two(viewport.world.y as u32 / conf.cascade_zero_spacing as u32) as usize;

    for entity in light_textures.iter() {
        commands.entity(entity).despawn();
    }

    let image = Image::new_fill(
        Extent3d {
            width: cascade_zero_x_axis_probes as u32,
            height: cascade_zero_y_axis_probes as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    let handle = images.add(image);

    commands.spawn((
        LightTexture,
        MaterialMesh2dBundle {
            mesh: Mesh2dHandle(meshes.add(Rectangle::new(viewport.world.x, viewport.world.y))),
            material: materials.add(Material {
                texture: Some(handle),
            }),
            transform: Transform::IDENTITY,
            ..default()
        },
    ));
}

fn write_to_texture(
    conf: Res<RadianceCascadeConfig>,
    cascade: Res<RadianceCascade>,
    viewport: Res<Viewport>,
    mut images: ResMut<Assets<Image>>,
    materials: Res<Assets<Material>>,
    light_texture: Query<(Entity, &Handle<Material>), With<LightTexture>>,
) {
    let cascade_zero_x_axis_probes =
        next_power_of_two(viewport.world.x as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_y_axis_probes =
        next_power_of_two(viewport.world.y as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_x_spacing = viewport.world.x / cascade_zero_x_axis_probes as f32;
    let cascade_zero_y_spacing = viewport.world.y / cascade_zero_y_axis_probes as f32;
    let rays_per_cascade =
        cascade_zero_x_axis_probes * cascade_zero_y_axis_probes * conf.cascade_zero_rays;

    if let Ok((_, handle)) = light_texture.get_single() {
        let material = materials.get(handle).unwrap();
        if let Some(image_handle) = &material.texture {
            let image = images.get_mut(image_handle).unwrap();

            let mut data = cascade.data.clone();

            for cascade_index in (0..(conf.cascades - 1)).rev() {
                for ray in 0..rays_per_cascade {
                    let x_axis_probes =
                        cascade_zero_x_axis_probes / 2_usize.pow(cascade_index as u32);
                    let x_spacing =
                        cascade_zero_x_spacing * 2_usize.pow(cascade_index as u32) as f32;
                    let y_spacing =
                        cascade_zero_y_spacing * 2_usize.pow(cascade_index as u32) as f32;

                    let rays_per_probe = conf.cascade_zero_rays * 4_usize.pow(cascade_index as u32);

                    let probe_x = (ray / rays_per_probe) % x_axis_probes;
                    let probe_y = (ray / rays_per_probe) / x_axis_probes;
                    let angle_offset = std::f32::consts::TAU / rays_per_probe as f32 / 2.;
                    let ray_index = ray % rays_per_probe;
                    let ray_angle = (ray_index as f32 / rays_per_probe as f32)
                        * std::f32::consts::TAU
                        + angle_offset;

                    let probe_pos = Vec2::new(
                        probe_x as f32 * x_spacing + x_spacing / 2.,
                        probe_y as f32 * y_spacing + y_spacing / 2.,
                    );

                    let c1_x_axis_probes =
                        cascade_zero_x_axis_probes / 2_usize.pow(cascade_index as u32 + 1);
                    let c1_rays_per_probe =
                        conf.cascade_zero_rays * 4_usize.pow(cascade_index as u32 + 1);
                    let probes = conf.get_bilinear(viewport.world, probe_pos, cascade_index + 1);

                    let mut colors = Vec::new();
                    for (probe, weight) in probes {
                        let i = probe.x as usize * c1_rays_per_probe
                            + probe.y as usize * c1_x_axis_probes * c1_rays_per_probe;

                        let angle_indices =
                            conf.get_interpolated_angle_indices(ray_angle, cascade_index + 1);
                        for angle_index in angle_indices.iter() {
                            colors.push(
                                data[cascade_index * rays_per_cascade + i + *angle_index as usize],
                            );
                        }
                    }
                    let color = Srgba::from_vec4(
                        colors
                            .iter()
                            .fold(Vec4::ZERO, |acc, c| acc + c.to_srgba().to_vec4())
                            / colors.len() as f32,
                    );

                    let i = cascade_index * rays_per_cascade + ray;
                    data[i] = Color::from(color);
                }
            }

            for (i, c) in data
                .chunks_exact(conf.cascade_zero_rays)
                .take(cascade_zero_x_axis_probes * cascade_zero_y_axis_probes)
                .enumerate()
            {
                let x = i % cascade_zero_x_axis_probes;
                let y = cascade_zero_y_axis_probes - 1 - i / cascade_zero_x_axis_probes;
                let flipped = y * cascade_zero_x_axis_probes + x;
                image.data[flipped * 4..flipped * 4 + 4].copy_from_slice(
                    &Srgba::from_vec4(
                        c.iter()
                            .fold(Vec4::ZERO, |acc, c| acc + c.to_srgba().to_vec4())
                            / c.len() as f32,
                    )
                    .to_u8_array(),
                );
            }
        }
    }
}

fn debug_mouse_rays(
    mouse: Res<Mouse>,
    conf: Res<RadianceCascadeConfig>,
    viewport: Res<Viewport>,
    mut gizmos: Gizmos,
    camera: Query<&Transform, With<Camera>>,
) {
    let camera = camera.get_single().unwrap();
    let camera_bottom_left = camera.translation.truncate() - viewport.world / 2.;

    let cascade_zero_x_axis_probes =
        next_power_of_two(viewport.world.x as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_y_axis_probes =
        next_power_of_two(viewport.world.y as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_x_spacing = viewport.world.x / cascade_zero_x_axis_probes as f32;
    let cascade_zero_y_spacing = viewport.world.y / cascade_zero_y_axis_probes as f32;

    let Some(cursor) = mouse.0 else {
        return;
    };
    let xy = cursor - camera_bottom_left;

    for cascade_index in (0..conf.cascades).rev() {
        let spacing = Vec2::new(
            cascade_zero_x_spacing * 2_usize.pow(cascade_index as u32) as f32,
            cascade_zero_y_spacing * 2_usize.pow(cascade_index as u32) as f32,
        );

        let [(bottom_left, _), (bottom_right, _), (top_left, _), (top_right, _)] =
            conf.get_bilinear(viewport.world, xy, cascade_index);

        let size = 4. * (1. + cascade_index as f32);
        gizmos.circle_2d(
            camera_bottom_left + (bottom_left.as_vec2()) * spacing + spacing / 2.,
            size,
            Color::srgb(0.2, 0.8, 0.2),
        );
        gizmos.circle_2d(
            camera_bottom_left + (bottom_right.as_vec2()) * spacing + spacing / 2.,
            size,
            Color::srgb(0.2, 0.8, 0.2),
        );
        gizmos.circle_2d(
            camera_bottom_left + (top_left.as_vec2()) * spacing + spacing / 2.,
            size,
            Color::srgb(0.2, 0.8, 0.2),
        );
        gizmos.circle_2d(
            camera_bottom_left + (top_right.as_vec2()) * spacing + spacing / 2.,
            size,
            Color::srgb(0.2, 0.8, 0.2),
        );
    }
}

fn force_texture_reload(
    mut materials: ResMut<Assets<Material>>,
    light_texture: Query<&Handle<Material>, With<LightTexture>>,
) {
    // TODO dirty way to force the updated texture to be sent to the gpu
    // there is probably an idiomatic way to do this, but im too tired to think or look it up
    for handle in light_texture.iter() {
        let material = materials.get_mut(handle).unwrap();
        material.texture = material.texture.clone();
    }
}

#[derive(Component)]
struct MouseEmitter;

fn mouse_emitter(
    mouse: Res<Mouse>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut commands: Commands,
    mut emitter_entity: Local<Option<Entity>>,
    debug: Res<RadianceCascadeDebug>,
) {
    match (debug.mouse_emitter, *emitter_entity, mouse.0) {
        (true, None, Some(mouse)) => {
            let color = Color::srgb(1., 1., 1.);
            let id = commands
                .spawn((
                    MouseEmitter,
                    MaterialMesh2dBundle {
                        mesh: Mesh2dHandle(meshes.add(Rectangle::new(32., 32.))),
                        material: materials.add(color),
                        transform: Transform::from_translation(mouse.extend(0.)),
                        ..default()
                    },
                    LightMaterial {
                        shape: Cuboid::new(Vector2::new(16., 16.)),
                        color,
                    },
                ))
                .id();
            *emitter_entity = Some(id);
        }
        (true, Some(entity), Some(mouse)) => {
            commands
                .entity(entity)
                .insert(Transform::from_translation(mouse.extend(0.)));
        }
        (false, Some(entity), _) => {
            commands.entity(entity).despawn();
            *emitter_entity = None;
        }
        _ => {}
    };
}

#[derive(Component)]
struct DebugTexture;

fn debug_texture(
    conf: Res<RadianceCascadeConfig>,
    debug: Res<RadianceCascadeDebug>,
    cascade: Res<RadianceCascade>,
    viewport: Res<Viewport>,
    camera: Query<&Transform, With<Camera>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<Material>>,
    light_textures: Query<Entity, With<DebugTexture>>,
    mut gizmos: Gizmos,
) {
    for entity in light_textures.iter() {
        commands.entity(entity).despawn();
    }

    if !debug.cascade_view {
        return;
    }

    let camera = camera.get_single().unwrap();
    let camera_bottom_left = camera.translation.truncate() - viewport.world / 2.;

    let cascade_zero_x_axis_probes =
        next_power_of_two(viewport.world.x as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_y_axis_probes =
        next_power_of_two(viewport.world.y as u32 / conf.cascade_zero_spacing as u32) as usize;

    for cascade_index in 0..conf.cascades {
        let start = cascade_index
            * cascade_zero_x_axis_probes
            * cascade_zero_y_axis_probes
            * conf.cascade_zero_rays;
        let end = (cascade_index + 1)
            * cascade_zero_x_axis_probes
            * cascade_zero_y_axis_probes
            * conf.cascade_zero_rays;

        let data = &cascade.data[start..end];
        let mut image = Image::new(
            Extent3d {
                width: cascade_zero_x_axis_probes as u32 * conf.cascade_zero_rays as u32,
                height: cascade_zero_y_axis_probes as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data.chunks_exact(cascade_zero_x_axis_probes * conf.cascade_zero_rays)
                // to invert the y axis (to get it with bottom left origin
                .rev()
                .flat_map(|y| y.iter().flat_map(|c| c.to_srgba().to_u8_array()))
                .collect::<Vec<u8>>(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        image.sampler = ImageSampler::nearest();

        let handle = images.add(image);

        let size = viewport.world / 10.;
        let mut origin = camera_bottom_left.with_y(camera_bottom_left.y + 600.);
        origin -= Vec2::new(0., size.y * cascade_index as f32);
        commands.spawn((
            DebugTexture,
            MaterialMesh2dBundle {
                mesh: Mesh2dHandle(meshes.add(Rectangle::new(size.x, size.y))),
                material: materials.add(Material {
                    texture: Some(handle),
                }),
                transform: Transform::from_translation((origin + size / 2.).extend(100.)),
                ..default()
            },
        ));
        gizmos.rect_2d(origin + size / 2., 0., size, Color::srgb(1., 1., 1.));
    }
}
