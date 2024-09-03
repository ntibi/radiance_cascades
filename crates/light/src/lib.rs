use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat,
};
use bevy::render::texture::ImageSampler;
use bevy::sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle, Mesh2dHandle};
use bevy::window::{PrimaryWindow, WindowResized};
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
        .init_resource::<RadianceCascadeDebug>()
        .insert_resource(RadianceCascadeConfig {
            cascades: 1,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 4,
            cascade_zero_ray_length: 20,
        })
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

fn run_if_debug_rays(debug: Res<RadianceCascadeDebug>) -> bool {
    debug.rays
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
        let y_axis_probes = cascade_zero_y_axis_probes / 2_usize.pow(cascade_index as u32);
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
    cascade: Res<RadianceCascade>,
    viewport: Res<Viewport>,
    camera: Query<&Transform, With<Camera>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<Material>>,
    mut resize: EventReader<WindowResized>,
    mut light_textures: Query<(Entity), With<LightTexture>>,
) {
    if resize.read().next().is_none() {
        return;
    }

    for entity in light_textures.iter() {
        commands.entity(entity).despawn();
    }

    let camera = camera.single();

    let mut image = Image::new_fill(
        Extent3d {
            width: viewport.logical.x as u32,
            height: viewport.logical.y as u32,
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
    //let camera = camera.get_single().unwrap();
    //let camera_bottom_left = camera.translation.truncate() - viewport.world / 2.;

    let cascade_zero_x_axis_probes =
        next_power_of_two(viewport.world.x as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_y_axis_probes =
        next_power_of_two(viewport.world.y as u32 / conf.cascade_zero_spacing as u32) as usize;
    let cascade_zero_x_spacing = viewport.world.x / cascade_zero_x_axis_probes as f32;
    let cascade_zero_y_spacing = viewport.world.y / cascade_zero_y_axis_probes as f32;
    let rays_per_cascade =
        cascade_zero_x_axis_probes * cascade_zero_y_axis_probes * conf.cascade_zero_rays;
    let rays_count = rays_per_cascade * conf.cascades;

    if let Ok((_, handle)) = light_texture.get_single() {
        let material = materials.get(handle).unwrap();
        if let Some(image_handle) = &material.texture {
            let image = images.get_mut(image_handle).unwrap();
            for i in 0..(viewport.logical.x as usize * viewport.logical.y as usize) {
                // cast everything as u32, so we dont get rounding errors
                // ie:
                // vp.x = 1920
                // vp.y = 1124
                // i1 1416239 and i2 1416240 are supposed to have the same y value
                //   since they are on the same y axis
                // but they will not
                // y of i1 386.375555
                // y of i2 386.375
                // with all the rounding, they will both have 386
                // its not perfect ? maybe i could just change the way i get x and y from i
                // but it works, so ill keep it like this for now
                let x = (i as u32 % viewport.logical.x as u32) as f32;
                let y = (viewport.logical.y as u32 - i as u32 / viewport.logical.x as u32) as f32;
                let pixel = Vec2::new(x, y);

                // TODO only did the interpolation for the first cascade
                let cascade_index = 0;
                let x_axis_probes = cascade_zero_x_axis_probes / 2_usize.pow(cascade_index as u32);
                let y_axis_probes = cascade_zero_y_axis_probes / 2_usize.pow(cascade_index as u32);
                let x_spacing = cascade_zero_x_spacing * 2_usize.pow(cascade_index as u32) as f32;
                let y_spacing = cascade_zero_y_spacing * 2_usize.pow(cascade_index as u32) as f32;

                let rays_per_probe = conf.cascade_zero_rays * 4_usize.pow(cascade_index as u32);
                let max_dist = ((x_spacing / 2.).powf(2.) + (y_spacing / 2.).powf(2.)).sqrt();

                macro_rules! get_probe {
                    ($name:ident, $dist_name:ident, $x_rounding_fn:ident, $y_rounding_fn:ident) => {
                        let ($name, $dist_name) = {
                            if x < x_spacing / 2. || y < y_spacing / 2. {
                                (None, None)
                            } else {
                                let probe_x =
                                    ((x - x_spacing / 2.) / x_spacing).$x_rounding_fn() as u32;
                                let probe_y =
                                    ((y - y_spacing / 2.) / y_spacing).$y_rounding_fn() as u32;

                                if probe_x >= cascade_zero_x_axis_probes as u32
                                    || probe_y >= cascade_zero_y_axis_probes as u32
                                {
                                    (None, None)
                                } else {
                                    let $dist_name = pixel.distance(Vec2::new(
                                        probe_x as f32 * x_spacing + x_spacing / 2.,
                                        probe_y as f32 * y_spacing + y_spacing / 2.,
                                    ));
                                    let $name = UVec2::new(probe_x, probe_y);
                                    let i = $name.y as usize
                                        * cascade_zero_x_axis_probes
                                        * rays_per_probe
                                        + $name.x as usize * rays_per_probe;
                                    (
                                        Some(Srgba::from_vec3(
                                            cascade.data[i..i + rays_per_probe].iter().fold(
                                                Vec3::new(0., 0., 0.),
                                                |acc, c| {
                                                    if c.alpha() > 0. {
                                                        acc + c.to_srgba().to_vec3()
                                                    } else {
                                                        acc
                                                    }
                                                },
                                            ) / rays_per_probe as f32,
                                        )),
                                        Some($dist_name),
                                    )
                                }
                            }
                        };
                    };
                }

                get_probe!(bottom_left, bottom_left_dist, floor, floor);
                get_probe!(bottom_right, bottom_right_dist, ceil, floor);
                get_probe!(top_left, top_left_dist, floor, ceil);
                get_probe!(top_right, top_right_dist, ceil, ceil);

                // TODO real merge instead of just adding
                let color = top_left.unwrap_or(Srgba::BLACK)
                    + top_right.unwrap_or(Srgba::BLACK)
                    + bottom_left.unwrap_or(Srgba::BLACK)
                    + bottom_right.unwrap_or(Srgba::BLACK);
                image.data[i * 4..i * 4 + 4].copy_from_slice(&color.to_u8_array());
            }
        }
    }
}

fn debug_mouse_rays(
    mouse: Res<Mouse>,
    conf: Res<RadianceCascadeConfig>,
    cascade: Res<RadianceCascade>,
    viewport: Res<Viewport>,
    mut gizmos: Gizmos,
    windows: Query<&Window, With<PrimaryWindow>>,
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
    let rays_per_cascade =
        cascade_zero_x_axis_probes * cascade_zero_y_axis_probes * conf.cascade_zero_rays;
    let rays_count = rays_per_cascade * conf.cascades;
    let window = windows.single();

    if let Some(world_position) = mouse.0 {
        let Some(cursor) = mouse.0 else {
            return;
        };
        // world pos but with the camera as origin
        let (x, y) = (cursor - camera_bottom_left).into();

        for cascade_index in (0..conf.cascades).rev() {
            let x_axis_probes = cascade_zero_x_axis_probes / 2_usize.pow(cascade_index as u32);
            let y_axis_probes = cascade_zero_y_axis_probes / 2_usize.pow(cascade_index as u32);
            let x_spacing = cascade_zero_x_spacing * 2_usize.pow(cascade_index as u32) as f32;
            let y_spacing = cascade_zero_y_spacing * 2_usize.pow(cascade_index as u32) as f32;

            let rays_per_probe = conf.cascade_zero_rays * 4_usize.pow(cascade_index as u32);

            // TODO macro like in write_to_texture
            // TODO bound check (u32 underflow and probe_xy > cascade_zero_xy_axis_probes)
            let probe_x = ((x - x_spacing / 2.) / x_spacing).floor() * x_spacing as f32
                + x_spacing as f32 / 2.;
            let probe_y = ((y - y_spacing / 2.) / y_spacing).floor() * y_spacing as f32
                + y_spacing as f32 / 2.;
            let bottom_left = camera_bottom_left + Vec2::new(probe_x, probe_y);

            let probe_x = ((x - x_spacing / 2.) / x_spacing).ceil() * x_spacing as f32
                + x_spacing as f32 / 2.;
            let probe_y = ((y - y_spacing / 2.) / y_spacing).floor() * y_spacing as f32
                + y_spacing as f32 / 2.;
            let bottom_right = camera_bottom_left + Vec2::new(probe_x, probe_y);

            let probe_x = ((x - x_spacing / 2.) / x_spacing).floor() * x_spacing as f32
                + x_spacing as f32 / 2.;
            let probe_y = ((y - y_spacing / 2.) / y_spacing).ceil() * y_spacing as f32
                + y_spacing as f32 / 2.;
            let top_left = camera_bottom_left + Vec2::new(probe_x, probe_y);

            let probe_x = ((x - x_spacing / 2.) / x_spacing).ceil() * x_spacing as f32
                + x_spacing as f32 / 2.;
            let probe_y = ((y - y_spacing / 2.) / y_spacing).ceil() * y_spacing as f32
                + y_spacing as f32 / 2.;
            let top_right = camera_bottom_left + Vec2::new(probe_x, probe_y);

            let size = 4. * (1. + cascade_index as f32);
            gizmos.circle_2d(bottom_left, size, Color::srgb(0.2, 0.8, 0.2));
            gizmos.circle_2d(bottom_right, size, Color::srgb(0.2, 0.8, 0.2));
            gizmos.circle_2d(top_left, size, Color::srgb(0.2, 0.8, 0.2));
            gizmos.circle_2d(top_right, size, Color::srgb(0.2, 0.8, 0.2));
        }
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
    mut light_textures: Query<(Entity), With<DebugTexture>>,
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
    let rays_per_probe = conf.cascade_zero_rays;

    let mut image = Image::new(
        Extent3d {
            width: cascade_zero_x_axis_probes as u32 * rays_per_probe as u32,
            height: cascade_zero_y_axis_probes as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        cascade
            .data
            .chunks_exact(cascade_zero_x_axis_probes * rays_per_probe)
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
    let origin = camera_bottom_left.with_y(camera_bottom_left.y + 500.);
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
