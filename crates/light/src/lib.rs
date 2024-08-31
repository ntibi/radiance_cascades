use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat,
};
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
use utils::Viewport;

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
                update_probes,
                ((recreate_texture, write_to_texture).chain(),),
            )
                .chain(),
        )
        .register_type::<RadianceCascadeConfig>()
        .insert_resource(RadianceCascadeConfig {
            cascades: 1,
            cascade_zero_spacing: 100,
            cascade_zero_rays: 4,
            cascade_zero_ray_length: 20,
        })
        .add_plugins(ResourceInspectorPlugin::<RadianceCascadeConfig>::default())
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
            gizmos.line_2d(start, start.lerp(end, toi / start.distance(end)), hit.color);
            // TODO color strength ?
            hit.color
        } else {
            gizmos.line_2d(start, end, Color::srgb(0.2, 0.2, 0.2));
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

    // TODO add a debug view that renders the cascade data
    //let image = Image::new(
    //Extent3d {
    //width: ??
    //height: ??
    //depth_or_array_layers: 1,
    //},
    //TextureDimension::D2,
    //cascade
    //.data
    //.iter()
    //.flat_map(|c| c.to_srgba().to_u8_array())
    //.collect::<Vec<u8>>(),
    //TextureFormat::Rgba8Unorm,
    //RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    //);

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
    mut materials: ResMut<Assets<Material>>,
    light_textures: Query<(Entity, &Handle<Material>), With<LightTexture>>,
) {
    if let Ok((_, handle)) = light_textures.get_single() {
        let material = materials.get(handle).unwrap();
        if let Some(image_handle) = &material.texture {
            let mut image = images.get_mut(image_handle).unwrap();
            for i in 0..(viewport.logical.x as usize * viewport.logical.y as usize) {
                let x = i % viewport.logical.x as usize;
                let y = i / viewport.logical.x as usize;

                // TODO color from probe sampling
                let color = Color::srgba(1., 1., 0., 1.);
                image.data[i * 4..i * 4 + 4].copy_from_slice(&color.to_srgba().to_u8_array());
            }
        }
    }
}
