use std::time::{SystemTime, UNIX_EPOCH};

use bevy::{
    a11y::AccessibilityPlugin,
    core::TaskPoolThreadAssignmentPolicy,
    core_pipeline::CorePipelinePlugin,
    diagnostic::DiagnosticsPlugin,
    gizmos::GizmoPlugin,
    input::InputPlugin,
    log::{Level, LogPlugin},
    prelude::*,
    render::{pipelined_rendering::PipelinedRenderingPlugin, RenderPlugin},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle, SpritePlugin},
    tasks::available_parallelism,
    text::TextPlugin,
    time::TimePlugin,
    ui::UiPlugin,
    winit::WinitPlugin,
};
use light::{Light, LightMaterial, LightPlugin};
use parry2d::{na::Vector2, shape::Cuboid};
use utils::UtilsPlugin;
mod rng;

fn main() {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u32;
    let rng = rng::Rng::<f32>::new(seed);

    App::new()
        .add_plugins(LogPlugin {
            level: Level::DEBUG,
            filter: [
                "wgpu=warn",
                // "wgpu=info",
                "bevy_render=warn",
                "bevy_app=info",
                "bevy_ecs=info",
                "naga=warn",
            ]
            .join(","),
            ..default()
        })
        .add_plugins(TaskPoolPlugin {
            task_pool_options: TaskPoolOptions {
                compute: TaskPoolThreadAssignmentPolicy {
                    min_threads: available_parallelism().into(),
                    max_threads: std::usize::MAX,
                    percent: 1.0,
                },
                io: TaskPoolThreadAssignmentPolicy {
                    min_threads: 0,
                    max_threads: 1,
                    percent: 0.1,
                },
                async_compute: TaskPoolThreadAssignmentPolicy {
                    min_threads: 0,
                    max_threads: 1,
                    percent: 0.1,
                },
                ..default()
            },
        })
        .add_plugins(WindowPlugin {
            primary_window: Some(Window {
                // TODO check different present modes
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        })
        .add_plugins(TypeRegistrationPlugin)
        .add_plugins(FrameCountPlugin)
        .add_plugins(TimePlugin)
        .add_plugins(TransformPlugin)
        .add_plugins(HierarchyPlugin)
        .add_plugins(DiagnosticsPlugin)
        .add_plugins(InputPlugin)
        .add_plugins(AccessibilityPlugin)
        .add_plugins(AssetPlugin::default())
        .add_plugins(<WinitPlugin as std::default::Default>::default())
        .add_plugins(RenderPlugin::default())
        .add_plugins(ImagePlugin::default())
        .add_plugins(PipelinedRenderingPlugin)
        .add_plugins(CorePipelinePlugin)
        .add_plugins(SpritePlugin)
        .add_plugins(TextPlugin)
        .add_plugins(UiPlugin)
        .add_plugins(GilrsPlugin)
        .add_plugins(GizmoPlugin)
        // custom plugins
        .add_plugins(LightPlugin)
        .add_plugins(UtilsPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, regen_shapes)
        .insert_resource(GlobalRng(rng))
        .run();
}

#[derive(Resource)]
struct GlobalRng(rng::Rng<f32>);

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn(
        TextBundle::from_section(
            "wasd: camera control\nr: regen\nspace: add\nbackspace: clear",
            TextStyle::default(),
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        }),
    );
}

const MAX_SIZE: f32 = 200.;
const MIN_SIZE: f32 = 50.;
const MAX_DIST_FROM_CENTER: f32 = 500.;

#[derive(Component)]
struct RandomEmitter;

fn regen_shapes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    shapes: Query<(Entity, &LightMaterial), With<RandomEmitter>>,
    mut rng: ResMut<GlobalRng>,
) {
    let mut emitters_to_add = 0;
    let mut occluders_to_add = 0;

    if keyboard.just_pressed(KeyCode::KeyR) {
        for (entity, material) in shapes.iter() {
            match material.behavior {
                Light::Emitter(_) => emitters_to_add += 1,
                Light::Occluder => occluders_to_add += 1,
            }
            commands.entity(entity).despawn();
        }
    }

    if keyboard.just_pressed(KeyCode::Space) {
        if keyboard.pressed(KeyCode::ShiftLeft) {
            occluders_to_add += 1;
        } else {
            emitters_to_add += 1;
        }
    }

    if keyboard.just_pressed(KeyCode::Backspace) {
        for (entity, _) in shapes.iter() {
            commands.entity(entity).despawn();
        }
    }

    while occluders_to_add > 0 || emitters_to_add > 0 {
        let size_x = rng.0.next() * (MAX_SIZE - MIN_SIZE) + MIN_SIZE;
        let size_y = rng.0.next() * (MAX_SIZE - MIN_SIZE) + MIN_SIZE;
        let r = rng.0.next();
        let g = rng.0.next();
        let b = rng.0.next();
        let x = rng.0.next() * MAX_DIST_FROM_CENTER * 2. - MAX_DIST_FROM_CENTER;
        let y = rng.0.next() * MAX_DIST_FROM_CENTER * 2. - MAX_DIST_FROM_CENTER;
        let z = rng.0.next();

        let behavior = if occluders_to_add > 0 {
            occluders_to_add -= 1;
            Light::Occluder
        } else {
            emitters_to_add -= 1;
            Light::Emitter(Color::srgb(r, g, b))
        };

        commands.spawn((
            RandomEmitter,
            MaterialMesh2dBundle {
                mesh: Mesh2dHandle(meshes.add(Rectangle::new(size_x, size_y))),
                material: materials.add(Color::srgb(r, g, b)),
                transform: Transform::from_translation(Vec2::new(x, y).extend(z)),
                ..default()
            },
            LightMaterial {
                shape: Cuboid::new(Vector2::new(size_x / 2., size_y / 2.)),
                behavior,
            },
        ));
    }
}
