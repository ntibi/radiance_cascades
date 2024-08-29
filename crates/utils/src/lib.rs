use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    window::{PrimaryWindow, WindowResized},
};

pub struct UtilsPlugin;

#[derive(Component)]
pub struct LightMaterial {
    pub shape: Cuboid,
    pub color: Color,
}

impl Plugin for UtilsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_info)
            .add_systems(
                PreUpdate,
                (
                    update_fps,
                    update_mouse,
                    (update_camera, update_viewport).chain(),
                ),
            )
            .init_resource::<Mouse>()
            .init_resource::<Viewport>()
            .add_plugins(FrameTimeDiagnosticsPlugin);
    }
}

#[derive(Resource, Default)]
pub struct Mouse(pub Option<Vec2>);

fn update_mouse(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    mut mouse: ResMut<Mouse>,
) {
    let window = windows.single();
    let (camera, camera_transform) = camera.single();
    mouse.0 = if let Some(world_position) = window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor))
        .map(|ray| ray.origin.truncate())
    {
        Some(world_position)
    } else {
        None
    };
}

#[derive(Resource, Debug, Default)]
pub struct Viewport {
    /// viewport size in world units
    pub world: Vec2,

    /// viewport size in logical pixels
    pub logical: Vec2,

    /// viewport size in physical pixels
    pub physical: Vec2,
}

fn update_viewport(
    mut resize: EventReader<WindowResized>,
    mut viewport: ResMut<Viewport>,
    camera_query: Query<(Entity, &Camera)>,
    window_query: Query<&Window>,
    transform_helper: TransformHelper,
) {
    let window = window_query.single();
    let (camera_entity, camera) = camera_query.single();

    let (w, h) = if let Some(resized) = resize.read().last() {
        // if we had a resize, use the last one instead of the current window size
        (resized.width, resized.height)
    } else {
        // if not, use the current window size
        (window.width(), window.height())
    };
    let camera_global = &transform_helper
        .compute_global_transform(camera_entity)
        .unwrap();
    let origin = camera
        .viewport_to_world_2d(camera_global, Vec2::new(0., 0.))
        .unwrap();
    let max = camera
        .viewport_to_world_2d(camera_global, Vec2::new(w, h))
        .unwrap();
    viewport.world = Vec2::new((max.x - origin.x).abs(), (max.y - origin.y).abs());

    viewport.logical = Vec2::new(w, h);

    viewport.physical = Vec2::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
    );

    trace!(
        "viewport:\n\tworld: {}\n\tlogical: {}\n\tphysical: {}",
        viewport.world,
        viewport.logical,
        viewport.physical
    );
}

#[derive(Component)]
struct FpsText;

fn setup_info(mut commands: Commands) {
    commands.spawn((
        Name::from("fps text"),
        FpsText,
        TextBundle::from_sections([
            TextSection::new(
                "fps: ",
                TextStyle {
                    font_size: 10.0,
                    ..default()
                },
            ),
            TextSection::new(
                "-",
                TextStyle {
                    font_size: 10.0,
                    ..default()
                },
            ),
            TextSection::new(
                " (",
                TextStyle {
                    font_size: 10.0,
                    ..default()
                },
            ),
            TextSection::new(
                "-",
                TextStyle {
                    font_size: 10.0,
                    ..default()
                },
            ),
            TextSection::new(
                "ms)",
                TextStyle {
                    font_size: 10.0,
                    ..default()
                },
            ),
        ])
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            right: Val::Px(50.0),
            ..default()
        }),
    ));
}

fn update_fps(mut query: Query<&mut Text, With<FpsText>>, diagnostics: Res<DiagnosticsStore>) {
    let mut text = query.single_mut();
    if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(value) = fps.smoothed() {
            text.sections[1].value = format!("{:3.0}", value);
        }
    }
    if let Some(framtime) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
        if let Some(value) = framtime.smoothed() {
            text.sections[3].value = format!("{:.0}", value);
        }
    }
}

const SPEED: f32 = 100.;

fn update_camera(
    mut camera: Query<&mut Transform, With<Camera>>,
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    for mut transform in camera.iter_mut() {
        if keyboard.pressed(KeyCode::KeyW) {
            transform.translation.y += SPEED * time.delta_seconds();
        }
        if keyboard.pressed(KeyCode::KeyA) {
            transform.translation.x -= SPEED * time.delta_seconds();
        }
        if keyboard.pressed(KeyCode::KeyS) {
            transform.translation.y -= SPEED * time.delta_seconds();
        }
        if keyboard.pressed(KeyCode::KeyD) {
            transform.translation.x += SPEED * time.delta_seconds();
        }
    }
}
