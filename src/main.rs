//! Renders a 2D scene containing a single, moving sprite.

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::ecs::query::Has;
use bevy::gizmos;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};
use bevy::sprite::MaterialMesh2dBundle;
use bevy::utils::{HashMap, HashSet};
use bevy::window::{PresentMode, WindowResolution};
use bevy_pixel_camera::{PixelCameraBundle, PixelCameraPlugin};
use bevy_rapier2d::prelude::{Ccd, Collider, NoUserData, RapierPhysicsPlugin, Restitution, RigidBody, Velocity};
use bevy_rapier2d::render::{DebugRenderContext, RapierDebugRenderPlugin};
use rand::Rng;

#[derive(Component)]
struct Player;

#[derive(Component)]
struct MySprite(usize);

#[derive(Resource)]
struct HullHandle(Handle<Image>, Handle<Image>, Vec<Vec<Vec2>>);

fn main() {
  App::new()
    .add_plugins(
      DefaultPlugins
        .set(
          // This sets image filtering to nearest
          // This is done to prevent textures with low resolution (e.g. pixel art) from being blurred
          // by linear filtering.
          ImagePlugin::default_nearest(),
        )
        .set(WindowPlugin {
          primary_window: Some(Window {
            resolution: WindowResolution::new(300., 300.).with_scale_factor_override(1.0),
            present_mode: PresentMode::AutoNoVsync,
            ..default()
          }),
          ..default()
        }),
    )
    .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(2.0))
    .add_plugins(RapierDebugRenderPlugin::default())
    .add_plugins((PixelCameraPlugin, FrameTimeDiagnosticsPlugin))
    .add_plugins(LogDiagnosticsPlugin::default())
    .add_systems(Startup, setup)
    .add_systems(
      Update,
      (setup2, flip_sprites, remove_hull_vec, render_strip, movement).chain(),
    )
    .run();
}

fn movement(
  mut player: Query<&mut Velocity, With<Player>>,
  input: Res<Input<KeyCode>>,
  time: Res<Time>,
  mut debug_render_context: ResMut<DebugRenderContext>,
) {
  let Ok(mut vel) = player.get_single_mut() else {
    return;
  };

  if input.just_pressed(KeyCode::D) {
    debug_render_context.enabled = !debug_render_context.enabled;
  }

  if input.just_pressed(KeyCode::Space) {
    vel.linvel.y = 100.;
  }
  if input.pressed(KeyCode::Left) {
    vel.linvel.x -= 80. * time.delta_seconds();
  }
  if input.pressed(KeyCode::Right) {
    vel.linvel.x += 80. * time.delta_seconds();
  }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
  commands.spawn(PixelCameraBundle::from_resolution(300, 300, false));

  let loaded_map = asset_server.load("map.png");

  commands.spawn((
    SpriteBundle {
      texture: loaded_map.clone(),
      transform: Transform::from_xyz(0., 0., 0.0),
      ..default()
    },
    MySprite(0),
  ));
}

fn setup2(
  mut commands: Commands,
  asset_server: Res<AssetServer>,
  mut images: ResMut<Assets<Image>>,
  hull_handle: Option<ResMut<HullHandle>>,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<ColorMaterial>>,
) {
  if hull_handle.is_some() {
    return;
  }
  let loaded_map = asset_server.load("map.png");
  let Some(map_image) = images.get(&loaded_map) else {
    return;
  };
  let hull = find_hull_pixels(map_image);
  let hull_image = render_hull_to_image(map_image, &hull);

  let hull_image_handle = images.add(hull_image);

  commands.spawn((
    SpriteBundle {
      texture: hull_image_handle.clone(),
      transform: Transform::from_xyz(0., 0., 0.0),
      ..default()
    },
    MySprite(1),
  ));

  println!("Found {} hull pixels", hull.len());

  let hulls = build_hull_list(hull);

  println!("Found {} hulls", hulls.len());

  let hull_image = render_hulls_to_image(images.get(&hull_image_handle).unwrap(), &hulls);

  let hull_image_handle = images.add(hull_image);

  commands.spawn((
    SpriteBundle {
      texture: hull_image_handle.clone(),
      transform: Transform::from_xyz(0., 0., 0.0),
      ..default()
    },
    MySprite(2),
  ));

  let collider_hulls = hulls.clone();
  commands.insert_resource(HullHandle(loaded_map, hull_image_handle.clone(), hulls));

  for hull in collider_hulls.iter() {
    let mut hull = hull.clone();
    while remove_least_errorful_hull_vec(&mut hull, 12.) {}
    let polygon_indices = split_into_convex_polygons(&hull, None);

    for polygon_idc in polygon_indices.iter() {
      let mut polygon = Vec::new();

      for i in polygon_idc {
        polygon.push(convert_vec2(hull[*i]));
      }

      commands
        .spawn(Collider::convex_hull(&polygon).unwrap())
        .insert(TransformBundle::from(Transform::from_xyz(0.0, 0.0, 0.0)));
    }
  }

  /* Create the bouncing ball. */
  commands
    .spawn(RigidBody::Dynamic)
    .insert(
      // Circle
      MaterialMesh2dBundle {
        mesh: meshes.add(shape::Circle::new(5.).into()).into(),
        material: materials.add(ColorMaterial::from(Color::PURPLE)),
        ..default()
      },
    )
    .insert(Collider::ball(5.0))
    .insert(Restitution::coefficient(0.7))
    .insert(TransformBundle::from(Transform::from_xyz(0.0, 100.0, 1.0)))
    .insert(Ccd::default())
    .insert(Player)
    .insert(Velocity::default());

  /*let size = Extent3d {
    width: 200,
    height: 200,
    ..default()
  };

  // This is the texture that will be rendered to.
  let mut image = Image {
    texture_descriptor: TextureDescriptor {
      label: None,
      size,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba8Unorm,
      mip_level_count: 1,
      sample_count: 1,
      usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
      view_formats: &[],
    },
    ..default()
  };

  // fill image.data with zeroes
  image.resize(size);
  image.data.fill(0xff);

  let mut rng = rand::thread_rng();

  // fill image.data with random values
  for y in 0..size.height {
    for x in 0..size.width {
      let index = ((x + y * size.width) * 4) as usize;
      let d = if rng.gen::<f32>() > 0.5 { 0x00 } else { 0xff };
      image.data[index] = d;
      image.data[index + 1] = d;
      image.data[index + 2] = d;
    }
  }

  let image_handle = images.add(image);

  commands.spawn(PixelCameraBundle::from_resolution(200, 200, false));
  //commands.spawn(Camera2dBundle::default());

  commands.spawn((
    SpriteBundle {
      texture: image_handle,
      transform: Transform::from_xyz(00., 0., 0.0),
      ..default()
    },
    MySprite,
  ));*/
}

// find all black pixels that are surrounded by at least one white pixel
// image format is Grayscale8Unorm
fn find_hull_pixels(image: &Image) -> Vec<(i32, i32)> {
  let mut hull_pixels = Vec::new();

  for y in 0..image.texture_descriptor.size.height {
    for x in 0..image.texture_descriptor.size.width {
      let index = ((x + y * image.texture_descriptor.size.width) * 4) as usize;
      let pixel = image.data[index];
      if pixel == 0 {
        let mut neighbors = 0;

        for dy in -1..=1 {
          for dx in -1..=1 {
            if dx != 0 && dy != 0 || dx == 0 && dy == 0 {
              continue;
            }

            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            let is_black = if nx < 0 || nx >= image.texture_descriptor.size.width as i32 {
              false
            } else if ny < 0 || ny >= image.texture_descriptor.size.height as i32 {
              false
            } else {
              let nindex = ((nx + ny * image.texture_descriptor.size.width as i32) * 4) as usize;
              let npixel = image.data[nindex];
              npixel == 0
            };

            if is_black {
              neighbors += 1;
            }
          }
        }

        if 2 <= neighbors && neighbors <= 3 {
          hull_pixels.push((x as i32, y as i32));
        }
      }
    }
  }

  hull_pixels
}

// copy image and render hull in red pixels
fn render_hull_to_image(image: &Image, hull_pixels: &Vec<(i32, i32)>) -> Image {
  let hull_pixels: HashSet<(i32, i32)> = hull_pixels.iter().copied().collect();
  let mut new_image = Image {
    texture_descriptor: TextureDescriptor {
      label: None,
      size: image.texture_descriptor.size,
      dimension: TextureDimension::D2,
      format: TextureFormat::Rgba8Unorm,
      mip_level_count: 1,
      sample_count: 1,
      usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
      view_formats: &[],
    },
    ..default()
  };

  new_image.resize(image.texture_descriptor.size);

  println!("Image size: {:?}", image.texture_descriptor.size);

  for y in 0..image.texture_descriptor.size.height {
    for x in 0..image.texture_descriptor.size.width {
      let index = ((x + y * image.texture_descriptor.size.width) * 4) as usize;

      let is_hull = hull_pixels.contains(&(x as i32, y as i32));

      if is_hull {
        new_image.data[index] = 0xff;
        new_image.data[index + 1] = 0x00;
        new_image.data[index + 2] = 0x00;
        new_image.data[index + 3] = 0xff;
      } else {
        // copy old
        new_image.data[index] = image.data[index];
        new_image.data[index + 1] = image.data[index + 1];
        new_image.data[index + 2] = image.data[index + 2];
        new_image.data[index + 3] = image.data[index + 3];
      }
    }
  }

  new_image
}

fn convert_vec2(v: Vec2) -> Vec2 {
  Vec2::new(v.x - 149.5, -v.y + 149.5)
}

fn draw_line(gizmos: &mut Gizmos, start: Vec2, end: Vec2, color0: Color, color1: Color) {
  let mut cur = start;
  let mut next = end;

  gizmos.line_gradient_2d(convert_vec2(cur), convert_vec2(next), color0, color1);
}

fn render_strip(mut gizmos: Gizmos, hull_handle: Option<ResMut<HullHandle>>, keyboard_input: Res<Input<KeyCode>>) {
  return;
  let hull_handle = match hull_handle {
    Some(hull_handle) => hull_handle,
    None => return,
  };

  let hulls = &hull_handle.2;

  for hull in hulls.iter() {
    if !keyboard_input.pressed(KeyCode::AltLeft) {
      let polygons = split_into_convex_polygons(hull, Some(&mut gizmos));
      for polygon in polygons.iter() {
        for i in 0..polygon.len() {
          let cur = hull[polygon[i]];
          let next = hull[polygon[(i + 1) % polygon.len()]];

          draw_line(
            &mut gizmos,
            cur,
            next,
            Color::rgb(0.5, 0.5, 1.0),
            Color::rgb(0.5, 1.0, 0.5),
          );
        }
      }
    } else {
      for i in 0..hull.len() {
        let cur = hull[i];
        let next = hull[(i + 1) % hull.len()];

        draw_line(
          &mut gizmos,
          cur,
          next,
          Color::rgb(0.5, 0.5, 1.0),
          Color::rgb(0.5, 1.0, 0.5),
        );
      }
    }
  }
}

// a system that changes the z position of all MySprite components when the space bar is pressed
fn flip_sprites(mut query: Query<(&mut Visibility, &mut MySprite)>, keyboard_input: Res<Input<KeyCode>>) {
  return;
  if keyboard_input.just_pressed(KeyCode::Space) {
    println!("Flipping sprites!");
    for (mut vis, mut sprite) in query.iter_mut() {
      sprite.0 = (sprite.0 + 1) % 3;
      if sprite.0 == 0 {
        *vis = Visibility::Visible;
      } else {
        *vis = Visibility::Hidden;
      }
    }
  }
}

// a system that removes a hull vec from all hulls when enter is pressed
fn remove_hull_vec(
  hull_handle: Option<ResMut<HullHandle>>,
  keyboard_input: Res<Input<KeyCode>>,
  mut images: ResMut<Assets<Image>>,
) {
  let Some(mut hull_handle) = hull_handle else {
    return;
  };

  let mut rerender = false;

  if keyboard_input.just_pressed(KeyCode::Return) {
    println!("Removing hull vec!");
    for hull in hull_handle.2.iter_mut() {
      for _ in 0..5 {
        remove_least_errorful_hull_vec(hull, f32::MAX);
      }
      println!("Hull len: {}", hull.len());
    }
    rerender = true;
  } else if keyboard_input.just_pressed(KeyCode::Back) {
    for hull in hull_handle.2.iter_mut() {
      let mut max_remove = 0.;
      let mut removed_some = false;
      while !removed_some {
        while remove_least_errorful_hull_vec(hull, max_remove) {
          removed_some = true;
        }
        max_remove += 0.5;
      }
      println!("Removed up to {max_remove}, resulting in hull len: {}", hull.len());
    }
    rerender = true;
  }

  if rerender {
    let image = images.get(&hull_handle.0).unwrap();
    let new_image = render_hulls_to_image(image, &hull_handle.2);

    let image = images.get_mut(&hull_handle.1).unwrap();
    image.data = new_image.data;
  }
}

fn angle(left: Vec2, middle: Vec2, right: Vec2) -> f32 {
  let v1 = left - middle;
  let v2 = right - middle;
  v1.angle_between(v2)
}

fn build_hull_list(hull_points: Vec<(i32, i32)>) -> Vec<Vec<Vec2>> {
  let mut hull_points_set: HashSet<(i32, i32)> = hull_points.iter().copied().collect();

  let dxdy_around: Vec<(i32, i32)> = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    .iter()
    .rev()
    .copied()
    .collect();

  let mut hulls = Vec::new();
  let mut first_it = hull_points.iter().copied();

  loop {
    let Some(start) = first_it.next() else {
      break;
    };
    if !hull_points_set.contains(&start) {
      continue;
    }
    let mut hull = Vec::new();
    let mut from = (-1, 0);
    let mut cur = start;
    loop {
      hull.push(cur);

      let start_idx = dxdy_around.iter().position(|&p| p == from).unwrap();

      // find next point
      for i in 0..dxdy_around.len() {
        let idx = (start_idx + i + 1) % dxdy_around.len();
        let dxdy = dxdy_around[idx];
        let next = (cur.0 + dxdy.0, cur.1 + dxdy.1);

        if hull_points_set.contains(&next) {
          from = (-dxdy.0, -dxdy.1);
          cur = next;
          break;
        }
      }
      if cur == start {
        break;
      }
    }

    for p in hull.iter() {
      hull_points_set.remove(p);
    }

    let mut hull_vec = Vec::new();
    for i in 0..hull.len() {
      let cur = hull[i];
      hull_vec.push(Vec2::new(cur.0 as f32, cur.1 as f32));
    }

    hulls.push(hull_vec);
  }

  hulls
}

fn render_hulls_to_image(image: &Image, hulls: &Vec<Vec<Vec2>>) -> Image {
  let mut new_image = image.clone();

  for hull in hulls.iter() {
    let mut color: u8 = 0;

    for v in hull.iter() {
      let x = v.x as u32;
      let y = v.y as u32;
      let index = ((x + y * image.texture_descriptor.size.width) * 4) as usize;
      new_image.data[index] = 0;
      new_image.data[index + 1] = color / 2 + 0xff / 2;
      new_image.data[index + 2] = 0; //color;
      new_image.data[index + 3] = 0xff;
      color = color.wrapping_add(1);
    }
  }

  new_image
}

fn triangle_area(v1: Vec2, v2: Vec2, v3: Vec2) -> f32 {
  let a = v2 - v1;
  let b = v3 - v1;

  // Calculate the area using the cross product formula
  let area = 0.5 * (a.x * b.y - a.y * b.x);

  area
}

// removes the hull vector from the strip that has the smallest biggest angle
fn remove_least_errorful_hull_vec(hull: &mut Vec<Vec2>, max_remove: f32) -> bool {
  if hull.len() <= 3 {
    return false;
  }

  let i = (0..hull.len())
    .min_by_key(|&i| {
      let prev = hull[(i + hull.len() - 1) % hull.len()];
      let cur = hull[i];
      let next = hull[(i + 1) % hull.len()];

      (triangle_area(prev, cur, next).abs() * 10000.0) as usize
    })
    .unwrap();

  let area = {
    let prev = hull[(i + hull.len() - 1) % hull.len()];
    let cur = hull[i];
    let next = hull[(i + 1) % hull.len()];

    triangle_area(prev, cur, next).abs()
  };

  if area <= max_remove {
    println!("Removing hull vec with area: {}", area);

    hull.remove(i);
    true
  } else {
    false
  }
}

fn split_into_convex_polygons(hull: &Vec<Vec2>, gizmos: Option<&mut Gizmos>) -> Vec<Vec<usize>> {
  let mut indices: Vec<usize> = (0..hull.len()).collect();
  let mut polygons = Vec::new();

  let concave_points = find_concave_points(&hull);
  //println!("Found {} concave points", concave_points.len());

  while !indices.is_empty() {
    let mut succ_start;
    let mut prev_end;
    let mut start = 0;
    let mut end;
    let mut debug = false;
    let mut polygon = Vec::new();
    loop {
      let middle = (start + 1) % indices.len();
      end = (start + 2) % indices.len();
      /*println!(
        "start: {start} ({}), middle: {middle} ({}), end: {end} ({})",
        indices[start], indices[middle], indices[end]
      );*/
      let triangle = (hull[indices[start]], hull[indices[middle]], hull[indices[end]]);
      if !is_concave(triangle.0, triangle.1, triangle.2) {
        if !has_concave_point_inside(triangle, &hull, &concave_points) {
          polygon.push(indices[start]);
          polygon.push(indices[middle]);
          polygon.push(indices[end]);
          succ_start = middle;
          prev_end = middle;
          break;
        }
        if debug {
          //println!("triangle {start}-{middle}-{end} has concave point inside");
        }
      } else if debug {
        //println!("middle {start}-{middle}-{end} is concave");
      }

      start += 1;
      start %= indices.len();
      if start == 0 {
        if debug {
          if let Some(gizmos) = gizmos {
            for i in 0..indices.len() {
              let cur = hull[indices[i]];
              let next = hull[indices[(i + 1) % indices.len()]];

              draw_line(gizmos, cur, next, Color::rgb(1.0, 0.0, 0.), Color::rgb(1.0, 0., 0.));
            }
          }
          polygons.push(indices);
          return Vec::new();
        }
        debug = true;
      }
    }

    // try extend the polygon
    loop {
      let next = (end + 1) % indices.len();
      //println!("next: {next}");
      if next == start {
        //println!("next == start");
        break;
      }

      let t = (hull[indices[end]], hull[indices[next]], hull[indices[start]]);

      if is_concave(t.0, t.1, t.2) {
        //println!("next {end}-{next}-{start} is concave");
        break;
      }

      if is_concave(hull[indices[prev_end]], t.0, t.1) {
        //println!("prev {prev_end}-{end}-{next} is concave, angle: {}", angle(hull[indices[prev_end]], t.0, t.1));
        break;
      }

      if is_concave(t.1, t.2, hull[indices[succ_start]]) {
        //println!("start {next}-{start}-{succ_start} is concave");
        break;
      }

      if has_concave_point_inside(t, &hull, &concave_points) {
        //println!("new triangle {end}-{next}-{start} has concave point inside");
        break;
      }

      polygon.push(indices[next]);
      prev_end = end;
      end = next;
    }

    // try extend the polygon
    loop {
      let prev = (start + indices.len() - 1) % indices.len();
      //println!("prev: {prev}");
      if prev == end {
        //println!("prev == end");
        break;
      }

      let t = (hull[indices[end]], hull[indices[prev]], hull[indices[start]]);

      if is_concave(t.0, t.1, t.2) {
        //println!("prev {end}-{prev}-{start} is concave");
        break;
      }

      if is_concave(t.1, t.2, hull[indices[succ_start]]) {
        //println!("start {prev}-{start}-{succ_start} is concave");
        break;
      }

      if is_concave(hull[indices[prev_end]], t.0, t.1) {
        //println!("end {prev_end}-{end}-{prev} is concave");
        break;
      }

      if has_concave_point_inside(t, &hull, &concave_points) {
        //println!("new triangle {end}-{prev}-{start} has concave point inside");
        break;
      }

      polygon.insert(0, indices[prev]);
      succ_start = start;
      start = prev;
    }

    // remove indices from polygon
    //println!("indices before: {indices:?}");
    if polygon.len() == indices.len() {
      indices.drain(..);
    } else if start <= end {
      indices.drain(start + 1..end);
    } else {
      indices.drain(start + 1..);
      indices.drain(..end);
    }
    //println!("indices after: {indices:?}");
    //println!("Found polygon: {:?}", polygon);
    //println!("Found polygon with {} points", polygon.len());
    polygons.push(polygon);
  }

  polygons
}

/*
#[cfg(test)]
mod tests {
  use bevy::prelude::Vec2;

  use crate::split_into_convex_polygons;

  #[test]
  fn test_split_triangle() {
    let hull = vec![
      Vec2::new(0., 0.),
      Vec2::new(1., 0.),
      Vec2::new(0.8, 0.5),
      Vec2::new(1., 1.),
      Vec2::new(0., 1.),
      Vec2::new(0.2, 0.5),
    ];
    let polygons = split_into_convex_polygons(&hull);
    assert_eq!(polygons.len(), 1);
    assert_eq!(polygons[0], vec![0, 1, 2, 3]);
  }
}*/

fn has_concave_point_inside(
  triangle: (Vec2, Vec2, Vec2),
  hull: &Vec<Vec2>,
  concave_points: &HashSet<usize>,
) -> bool {
  for i in concave_points.iter() {
    let point = hull[*i];
    if is_point_inside_triangle(point, triangle) {
      return true;
    }
  }
  false
}

fn is_point_inside_triangle(point: Vec2, t: (Vec2, Vec2, Vec2)) -> bool {
  if point == t.0 || point == t.1 || point == t.2 {
    return false;
  }
  let e1 = t.1 - t.0;
  let e2 = t.2 - t.1;
  let e3 = t.0 - t.2;

  // Calculate vectors from the vertices of the triangle to the point
  let to_point1 = point - t.0;
  let to_point2 = point - t.1;
  let to_point3 = point - t.2;

  // Calculate the cross products
  let cross1 = e1.perp_dot(to_point1);
  let cross2 = e2.perp_dot(to_point2);
  let cross3 = e3.perp_dot(to_point3);

  // Check if the point is on the same side of all three edges
  if cross1 > 0.0 && cross2 > 0.0 && cross3 > 0.0 {
    true
  } else if cross1 < 0.0 && cross2 < 0.0 && cross3 < 0.0 {
    true
  } else {
    false
  }
}

fn find_concave_points(hull: &Vec<Vec2>) -> HashSet<usize> {
  let mut concave_points = HashSet::new();

  for i in 0..hull.len() {
    let prev = hull[(i + hull.len() - 1) % hull.len()];
    let cur = hull[i];
    let next = hull[(i + 1) % hull.len()];

    if is_concave(prev, cur, next) {
      concave_points.insert(i);
    }
  }

  concave_points
}

fn is_concave(prev: Vec2, cur: Vec2, next: Vec2) -> bool {
  let angle = angle(prev, cur, next);

  angle < 0. && angle > -std::f32::consts::PI
}
