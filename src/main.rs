use clap::Parser;
use constants::EPSILON;
use crossterm::{cursor, terminal, ExecutableCommand, QueueableCommand};
use fastrand;
use image::{ImageBuffer, Rgb};
use primatives::{Aabb, Aabb8, Plane, Ray, Ray8, RayInverse8, Sphere, Triangle, Triangle8};
use render_objects::{
    bvh::{BvhNode, BvhNodeIndex, RenderObjectBVH},
    Hit, Hittable,
};
use std::{
    collections::hash_set::HashSet,
    fs::File,
    io::{prelude::*, stdout, BufReader, BufWriter, Write},
    ops::Range,
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use ultraviolet::{f32x8, Lerp, Vec3, Vec3x8};

use crate::render_objects::{
    bvh::RenderObjectBvhBuilder, PlaneRenderObject, RenderObjectList, TriClusterRenderObject,
};

mod constants;
mod primatives;
mod render_objects;

trait Material {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray>;
    fn sample(&self, ray: &Ray, hit: &Hit) -> Vec3;
}

struct Diffuse {
    col: Vec3,
    roughness: f32,
}

struct Glossy {
    col: Vec3,
    roughness: f32,
}

struct Emmisive {
    col: Vec3,
}

impl Default for Diffuse {
    fn default() -> Self {
        Diffuse {
            col: Vec3::new(0.0, 0.0, 0.0),
            roughness: 1.0,
        }
    }
}

impl Material for Glossy {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray> {
        Some(Ray {
            origin: hit.pos + hit.norm * EPSILON,
            direction: (ray.direction - 2.0 * (ray.direction.dot(hit.norm)) * hit.norm
                + self.roughness * random_sphere().normalized())
            .normalized(),
        })
    }
    fn sample(&self, _ray: &Ray, _hit: &Hit) -> Vec3 {
        self.col
    }
}

impl Material for Diffuse {
    fn scatter(&self, _ray: &Ray, hit: &Hit) -> Option<Ray> {
        // Get random unit vector on sphere surface

        Some(Ray {
            origin: hit.pos + hit.norm * EPSILON,
            direction: (hit.norm * (1.0 + EPSILON) + random_sphere().normalized() * self.roughness)
                .normalized(),
        })
    }

    fn sample(&self, _ray: &Ray, _hit: &Hit) -> Vec3 {
        self.col
    }
}

fn random_sphere() -> Vec3 {
    loop {
        let x = 2.0 * fastrand::f32() - 1.0;
        let y = 2.0 * fastrand::f32() - 1.0;
        let z = 2.0 * fastrand::f32() - 1.0;
        let random_unit = Vec3::new(x, y, z);
        if random_unit.mag_sq() <= 1.0 {
            return random_unit;
        }
    }
}

impl Material for Emmisive {
    fn scatter(&self, _ray: &Ray, _hit: &Hit) -> Option<Ray> {
        None
    }
    fn sample(&self, _ray: &Ray, _hit: &Hit) -> Vec3 {
        self.col
    }
}

fn sample_sky(ray: &Ray) -> Vec3 {
    let apex = Vec3::new(0.5, 0.7, 0.8);
    let horizon = Vec3::new(1.0, 1.0, 1.0);
    let ground = Vec3::new(0.0, 0.0, 0.0);

    let sun = Vec3::new(1.0, 0.9, 0.9);
    let sun_dir = Vec3::new(0.5, 1.0, 1.0).normalized();

    let sky_sample = horizon
        .lerp(apex, ray.direction.y.clamp(0.0, 1.0))
        .lerp(ground, (-5.0 * ray.direction.y).clamp(0.0, 1.0).powf(0.5));
    let sun_sample = if ray.direction.dot(sun_dir) < 0.9 {
        Vec3::new(0.0, 0.0, 0.0)
    } else {
        sun
    };

    2.0 * (sky_sample + 2.0 * sun_sample)
}

fn trace_ray(ray: &Ray, scene: &dyn Hittable, depth: i32) -> Vec3 {
    let mut col = Vec3::new(1.0, 1.0, 1.0);
    let mut working_ray: Ray = Ray {
        origin: ray.origin,
        direction: ray.direction,
    };

    for _ in 0..depth {
        let hit: Option<Hit> = scene.ray_test(&working_ray);
        if hit.is_none() {
            return col * sample_sky(&working_ray);
        }
        let hit: Hit = hit.unwrap();

        col *= hit.mat.sample(&working_ray, &hit);
        let scatter_ray = hit.mat.scatter(&working_ray, &hit);

        if scatter_ray.is_none() {
            return col;
        }
        working_ray = scatter_ray.unwrap();
    }

    Vec3::new(0.0, 0.0, 0.0)
}

fn check_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.z.is_finite() && v.y.is_finite()
}

fn model_loader(model_string: &str) -> Vec<Triangle8> {
    println!("\nLoading model \"{model_string}\"");
    let path = Path::new(model_string);
    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };

    let reader = BufReader::new(file);

    let mut face_verts: Vec<(usize, usize, usize)> = Vec::new();
    let mut face_normals: Vec<(usize, usize, usize)> = Vec::new();
    let mut verts: Vec<Vec3> = Vec::new();
    let mut vert_normals: Vec<Vec3> = Vec::new();
    let mut faces_on_vert: Vec<Vec<usize>> = Vec::new();

    for line in reader.lines().flat_map(|line| line.ok()) {
        let mut line_iter = line.split_ascii_whitespace();
        match line_iter.next() {
            Some("v") => {
                verts.push(Vec3::new(
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                ));

                faces_on_vert.push(Vec::new());
            }
            Some("vn") => {
                vert_normals.push(Vec3::new(
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                ));
            }

            Some("f") => {
                let lines = [
                    line_iter.next().unwrap(),
                    line_iter.next().unwrap(),
                    line_iter.next().unwrap(),
                ];

                if let [Some((v0, n0)), Some((v1, n1)), Some((v2, n2))] =
                    lines.map(|line| line.split_once("//"))
                {
                    let i = face_verts.len();
                    let verts = (
                        v0.parse::<usize>().unwrap() - 1,
                        v1.parse::<usize>().unwrap() - 1,
                        v2.parse::<usize>().unwrap() - 1,
                    );
                    let norms = (
                        n0.parse::<usize>().unwrap() - 1,
                        n1.parse::<usize>().unwrap() - 1,
                        n2.parse::<usize>().unwrap() - 1,
                    );
                    faces_on_vert[verts.0].push(i);
                    faces_on_vert[verts.1].push(i);
                    faces_on_vert[verts.2].push(i);
                    face_verts.push(verts);
                    face_normals.push(norms);
                } else {
                    let [v0, v1, v2] = lines.map(|line| line.parse::<usize>().unwrap() - 1);
                    let i = face_verts.len();
                    faces_on_vert[v0].push(i);
                    faces_on_vert[v1].push(i);
                    faces_on_vert[v2].push(i);
                    face_verts.push((v0, v1, v2));
                }
            }
            Some(&_) => {}
            None => {}
        }
    }

    println!("\tCreating triangle clusters");
    let mut remaining_faces: HashSet<usize> = (0..face_verts.len()).collect();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    while remaining_faces.len() > 0 {
        let mut i: usize = 0;

        let mut cluster: Vec<usize> = vec![remaining_faces
            .take(&remaining_faces.iter().next().cloned().unwrap())
            .unwrap()];

        let mut connected_faces = Vec::new();

        'cluster_loop: while cluster.len() < 8 {
            while connected_faces.len() == 0 {
                if i == cluster.len() {
                    break 'cluster_loop;
                }
                let face_verts = face_verts[cluster[i]];
                connected_faces.append(&mut faces_on_vert[face_verts.0].clone());
                connected_faces.append(&mut faces_on_vert[face_verts.1].clone());
                connected_faces.append(&mut faces_on_vert[face_verts.2].clone());
                i += 1;
            }
            let face: usize = connected_faces.pop().unwrap();
            if connected_faces.contains(&face) || !remaining_faces.contains(&face) {
                continue;
            }
            remaining_faces.remove(&face);
            cluster.push(face)
        }
        clusters.push(cluster)
    }
    println!("\tPacking cluster data");
    let mut triangle_clusters: Vec<Triangle8> = Vec::new();
    for cluster in clusters.iter() {
        let nan: [f32; 8] = f32x8::splat(0.0).cmp_eq(f32x8::splat(0.0)).into();
        let nan = nan[0];

        let nan3 = Vec3::new(nan, nan, nan);
        let mut p0: [Vec3; 8] = [nan3; 8];
        let mut p1: [Vec3; 8] = [nan3; 8];
        let mut p2: [Vec3; 8] = [nan3; 8];
        let mut n0: [Vec3; 8] = [nan3; 8];
        let mut n1: [Vec3; 8] = [nan3; 8];
        let mut n2: [Vec3; 8] = [nan3; 8];
        for (i, v) in cluster.iter().enumerate() {
            let face_verts = face_verts[*v];
            let offset = Vec3::new(0.0, 0.0, -0.0);
            p0[i] = verts[face_verts.0 as usize] + offset;
            p1[i] = verts[face_verts.1 as usize] + offset;
            p2[i] = verts[face_verts.2 as usize] + offset;

            if !vert_normals.is_empty() {
                let face_normals = face_normals[*v];
                n0[i] = vert_normals[face_normals.0 as usize].normalized();
                n1[i] = vert_normals[face_normals.1 as usize].normalized();
                n2[i] = vert_normals[face_normals.2 as usize].normalized();

                if !check_finite(n0[i]) {
                    n0[i] = (p1[i] - p0[i]).cross(p2[i] - p0[i]).normalized();
                }
                if !check_finite(n1[i]) {
                    n1[i] = (p1[i] - p0[i]).cross(p2[i] - p0[i]).normalized();
                }
                if !check_finite(n2[i]) {
                    n2[i] = (p1[i] - p0[i]).cross(p2[i] - p0[i]).normalized();
                }
            } else {
                n0[i] = (p1[i] - p0[i]).cross(p2[i] - p0[i]).normalized();
                n1[i] = (p1[i] - p0[i]).cross(p2[i] - p0[i]).normalized();
                n2[i] = (p1[i] - p0[i]).cross(p2[i] - p0[i]).normalized();
            }
        }
        let packed_cluster = Triangle8 {
            p0: p0.into(),
            p1: p1.into(),
            p2: p2.into(),
            n0,
            n1,
            n2,
        };
        triangle_clusters.push(packed_cluster);
    }
    println!("\tDone!");

    // println!("Writing cluster objects");
    // let f = File::create("models/clusters.obj").expect("Unable to create file");
    // let mut f = BufWriter::new(f);

    // for (n,cluster) in triangle_clusters.iter().enumerate(){
    //     let p0: [Vec3; 8] = cluster.p0.into();
    //     let p1: [Vec3; 8] = cluster.p1.into();
    //     let p2: [Vec3; 8] = cluster.p2.into();

    //     let mut pts = p0.to_vec();
    //     pts.append(&mut p1.to_vec());
    //     pts.append(&mut p2.to_vec());

    //     f.write_fmt(format_args!("o cluster {}\n",n));
    //     for p in pts.iter(){
    //         f.write_fmt(format_args!("v {:0.6} {:0.6} {:0.6}\n",p.x,p.y,p.z));
    //     }
    //     for i in 0..8{
    //         if  f32::is_nan((&pts[i].x).clone() as f32) { continue;}
    //         f.write_fmt(format_args!("f {} {} {}\n",i+1+24*n,i+9+24*n,i+17+24*n));
    //     }

    // }

    triangle_clusters
}

#[derive(Parser, Debug)]
#[clap(
    author = "Robert Chrisite",
    about = "Simple multithreaded SIMD raytracer written in rust"
)]
struct CliArguments {
    #[clap(short = 'W', long, default_value = "512")]
    width: usize,

    #[clap(short = 'H', long, default_value = "512")]
    height: usize,

    #[clap(short = 's', long, default_value = "32")]
    samples: usize,

    #[clap(short = 'o', long, default_value = "render.png")]
    output: String,

    #[clap(short = 't', long, default_value = "16")]
    tile_size: usize,

    #[clap(long, action)]
    incremental: bool,

    #[clap(long)]
    threads: Option<usize>,

    #[clap(long)]
    dump_bvh: Option<String>,
}
struct RenderTile {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}
impl RenderTile {
    fn x_range(&self) -> Range<usize> {
        self.x..self.x + self.width
    }
    fn y_range(&self) -> Range<usize> {
        self.y..self.y + self.height
    }
}

#[derive(Clone)]
struct RenderImage {
    width: usize,
    height: usize,
    samples: usize,
    tile_size: usize,
    tile_count_x: usize,
    tile_count_y: usize,
    buffer: Arc<Mutex<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    next_tile: Arc<AtomicUsize>,
    finished_tiles: Arc<AtomicUsize>,
    aspect_ratio: f32,
}
impl RenderImage {
    fn new(width: usize, height: usize, tile_size: usize, samples: usize) -> Self {
        Self {
            width,
            height,
            samples,
            tile_size,
            tile_count_x: width.div_ceil(tile_size),
            tile_count_y: height.div_ceil(tile_size),
            next_tile: Arc::new(AtomicUsize::new(0)),
            finished_tiles: Arc::new(AtomicUsize::new(0)),
            buffer: Arc::new(Mutex::new(ImageBuffer::new(width as u32, height as u32))),
            aspect_ratio: width as f32 / height as f32,
        }
    }

    fn tile_count(&self) -> usize {
        self.tile_count_x * self.tile_count_y
    }

    fn get_tile(&mut self) -> Option<RenderTile> {
        let index = self.next_tile.fetch_add(1, Ordering::Relaxed);
        if index >= self.tile_count_x * self.tile_count_y {
            return None;
        }

        let tile_x = index.rem_euclid(self.tile_count_x);
        let tile_y = index.div_euclid(self.tile_count_x);

        let pixel_x = tile_x * self.tile_size;
        let pixel_y = tile_y * self.tile_size;
        let tile_width = (pixel_x + self.tile_size).min(self.width) - pixel_x;
        let tile_height = (pixel_y + self.tile_size).min(self.height) - pixel_y;

        let tile = RenderTile {
            x: pixel_x,
            y: pixel_y,
            width: tile_width,
            height: tile_height,
        };

        {
            let mut buffer = self.buffer.lock().unwrap();
            let mut flat = buffer.as_flat_samples_mut();
            let slice = flat.as_mut_slice();
            for target_y in tile.y_range() {
                let target_start = 3 * (target_y * self.width + tile.x);
                let target_end = 3 * (target_y * self.width + tile.x + tile.width - 1);

                slice[target_start] = 255;
                slice[target_end] = 255;
            }

            for pixel in (tile.y * self.width + tile.x)..(tile.y * self.width + tile.x + tile.width)
            {
                slice[3 * pixel] = 255;
            }
            for pixel in ((tile.y + tile.height - 1) * self.width + tile.x)
                ..((tile.y + tile.height - 1) * self.width + tile.x + tile.width)
            {
                slice[3 * pixel] = 255;
            }
        }

        Some(tile)
    }

    fn write_tile(&mut self, tile: RenderTile, data: &[u8]) {
        let mut buffer = self.buffer.lock().unwrap();
        for (source_y, target_y) in tile.y_range().enumerate() {
            let target_start = 3 * (target_y * self.width + tile.x);
            let target_end = 3 * (target_y * self.width + tile.x + tile.width);

            let mut flat = buffer.as_flat_samples_mut();
            let slice = flat.as_mut_slice();
            let mut subslice = &mut slice[target_start..target_end];

            let source_start = 3 * source_y * tile.width;
            let source_end = source_start + 3 * tile.width;

            subslice.write_all(&data[source_start..source_end]).unwrap();
        }
        self.finished_tiles.fetch_add(1, Ordering::Relaxed);
    }

    fn save(&self, output_path: &Path) -> Option<()> {
        self.buffer.lock().ok()?.save(output_path).ok()?;
        Some(())
    }
}

fn spawn_render_thread(
    mut render_image: RenderImage,
    render_object: Arc<dyn Hittable + Send + Sync>,
    exit_flag: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut tile_buffer: Vec<u8> = vec![0; 3 * render_image.tile_size * render_image.tile_size];
        while let Some(render_tile) = render_image.get_tile() {
            let mut tile_pixel_index: usize = 0;

            for y in render_tile.y_range() {
                for x in render_tile.x_range() {
                    if exit_flag.as_ref().load(Ordering::Relaxed) {
                        panic!("Interrupt received");
                    }
                    let u = 1.0 - 2.0 * y as f32 / render_image.height as f32;
                    let v = 2.0 * x as f32 / render_image.width as f32 - 1.0;

                    let du = -2.0 / render_image.height as f32;
                    let dv = 2.0 / render_image.width as f32;

                    let mut col: Vec3 = Vec3::new(0.0, 0.0, 0.0);
                    for _ in 0..render_image.samples {
                        let jitter_u = (fastrand::f32() - 0.5) * du;
                        let jitter_v = (fastrand::f32() - 0.5) * dv;
                        let ray: Ray = Ray {
                            origin: Vec3::new(5.0, 2.0, 0.0),
                            direction: Vec3::new(
                                -1.0,
                                u + jitter_u,
                                (v + jitter_v) * render_image.aspect_ratio,
                            )
                            .normalized(),
                        };
                        col += trace_ray(&ray, render_object.as_ref(), 4)
                            / (render_image.samples as f32);
                    }

                    tile_buffer[tile_pixel_index + 0] = (col.x.clamp(0.0, 1.0) * 255.0) as u8;
                    tile_buffer[tile_pixel_index + 1] = (col.y.clamp(0.0, 1.0) * 255.0) as u8;
                    tile_buffer[tile_pixel_index + 2] = (col.z.clamp(0.0, 1.0) * 255.0) as u8;
                    tile_pixel_index += 3;
                }
            }
            render_image.write_tile(render_tile, &tile_buffer);
        }
    })
}

fn main() {
    let args = CliArguments::parse();
    let output_path = Path::new(&args.output);

    let exit_flag = Arc::new(AtomicBool::new(false));
    {
        let handler_exit_flag = exit_flag.clone();
        ctrlc::set_handler(move || {
            let mut stdout = stdout();
            stdout.execute(cursor::Show).unwrap();
            handler_exit_flag.store(true, Ordering::Relaxed)
        })
        .unwrap();
    }

    // Scene creation
    let test_floor = PlaneRenderObject {
        plane: Plane {
            origin: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
        },
        mat: Arc::new(Diffuse {
            col: Vec3::new(1.0, 1.0, 1.0),
            roughness: 0.9,
        }),
    };

    let teapot_tris = model_loader("models/teapot.obj");
    let sponza_tris = model_loader("models/sponza_normals.obj");

    let mut scene_model: Vec<Box<dyn Hittable + Send + Sync>> =
        Vec::with_capacity(sponza_tris.len() + teapot_tris.len());

    let sponza_mat: Arc<dyn Material + Send + Sync> = Arc::new(Diffuse {
        col: Vec3::new(0.7, 0.7, 0.7),
        roughness: 0.9,
    });

    for tri_cluster in teapot_tris.into_iter() {
        let tau: [f32; 8] = f32x8::TAU.into();
        let tau = tau[0];
        let hue = fastrand::f32() * tau;
        let col = 1.25
            * Vec3::new(
                0.5 + 0.5 * f32::sin(hue),
                0.5 + 0.5 * f32::sin(hue + tau / 3.0),
                0.5 + 0.5 * f32::sin(hue + 2.0 * tau / 3.0),
            );
        scene_model.push(Box::new(TriClusterRenderObject {
            tris: Triangle8 {
                p0: tri_cluster.p0 / f32x8::splat(2.0),
                p1: tri_cluster.p1 / f32x8::splat(2.0),
                p2: tri_cluster.p2 / f32x8::splat(2.0),
                n0: tri_cluster.n0,
                n1: tri_cluster.n1,
                n2: tri_cluster.n2,
            },
            mat: Arc::new(Emmisive { col: 2.0 * col }), //mat: Rc::clone(&tea_pot_mat)
        }));
    }

    for tri_cluster in sponza_tris.into_iter() {
        scene_model.push(Box::new(TriClusterRenderObject {
            tris: tri_cluster,
            mat: Arc::clone(&sponza_mat),
        }));
    }

    let mut bvh_builder = RenderObjectBvhBuilder::new(scene_model);
    bvh_builder.update_bvh();

    if let Some(bvh_dump) = args.dump_bvh {
        bvh_builder.dump_bvh(Path::new(&bvh_dump));
    }
    let root_render_object = RenderObjectList {
        objects: vec![Box::new(bvh_builder.build()), Box::new(test_floor)],
    };

    let render_object = Arc::new(root_render_object);

    let mut stdout = stdout();
    stdout.execute(cursor::Hide).unwrap();
    let start_time = Instant::now();

    let num_threads = args.threads.unwrap_or(
        std::thread::available_parallelism()
            .map(|p| p.get())
            .ok()
            .unwrap_or(1),
    );

    let tile_size = args.tile_size;
    let render_image = RenderImage::new(args.width, args.height, tile_size, args.samples);

    println!("\nStarting render");
    println!("\tImage size: {}x{}", args.width, args.height);
    println!("\tSamples:    {}", args.samples);
    println!("\tThreads:    {}", num_threads);
    println!();

    let mut threads = (0..num_threads)
        .map(|_| {
            spawn_render_thread(
                render_image.clone(),
                render_object.clone(),
                exit_flag.clone(),
            )
        })
        .collect::<Vec<_>>();

    let mut counter: usize = 0;
    let mut done = false;
    let total_tiles = render_image.tile_count();

    loop {
        if let Some(thread) = threads.last() {
            if thread.is_finished() {
                threads.pop().unwrap().join().unwrap();
                continue;
            }
        } else {
            done = true;
        }
        let completed_tiles = render_image.finished_tiles.load(Ordering::Relaxed);

        let elapsed = start_time.elapsed().as_secs_f32();
        let total = (elapsed / completed_tiles as f32) * total_tiles as f32;

        let term_width = terminal::size().map(|(w, _)| w).unwrap_or(32) as usize;
        let msg_time =
            format!("{completed_tiles}/{total_tiles} Tiles in {elapsed:0.2}s / {total:0.2}s");
        let progress_width = (term_width - msg_time.len() - 3).max(8);
        let render_progress =
            (progress_width as f32 * completed_tiles as f32 / total_tiles as f32).ceil() as usize;

        let msg_progress = (0..progress_width)
            .map(|i| if i <= render_progress { '#' } else { ' ' })
            .collect::<String>();

        stdout.queue(cursor::SavePosition).unwrap();
        // stdout
        //     .write_all(format!("Tiles {completed_tiles}/{total_tiles}\n").as_bytes())
        //     .unwrap();
        stdout
            .write_all(format!("{msg_time} [{msg_progress}]").as_bytes())
            .unwrap();
        stdout.queue(cursor::RestorePosition).unwrap();
        stdout.flush().unwrap();

        stdout.queue(cursor::RestorePosition).unwrap();
        stdout
            .queue(terminal::Clear(terminal::ClearType::FromCursorDown))
            .unwrap();

        if args.incremental && counter.rem_euclid(8) == 0 {
            render_image.save(output_path).unwrap();
        }

        if done {
            break;
        }
        counter += 1;
        std::thread::sleep(Duration::from_millis(250));
    }

    let duration = start_time.elapsed().as_secs_f32();
    stdout.execute(cursor::Show).unwrap();
    println!("Rendered {total_tiles} tiles in {duration:0.2}s");
    render_image.save(output_path).unwrap();
    println!("Image saved to \"{}\"", output_path.to_str().unwrap());
}
