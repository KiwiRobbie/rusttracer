use clap::Parser;
use crossterm::{cursor, terminal, ExecutableCommand, QueueableCommand};
use fastrand;
use image::{ImageBuffer, Rgb};
use std::io::{stdout, Write};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use ultraviolet as uv;
use uv::Lerp;

use std::time::{Duration, Instant};
use uv::Vec3;

use std::collections::hash_set::HashSet;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;

const EPSILON: f32 = 0.00001;

struct Ray {
    origin: uv::Vec3,
    direction: uv::Vec3,
}

struct Ray8 {
    origin: uv::Vec3x8,
    direction: uv::Vec3x8,
}

impl Ray8 {
    fn splat(ray: &Ray) -> Ray8 {
        Ray8 {
            origin: uv::Vec3x8::splat(ray.origin),
            direction: uv::Vec3x8::splat(ray.direction),
        }
    }
}

struct RayInverse8 {
    origin: uv::Vec3x8,
    inverse_direction: uv::Vec3x8,
}

impl RayInverse8 {
    fn splat(ray: &Ray) -> RayInverse8 {
        RayInverse8 {
            origin: uv::Vec3x8::splat(ray.origin),
            inverse_direction: uv::Vec3x8::splat(uv::Vec3::new(
                1.0 / ray.direction.x,
                1.0 / ray.direction.y,
                1.0 / ray.direction.z,
            )),
        }
    }
}

struct Tri {
    p0: uv::Vec3,
    p1: uv::Vec3,
    p2: uv::Vec3,
    n: uv::Vec3,
}
struct Trix8 {
    p0: uv::Vec3x8,
    p1: uv::Vec3x8,
    p2: uv::Vec3x8,
    n: [uv::Vec3; 8],
}

struct Sphere {
    origin: uv::Vec3,
    radius_squared: f32,
}

struct Plane {
    origin: uv::Vec3,
    normal: uv::Vec3,
}

type Aabb = (uv::Vec3, uv::Vec3);
type Aabb8 = (uv::Vec3x8, uv::Vec3x8);

// Render Objects
trait Hittable {
    fn ray_test(&self, ray: &Ray) -> Option<Hit>;
    fn bounding_volume(&self) -> Aabb;
}

struct Hit<'a> {
    t: f32,
    pos: uv::Vec3,
    norm: uv::Vec3,
    mat: &'a (dyn Material + Send + Sync),
}

struct NullRenderObject;

impl Hittable for NullRenderObject {
    fn ray_test(&self, _ray: &Ray) -> Option<Hit> {
        None
    }
    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

struct PlaneRenderObject {
    plane: Plane,
    mat: Arc<(dyn Material + Send + Sync)>,
}

impl PlaneRenderObject {
    fn normal(&self, _pos: uv::Vec3) -> uv::Vec3 {
        self.plane.normal
    }
}

impl Hittable for PlaneRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let div: f32 = ray.direction.dot(self.plane.normal);
        let t = (self.plane.origin - ray.origin).dot(self.plane.normal) / div;
        if t > 0.0 {
            let pos = ray.origin + ray.direction * t;

            return Some(Hit {
                t,
                pos,
                norm: self.normal(pos),
                mat: &*self.mat,
            });
        }
        None
    }
    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

struct SphereRenderObject {
    sphere: Sphere,
    mat: Arc<dyn Material + Send + Sync>,
}
impl SphereRenderObject {
    fn ray_sphere_intersect(&self, ray: &Ray) -> f32 {
        let oc = ray.origin - self.sphere.origin;
        let b = oc.dot(ray.direction);
        let c = oc.mag_sq() - self.sphere.radius_squared;
        let descrim = b * b - c;

        if descrim > 0.0 {
            let desc_sqrt = descrim.sqrt();

            let t1 = -b - desc_sqrt;
            if t1 > 0.0 {
                t1
            } else {
                let t2 = -b + desc_sqrt;
                if t2 > 0.0 {
                    t2
                } else {
                    f32::MAX
                }
            }
        } else {
            f32::MAX
        }
    }

    fn normal(&self, pos: uv::Vec3) -> uv::Vec3 {
        (pos - self.sphere.origin).normalized()
    }
}

impl Hittable for SphereRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = self.ray_sphere_intersect(ray);

        if t == f32::MAX {
            return None;
        }

        let pos: uv::Vec3 = ray.origin + ray.direction * t;
        let norm: uv::Vec3 = self.normal(pos);

        Some(Hit {
            t,
            pos,
            norm,
            mat: &*self.mat,
        })
    }
    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

struct TriRenderObject {
    tri: Tri,
    mat: Arc<dyn Material + Send + Sync>,
}

impl Hittable for TriRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = tri_intersect(&self.tri, ray);
        if t == f32::MAX {
            return None;
        }
        let pos = ray.origin + ray.direction * t;
        let norm = self.tri.n;

        Some(Hit {
            t,
            pos,
            norm,
            mat: &*self.mat,
        })
    }
    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

struct TriClusterRenderObject {
    tris: Trix8,
    mat: Arc<dyn Material + Send + Sync>,
}

impl Hittable for TriClusterRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: uv::f32x8 = tri_intersect_8(&self.tris, ray);
        let t_in: [f32; 8] = t.into();
        let (i, t) = t_in
            .into_iter()
            .enumerate()
            .reduce(|accum, item| if accum.1 <= item.1 { accum } else { item })
            .unwrap();

        if t == f32::MAX {
            return None;
        }

        let pos = ray.origin + ray.direction * t;
        let norm: uv::Vec3 = self.tris.n[i];

        Some(Hit {
            t,
            pos,
            norm,
            mat: &*self.mat,
        })
    }

    fn bounding_volume(&self) -> Aabb {
        let p0: [uv::Vec3; 8] = self.tris.p0.into();
        let p1: [uv::Vec3; 8] = self.tris.p1.into();
        let p2: [uv::Vec3; 8] = self.tris.p2.into();

        let mut pts = p0.to_vec();
        pts.append(&mut p1.to_vec());
        pts.append(&mut p2.to_vec());
        let filtered = pts
            .iter()
            .filter(|n| !n.x.is_nan())
            .collect::<Vec<&uv::Vec3>>();

        let mut min = uv::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = uv::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for v in filtered {
            min = min.min_by_component(v.clone());
            max = max.max_by_component(v.clone());
        }
        (min, max)
    }
}

struct RenderObjectList {
    objects: Vec<Box<dyn Hittable + Send + Sync>>,
}

impl Hittable for RenderObjectList {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let mut hit: Option<Hit> = None;

        for object in self.objects.iter() {
            let temp_hit = (&*object).ray_test(ray);

            if let Some(Hit { t, .. }) = hit {
                if let Some(u_t_hit) = temp_hit {
                    if u_t_hit.t < t {
                        hit = Some(u_t_hit);
                    }
                }
            } else {
                hit = temp_hit;
            }
        }
        hit
    }
    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

struct BvhNodeBuilder {
    bound: Option<Aabb>,
    subnodes: Option<Vec<usize>>,
    subnode_bounds: Option<Aabb8>,
    leaf: Option<usize>,
}

#[derive(Clone, Copy)]
enum BvhNodeIndex {
    Node(usize),
    Leaf(usize),
    None,
}

#[derive(Clone)]
struct BvhNode {
    subnodes: [BvhNodeIndex; 8],
    subnode_bounds: Aabb8,
}

struct RenderObjectBvhBuilder {
    objects: Vec<Box<dyn Hittable + Send + Sync>>,
    nodes: Vec<BvhNodeBuilder>,
}

struct RenderObjectBVH {
    objects: Vec<Box<dyn Hittable + Send + Sync>>,
    nodes: Vec<BvhNode>,
}

impl RenderObjectBvhBuilder {
    fn build(self) -> RenderObjectBVH {
        let mut new_indices: Vec<BvhNodeIndex> = vec![BvhNodeIndex::None; self.nodes.len()];
        let mut new_index: usize = 0;
        for (old_index, node) in self.nodes.iter().enumerate() {
            if let Some(leaf) = node.leaf {
                new_indices[old_index] = BvhNodeIndex::Leaf(leaf);
            } else {
                new_indices[old_index] = BvhNodeIndex::Node(new_index);
                new_index += 1;
            }
        }
        let mut nodes: Vec<Option<BvhNode>> = vec![None; new_index];

        for (old_index, new_index) in new_indices.iter().enumerate() {
            if let BvhNodeIndex::Node(id) = new_index {
                let old_node = &self.nodes[old_index];
                nodes[*id] = Some(BvhNode {
                    subnode_bounds: old_node.subnode_bounds.unwrap(),
                    subnodes: [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
                        old_node
                            .subnodes
                            .as_ref()
                            .unwrap()
                            .get(i)
                            .map(|i| new_indices[*i])
                            .unwrap_or(BvhNodeIndex::None)
                            .clone()
                    }),
                });
            }
        }
        let nodes: Vec<BvhNode> = nodes.into_iter().map(|node| node.unwrap()).collect();
        RenderObjectBVH {
            objects: self.objects,
            nodes,
        }
    }

    fn bounding_volume(mut self: &mut Self, idx: usize) -> &mut Self {
        let mut min = uv::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = uv::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        let mut s_min: [uv::Vec3; 8] = [uv::Vec3::new(f32::MAX, f32::MAX, f32::MAX); 8];
        let mut s_max: [uv::Vec3; 8] = [uv::Vec3::new(f32::MIN, f32::MIN, f32::MIN); 8];

        for (i, subnode) in self.nodes[idx]
            .subnodes
            .as_ref()
            .unwrap()
            .clone()
            .iter()
            .enumerate()
        {
            if self.nodes[subnode.clone()].bound.is_none() {
                self = RenderObjectBvhBuilder::bounding_volume(self, subnode.clone());
            }
            let subnode_bounds = self.nodes[*subnode].bound.unwrap();

            min = min.min_by_component(subnode_bounds.0);
            max = max.max_by_component(subnode_bounds.1);
            s_min[i] = subnode_bounds.0;
            s_max[i] = subnode_bounds.1;
        }
        let target_node = &mut self.nodes[idx];
        target_node.bound = Some((min, max));
        target_node.subnode_bounds = Some((s_min.into(), s_max.into()));
        self
    }

    fn split_nodes(&self, mut nodes: Vec<usize>, axis: usize) -> (Vec<usize>, Vec<usize>) {
        if nodes.len() <= 8 {
            return (nodes, vec![]);
        }

        nodes.sort_by(|a, b| {
            (self.nodes[a.clone() as usize].bound.unwrap().0[axis]
                + self.nodes[a.clone() as usize].bound.unwrap().1[axis])
                .partial_cmp(
                    &(self.nodes[b.clone() as usize].bound.unwrap().0[axis]
                        + self.nodes[b.clone() as usize].bound.unwrap().1[axis]),
                )
                .unwrap()
        });

        let split = nodes.split_off(nodes.len() / 2);
        (nodes, split)
    }

    fn bin_children(&mut self, idx: usize) {
        let (a, e) = self.split_nodes(self.nodes[idx].subnodes.as_ref().unwrap().clone(), 0);
        let (a, c) = self.split_nodes(a, 1);
        let (a, b) = self.split_nodes(a, 2);
        let (c, d) = self.split_nodes(c, 2);

        let (e, g) = self.split_nodes(e, 1);
        let (e, f) = self.split_nodes(e, 2);
        let (g, h) = self.split_nodes(g, 2);

        let mut subnodes: Vec<usize> = Vec::new();

        for node in [a, b, c, d, e, f, g, h] {
            if node.len() > 0 {
                subnodes.push(self.nodes.len() as usize);
                self.nodes.push(BvhNodeBuilder {
                    bound: None,
                    subnodes: Some(node),
                    subnode_bounds: None,
                    leaf: None,
                });
            }
        }
        self.nodes[idx as usize].subnodes = Some(subnodes);
    }

    fn update_bvh(&mut self) {
        self.nodes = vec![BvhNodeBuilder {
            bound: None,
            subnodes: None,
            subnode_bounds: None,
            leaf: None,
        }];

        let mut leaves: Vec<usize> = Vec::new();
        for (i, object) in self.objects.iter().enumerate() {
            leaves.push(self.nodes.len());
            self.nodes.push(BvhNodeBuilder {
                bound: Some(object.bounding_volume()),
                subnodes: None,
                subnode_bounds: None,
                leaf: Some(i),
            })
        }

        self.nodes[0].subnodes = Some(leaves);

        if self.nodes[0].subnodes.as_ref().unwrap().len() <= 8 {
            return;
        }

        // Check sub node count
        let mut remaining_nodes = vec![0usize];

        while remaining_nodes.len() > 0 {
            let parent = remaining_nodes.pop().unwrap();
            self.bin_children(parent);
            for node in self.nodes[parent as usize].subnodes.as_ref().unwrap() {
                if self.nodes[node.clone() as usize]
                    .subnodes
                    .as_ref()
                    .unwrap()
                    .len()
                    > 8
                {
                    remaining_nodes.push(*node);
                }
            }
        }
        self.bounding_volume(0);
    }

    fn dump_bvh(&self, file: &Path) {
        println!("Writting BVH to file");
        let f = File::create(file).expect("Unable to create file to dump BVH");
        let mut f = BufWriter::new(f);

        let mut boxes = 0;
        for node in self.nodes.iter() {
            if node.leaf.is_none() {
                continue;
            }

            boxes += 1;
            let min = node.bound.unwrap().0;
            let max: Vec3 = node.bound.unwrap().1;

            for v in 0..8 {
                let x = if v % 2 == 0 { min.x } else { max.x };
                let y = if (v / 2) % 2 == 0 { min.y } else { max.y };
                let z = if (v / 4) % 2 == 0 { min.z } else { max.z };
                f.write_fmt(format_args!("v {:0.6} {:0.6} {:0.6}\n", x, y, z))
                    .unwrap();
            }
        }

        for n in 0..boxes {
            for (a, b) in [
                (0, 1),
                (0, 2),
                (0, 4),
                (7, 3),
                (7, 5),
                (7, 6),
                (1, 5),
                (5, 4),
                (4, 6),
                (6, 2),
                (2, 3),
                (3, 1),
            ] {
                let i = 1 + n * 8 + a;
                let j = 1 + n * 8 + b;
                f.write_fmt(format_args!("l {} {}\n", i, j)).unwrap();
            }
        }
        for n in 0..boxes {
            for (u, v) in [(1, 2), (2, 4), (4, 1)] {
                let i = 1 + n * 8;
                let j = 1 + n * 8 + u;
                let k = 1 + n * 8 + u + v;
                let l = 1 + n * 8 + v;
                f.write_fmt(format_args!("f {} {} {} {}\n", i, j, k, l))
                    .unwrap();
                let i = n * 8 + 8;
                let j = n * 8 + 8 - u;
                let k = n * 8 + 8 - u - v;
                let l = n * 8 + 8 - v;
                f.write_fmt(format_args!("f {} {} {} {}\n", i, j, k, l))
                    .unwrap();
            }
        }
        println!("BVH written to file")
    }
}

impl Hittable for RenderObjectBVH {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let ray_inverse = RayInverse8::splat(ray);

        let mut hit: Option<Hit> = None;
        let mut min_t = f32::INFINITY;

        let mut bvh_tests: Vec<&BvhNode> = vec![self.nodes.first().unwrap()];

        while let Some(bvh_test_node) = bvh_tests.pop() {
            let hit_data: [f32; 8] = aabb_hit_8(
                &ray_inverse,
                bvh_test_node.subnode_bounds.0,
                bvh_test_node.subnode_bounds.1,
            )
            .into();

            for (i, t) in hit_data.into_iter().enumerate() {
                if !t.is_nan() && min_t > t {
                    match bvh_test_node.subnodes[i] {
                        BvhNodeIndex::Node(index) => {
                            bvh_tests.push(&self.nodes[index]);
                        }
                        BvhNodeIndex::Leaf(index) => {
                            if let Some(new_hit) = self.objects[index].ray_test(ray) {
                                if let Some(existing_hit) = hit.as_ref() {
                                    if existing_hit.t > new_hit.t {
                                        min_t = new_hit.t;
                                        hit = Some(new_hit);
                                    }
                                } else {
                                    min_t = new_hit.t;
                                    hit = Some(new_hit);
                                }
                            }
                        }
                        BvhNodeIndex::None => {}
                    }
                }
            }
        }

        hit
    }

    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

trait Material {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray>;
    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3;
}

struct Diffuse {
    col: uv::Vec3,
    roughness: f32,
}

struct Glossy {
    col: uv::Vec3,
    roughness: f32,
}

struct Emmisive {
    col: uv::Vec3,
}

impl Default for Diffuse {
    fn default() -> Self {
        Diffuse {
            col: uv::Vec3::new(0.0, 0.0, 0.0),
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
    fn sample(&self, _ray: &Ray, _hit: &Hit) -> uv::Vec3 {
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

    fn sample(&self, _ray: &Ray, _hit: &Hit) -> uv::Vec3 {
        self.col
    }
}

fn random_sphere() -> Vec3 {
    loop {
        let x = 2.0 * fastrand::f32() - 1.0;
        let y = 2.0 * fastrand::f32() - 1.0;
        let z = 2.0 * fastrand::f32() - 1.0;
        let random_unit = uv::Vec3::new(x, y, z);
        if random_unit.mag_sq() <= 1.0 {
            return random_unit;
        }
    }
}

impl Material for Emmisive {
    fn scatter(&self, _ray: &Ray, _hit: &Hit) -> Option<Ray> {
        None
    }
    fn sample(&self, _ray: &Ray, _hit: &Hit) -> uv::Vec3 {
        self.col
    }
}

fn aabb_hit_8(r: &RayInverse8, min: uv::Vec3x8, max: uv::Vec3x8) -> uv::f32x8 {
    let t1 = (min.x - r.origin.x) * r.inverse_direction.x;
    let t2 = (max.x - r.origin.x) * r.inverse_direction.x;
    let t3 = (min.y - r.origin.y) * r.inverse_direction.y;
    let t4 = (max.y - r.origin.y) * r.inverse_direction.y;
    let t5 = (min.z - r.origin.z) * r.inverse_direction.z;
    let t6 = (max.z - r.origin.z) * r.inverse_direction.z;

    let tmin = uv::f32x8::max(
        uv::f32x8::max(uv::f32x8::min(t1, t2), uv::f32x8::min(t3, t4)),
        uv::f32x8::min(t5, t6),
    );
    let tmax = uv::f32x8::min(
        uv::f32x8::min(uv::f32x8::max(t1, t2), uv::f32x8::max(t3, t4)),
        uv::f32x8::max(t5, t6),
    );

    tmax.cmp_lt(uv::f32x8::ZERO) | tmax.cmp_lt(tmin) | tmin
}

fn sample_sky(ray: &Ray) -> uv::Vec3 {
    let apex = uv::Vec3::new(0.5, 0.7, 0.8);
    let horizon = uv::Vec3::new(1.0, 1.0, 1.0);
    let ground = uv::Vec3::new(0.0, 0.0, 0.0);

    let sun = uv::Vec3::new(1.0, 0.9, 0.9);
    let sun_dir = uv::Vec3::new(0.5, 1.0, 1.0).normalized();

    let sky_sample = horizon
        .lerp(apex, ray.direction.y.clamp(0.0, 1.0))
        .lerp(ground, (-5.0 * ray.direction.y).clamp(0.0, 1.0).powf(0.5));
    let sun_sample = if ray.direction.dot(sun_dir) < 0.9 {
        uv::Vec3::new(0.0, 0.0, 0.0)
    } else {
        sun
    };

    2.0 * (sky_sample + 2.0 * sun_sample)
}

fn trace_ray(ray: &Ray, scene: &dyn Hittable, depth: i32) -> uv::Vec3 {
    let mut col = uv::Vec3::new(1.0, 1.0, 1.0);
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

    uv::Vec3::new(0.0, 0.0, 0.0)
}

fn tri_intersect(tri: &Tri, ray: &Ray) -> f32 {
    let (edge1, edge2, h, s, q): (uv::Vec3, uv::Vec3, uv::Vec3, uv::Vec3, uv::Vec3);
    let (a, f, u, v): (f32, f32, f32, f32);
    edge1 = tri.p1 - tri.p0;
    edge2 = tri.p2 - tri.p0;
    h = ray.direction.cross(edge2);
    a = edge1.dot(h);
    if a > -EPSILON && a < EPSILON {
        // This ray is parallel to this triangle.
        return f32::MAX;
    }

    f = 1.0 / a;
    s = ray.origin - tri.p0;
    u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return f32::MAX;
    }

    q = s.cross(edge1);
    v = f * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return f32::MAX;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = f * edge2.dot(q);
    if t > EPSILON {
        // ray intersection
        return t;
    }
    // This means that there is a line intersection but not a ray intersection.
    f32::MAX
}

fn tri_intersect_8(tri: &Trix8, ray_single: &Ray) -> uv::f32x8 {
    let epsilon_x8: uv::f32x8 = uv::f32x8::splat(EPSILON);
    let ray: Ray8 = Ray8::splat(ray_single);
    let (edge1, edge2, h, s, q): (uv::Vec3x8, uv::Vec3x8, uv::Vec3x8, uv::Vec3x8, uv::Vec3x8);
    let (a, f, u, v): (uv::f32x8, uv::f32x8, uv::f32x8, uv::f32x8);
    edge1 = tri.p1 - tri.p0;
    edge2 = tri.p2 - tri.p0;
    h = ray.direction.cross(edge2);
    a = edge1.dot(h);

    let invalid = a.cmp_gt(-epsilon_x8) & a.cmp_lt(epsilon_x8);

    f = uv::f32x8::ONE / a;
    s = ray.origin - tri.p0;
    u = f * s.dot(h);

    let invalid = invalid | u.cmp_lt(uv::f32x8::ZERO) | u.cmp_gt(uv::f32x8::ONE);

    q = s.cross(edge1);
    v = f * ray.direction.dot(q);

    let invalid = invalid | v.cmp_lt(uv::f32x8::ZERO) | (u + v).cmp_gt(uv::f32x8::ONE);

    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = f * edge2.dot(q);

    // This means that there is a line intersection but not a ray intersection.
    let invalid = invalid | t.cmp_le(epsilon_x8) | (tri.p0.x * uv::f32x8::ZERO);

    invalid.blend(uv::f32x8::splat(f32::MAX), t)
}

fn model_loader(model_string: &str) -> Vec<Trix8> {
    println!("\nLoading model \"{model_string}\"");
    // Create a path to the file // sponza_simple
    let path = Path::new(model_string);
    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };

    let reader = BufReader::new(file);

    let mut faces: Vec<(u32, u32, u32)> = Vec::new();
    let mut verts: Vec<uv::Vec3> = Vec::new();
    let mut faces_on_vert: Vec<Vec<u32>> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let mut line_iter = line.split_ascii_whitespace();

        let mext = line_iter.next();

        match mext {
            Some("v") => {
                verts.push(uv::Vec3::new(
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                    line_iter.next().unwrap().parse::<f32>().unwrap(),
                ));

                faces_on_vert.push(Vec::new());
            }
            Some("f") => {
                let attached_verts = (
                    line_iter.next().unwrap().parse::<u32>().unwrap() - 1,
                    line_iter.next().unwrap().parse::<u32>().unwrap() - 1,
                    line_iter.next().unwrap().parse::<u32>().unwrap() - 1,
                );
                let i = faces.len() as u32;
                faces_on_vert[attached_verts.0 as usize].push(i);
                faces_on_vert[attached_verts.1 as usize].push(i);
                faces_on_vert[attached_verts.2 as usize].push(i);
                faces.push(attached_verts);
            }
            Some(&_) => {}
            None => {}
        }
    }

    println!("\tCreating triangle clusters");
    let mut remaining_faces: HashSet<u32> = (0..faces.len() as u32).collect();
    let mut clusters: Vec<Vec<u32>> = Vec::new();
    while remaining_faces.len() > 0 {
        let mut i: u32 = 0;

        let mut cluster: Vec<u32> = vec![remaining_faces
            .take(&remaining_faces.iter().next().cloned().unwrap())
            .unwrap()];

        let mut connected_faces = Vec::new();

        'cluster_loop: while cluster.len() < 8 {
            while connected_faces.len() == 0 {
                if i == cluster.len() as u32 {
                    break 'cluster_loop;
                }
                let face_verts = faces[cluster[i as usize] as usize];
                connected_faces.append(&mut faces_on_vert[face_verts.0 as usize].clone());
                connected_faces.append(&mut faces_on_vert[face_verts.1 as usize].clone());
                connected_faces.append(&mut faces_on_vert[face_verts.2 as usize].clone());
                i += 1;
            }
            let face: u32 = connected_faces.pop().unwrap();
            if connected_faces.contains(&face) || !remaining_faces.contains(&face) {
                continue;
            }
            remaining_faces.remove(&face);
            cluster.push(face)
        }
        clusters.push(cluster)
    }
    println!("\tPacking cluster data");
    let mut triangle_clusters: Vec<Trix8> = Vec::new();
    for cluster in clusters.iter() {
        let nan: [f32; 8] = uv::f32x8::splat(0.0).cmp_eq(uv::f32x8::splat(0.0)).into();
        let nan = nan[0];

        let nan3 = uv::Vec3::new(nan, nan, nan);
        let mut p0: [uv::Vec3; 8] = [nan3; 8];
        let mut p1: [uv::Vec3; 8] = [nan3; 8];
        let mut p2: [uv::Vec3; 8] = [nan3; 8];
        let mut n: [uv::Vec3; 8] = [nan3; 8];
        for (i, v) in cluster.iter().enumerate() {
            let face = faces[*v as usize];
            let offset = uv::Vec3::new(0.0, 0.0, -0.0);
            p0[i] = verts[face.0 as usize] + offset;
            p1[i] = verts[face.1 as usize] + offset;
            p2[i] = verts[face.2 as usize] + offset;
            n[i] = (p1[i] - p0[i]).cross(p2[i] - p0[i]).normalized();
        }
        let packed_cluster = Trix8 {
            p0: uv::Vec3x8::from(p0),
            p1: uv::Vec3x8::from(p1),
            p2: uv::Vec3x8::from(p2),
            n,
        };
        triangle_clusters.push(packed_cluster);
    }
    println!("\tDone!");

    // println!("Writing cluster objects");
    // let f = File::create("models/clusters.obj").expect("Unable to create file");
    // let mut f = BufWriter::new(f);

    // for (n,cluster) in triangle_clusters.iter().enumerate(){
    //     let p0: [uv::Vec3; 8] = cluster.p0.into();
    //     let p1: [uv::Vec3; 8] = cluster.p1.into();
    //     let p2: [uv::Vec3; 8] = cluster.p2.into();

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
    #[clap(short = 'w', long, default_value = "512")]
    width: usize,

    #[clap(short = 'h', long, default_value = "512")]
    height: usize,

    #[clap(short = 's', long, default_value = "32")]
    samples: usize,

    #[clap(short = 'o', long, default_value = "render.png")]
    output: String,

    #[clap(short = 't', long, default_value = "16")]
    tile_size: usize,

    #[clap(long, action)]
    incremetal: bool,

    #[clap(long)]
    threads: Option<usize>,

    #[clap(long)]
    dump_bvh: Option<String>,
}
struct RenderTile {
    pixel_x: usize,
    pixel_y: usize,
    pixel_width: usize,
    pixel_height: usize,
}
impl RenderTile {
    fn x_range(&self) -> Range<usize> {
        self.pixel_x..self.pixel_x + self.pixel_width
    }
    fn y_range(&self) -> Range<usize> {
        self.pixel_y..self.pixel_y + self.pixel_height
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

        Some(RenderTile {
            pixel_x,
            pixel_y,
            pixel_width: tile_width,
            pixel_height: tile_height,
        })
    }

    fn write_tile(&mut self, tile: RenderTile, data: &[u8]) {
        let mut buffer = self.buffer.lock().unwrap();
        for (source_y, target_y) in tile.y_range().enumerate() {
            let target_start = 3 * (target_y * self.width + tile.pixel_x);
            let target_end = 3 * (target_y * self.width + tile.pixel_x + tile.pixel_width);

            let mut flat = buffer.as_flat_samples_mut();
            let slice = flat.as_mut_slice();
            let mut subslice = &mut slice[target_start..target_end];

            let source_start = 3 * source_y * tile.pixel_width;
            let source_end = source_start + 3 * tile.pixel_width;

            subslice.write_all(&data[source_start..source_end]).unwrap();
        }
        self.finished_tiles.fetch_add(1, Ordering::Relaxed);
    }

    fn save(&self, output_path: &Path) -> Option<()> {
        self.buffer.lock().ok()?.save(output_path).ok()?;
        Some(())
    }
}

fn spaww_render_thread(
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
                        panic!("Interupt received");
                    }
                    let u = 1.0 - 2.0 * y as f32 / render_image.height as f32;
                    let v = 2.0 * x as f32 / render_image.width as f32 - 1.0;

                    let du = -2.0 / render_image.height as f32;
                    let dv = 2.0 / render_image.width as f32;

                    let mut col: uv::Vec3 = uv::Vec3::new(0.0, 0.0, 0.0);
                    for _ in 0..render_image.samples {
                        let jitter_u = (fastrand::f32() - 0.5) * du;
                        let jitter_v = (fastrand::f32() - 0.5) * dv;
                        let ray: Ray = Ray {
                            origin: uv::Vec3::new(5.0, 2.0, 0.0),
                            direction: uv::Vec3::new(
                                -1.0,
                                u + jitter_u,
                                (v + jitter_v) * render_image.aspect_ratio,
                            )
                            .normalized(),
                        };
                        col += trace_ray(&ray, render_object.as_ref(), 4)
                            / (render_image.samples as f32);
                    }
                    tile_buffer[tile_pixel_index + 0] = (col.x * 255.0) as u8;
                    tile_buffer[tile_pixel_index + 1] = (col.y * 255.0) as u8;
                    tile_buffer[tile_pixel_index + 2] = (col.z * 255.0) as u8;
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
            origin: uv::Vec3::new(0.0, 0.0, 0.0),
            normal: uv::Vec3::new(0.0, 1.0, 0.0),
        },
        mat: Arc::new(Diffuse {
            col: uv::Vec3::new(1.0, 1.0, 1.0),
            roughness: 1.0,
        }),
    };

    let teapot_tris = model_loader("models/teapot.obj");
    let sponza_tris = model_loader("models/sponza_simple.obj");

    let mut scene_model: Vec<Box<dyn Hittable + Send + Sync>> =
        Vec::with_capacity(sponza_tris.len() + teapot_tris.len());

    let sponza_mat: Arc<dyn Material + Send + Sync> = Arc::new(Glossy {
        col: uv::Vec3::new(0.7, 0.7, 0.7),
        roughness: 0.2,
    });

    for tri_cluster in teapot_tris.into_iter() {
        let tau: [f32; 8] = uv::f32x8::TAU.into();
        let tau = tau[0];
        let hue = fastrand::f32() * tau;
        let col = 1.25
            * uv::Vec3::new(
                0.5 + 0.5 * f32::sin(hue),
                0.5 + 0.5 * f32::sin(hue + tau / 3.0),
                0.5 + 0.5 * f32::sin(hue + 2.0 * tau / 3.0),
            );
        scene_model.push(Box::new(TriClusterRenderObject {
            tris: Trix8 {
                p0: tri_cluster.p0 / uv::f32x8::splat(2.0),
                p1: tri_cluster.p1 / uv::f32x8::splat(2.0),
                p2: tri_cluster.p2 / uv::f32x8::splat(2.0),
                n: tri_cluster.n,
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

    let mut bvh_builder = RenderObjectBvhBuilder {
        objects: scene_model,
        nodes: vec![],
    };

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
            spaww_render_thread(
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

        if args.incremetal && counter.rem_euclid(8) == 0 {
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
