#![feature(test)]

use clap::Parser;
use crossterm::{cursor, terminal, ExecutableCommand, QueueableCommand};
use fastrand;
use std::io::{stdout, Write};
use ultraviolet as uv;
use uv::Lerp;

extern crate test;
use std::rc::Rc;
use std::time::{Duration, Instant};
use test::Bencher;
use uv::Vec3;
use uv::Vec3x8;

use std::collections::hash_set::HashSet;
use std::default;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;

const EPSILON: f32 = 0.00001;

// Basic data structs
struct Ray {
    o: uv::Vec3,
    d: uv::Vec3,
}

struct Rayx8 {
    o: uv::Vec3x8,
    d: uv::Vec3x8,
}

impl Rayx8 {
    fn splat(ray: &Ray) -> Rayx8 {
        Rayx8 {
            o: uv::Vec3x8::splat(ray.o),
            d: uv::Vec3x8::splat(ray.d),
        }
    }
}

struct RayCx8 {
    o: uv::Vec3x8,
    d: uv::Vec3x8,
    i: uv::Vec3x8,
}

impl RayCx8 {
    fn splat(ray: &Ray) -> RayCx8 {
        RayCx8 {
            o: uv::Vec3x8::splat(ray.o),
            d: uv::Vec3x8::splat(ray.d),
            i: uv::Vec3x8::splat(uv::Vec3::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z)),
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
    o: uv::Vec3,
    r_sq: f32,
}

struct Plane {
    o: uv::Vec3,
    n: uv::Vec3,
}

type Aabb = (uv::Vec3, uv::Vec3);
type Aabbx8 = (uv::Vec3x8, uv::Vec3x8);

// Render Objects
trait Hittable {
    fn ray_test(&self, ray: &Ray) -> Option<Hit>;
    fn bounding_volume(&self) -> Aabb;
}

struct Hit<'a> {
    t: f32,
    pos: uv::Vec3,
    norm: uv::Vec3,
    mat: &'a dyn Material,
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
    mat: Rc<dyn Material>,
}

impl PlaneRenderObject {
    fn normal(&self, pos: uv::Vec3) -> uv::Vec3 {
        self.plane.n
    }
}

impl Hittable for PlaneRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let div: f32 = ray.d.dot(self.plane.n);
        let t = (self.plane.o - ray.o).dot(self.plane.n) / div;
        if t > 0.0 {
            let pos = ray.o + ray.d * t;

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
    mat: Rc<dyn Material>,
}
impl SphereRenderObject {
    fn ray_sphere_intersect(&self, ray: &Ray) -> f32 {
        let oc = ray.o - self.sphere.o;
        let b = oc.dot(ray.d);
        let c = oc.mag_sq() - self.sphere.r_sq;
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
        (pos - self.sphere.o).normalized()
    }
}

impl Hittable for SphereRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = self.ray_sphere_intersect(ray);

        if t == f32::MAX {
            return None;
        }

        let pos: uv::Vec3 = ray.o + ray.d * t;
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
    mat: Rc<dyn Material>,
}

impl Hittable for TriRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = tri_intersect(&self.tri, ray);
        if t == f32::MAX {
            return None;
        }
        let pos = ray.o + ray.d * t;
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
    mat: Rc<dyn Material>,
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

        let pos = ray.o + ray.d * t;
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
    objects: Vec<Box<dyn Hittable>>,
}

impl Hittable for RenderObjectList {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let mut hit: Option<Hit> = None;

        for (i, object) in self.objects.iter().enumerate() {
            let temp_hit = (&*object).ray_test(ray);

            if hit.is_some() {
                if let Some(u_t_hit) = temp_hit {
                    let u_hit = hit.as_ref().unwrap();
                    if u_t_hit.t < u_hit.t {
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

struct BvhNode {
    bound: Option<Aabb>,
    subnodes: Option<Vec<u32>>,
    subnode_bounds: Option<Aabbx8>,
    leaf: Option<u32>,
}

struct RenderObjectBVH {
    objects: Vec<Box<dyn Hittable>>,
    nodes: Vec<BvhNode>,
    mat: Rc<dyn Material>,
}

impl RenderObjectBVH {
    fn bounding_volume(mut self: &mut Self, idx: u32) -> &mut Self {
        let mut min = uv::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = uv::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        let mut s_min: [uv::Vec3; 8] = [uv::Vec3::new(f32::MAX, f32::MAX, f32::MAX); 8];
        let mut s_max: [uv::Vec3; 8] = [uv::Vec3::new(f32::MIN, f32::MIN, f32::MIN); 8];

        for (i, subnode) in self.nodes[idx as usize]
            .subnodes
            .clone()
            .unwrap()
            .iter()
            .enumerate()
        {
            if self.nodes[subnode.clone() as usize].bound.is_none() {
                self = RenderObjectBVH::bounding_volume(self, subnode.clone());
            }
            let subnode_bounds = self.nodes[subnode.clone() as usize].bound.unwrap();

            min = min.min_by_component(subnode_bounds.0);
            max = max.max_by_component(subnode_bounds.1);
            s_min[i] = subnode_bounds.0;
            s_max[i] = subnode_bounds.1;
        }
        let target_node = &mut self.nodes[idx as usize];
        target_node.bound = Some((min, max));
        target_node.subnode_bounds = Some((s_min.into(), s_max.into()));
        self
    }

    fn split_nodes(&self, mut nodes: Vec<u32>, axis: usize) -> (Vec<u32>, Vec<u32>) {
        if (nodes.len() <= 8) {
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

    fn bin_children(&mut self, idx: u32) {
        let (a, e) = self.split_nodes(
            self.nodes[idx as usize].subnodes.as_ref().unwrap().clone(),
            0,
        );
        let (a, c) = self.split_nodes(a, 1);
        let (a, b) = self.split_nodes(a, 2);
        let (c, d) = self.split_nodes(c, 2);

        let (e, g) = self.split_nodes(e, 1);
        let (e, f) = self.split_nodes(e, 2);
        let (g, h) = self.split_nodes(g, 2);

        let mut subnodes: Vec<u32> = Vec::new();

        for node in [a, b, c, d, e, f, g, h] {
            if node.len() > 0 {
                subnodes.push(self.nodes.len() as u32);
                self.nodes.push(BvhNode {
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
        self.nodes = vec![BvhNode {
            bound: None,
            subnodes: None,
            subnode_bounds: None,
            leaf: None,
        }];

        let mut leaves: Vec<u32> = Vec::new();
        for (i, object) in self.objects.iter().enumerate() {
            leaves.push(self.nodes.len() as u32);
            self.nodes.push(BvhNode {
                bound: Some(object.bounding_volume()),
                subnodes: None,
                subnode_bounds: None,
                leaf: Some(i as u32),
            })
        }

        self.nodes[0].subnodes = Some(leaves);

        if self.nodes[0].subnodes.as_ref().unwrap().len() <= 8 {
            return;
        }

        // Check sub node count
        let mut remaining_nodes = vec![0u32];

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
                    // >8?
                    remaining_nodes.push(*node);
                }
            }
        }
        self.bounding_volume(0);

        return;
        // for node in self.nodes.iter(){
        //     if node.subnodes.is_none() {continue;}
        //     println!("{}",node.subnodes.as_ref().unwrap().len());
        // }

        println!("Writting bvh to file");
        let f = File::create("models/bvh.obj").expect("Unable to create file");
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
                f.write_fmt(format_args!("v {:0.6} {:0.6} {:0.6}\n", x, y, z));
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
                let A = 1 + n * 8 + a;
                let B = 1 + n * 8 + b;
                f.write_fmt(format_args!("l {} {}\n", A, B));
            }
        }
        let a = 0;
    }
}

impl Hittable for RenderObjectBVH {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        // let bvh_start = Instant::now();
        let ray_c = RayCx8::splat(ray);

        let mut bvh_tests: Vec<u32> = vec![0];
        let mut final_tests: Vec<u32> = vec![];

        while bvh_tests.len() > 0 {
            let node = &self.nodes[bvh_tests.pop().unwrap().clone() as usize];
            let hit_data: [f32; 8] = aabb_hit_8(
                &ray_c,
                node.subnode_bounds.unwrap().0,
                node.subnode_bounds.unwrap().1,
            )
            .into();

            for (i, subnode) in node.subnodes.as_ref().unwrap().iter().enumerate() {
                if hit_data[i].is_nan() {
                    let subnode_object = &self.nodes[subnode.clone() as usize];
                    if subnode_object.subnodes.is_some() {
                        bvh_tests.push(subnode.clone());
                    } else if subnode_object.leaf.is_some() {
                        final_tests.push(subnode_object.leaf.unwrap());
                    }
                }
            }
            //println!("Took {}ns", start_hit.elapsed().as_nanos());
        }

        // let bvh_duration = bvh_start.elapsed().as_nanos();
        // let hits_found = final_tests.len();
        // let final_start = Instant::now();

        let mut hit: Option<Hit> = None;
        while final_tests.len() > 0 {
            let n = final_tests.pop().unwrap();
            let temp_hit = self.objects[n as usize].as_ref().ray_test(ray);
            if temp_hit.is_some() {
                if hit.is_some() {
                    if hit.as_ref().unwrap().t > temp_hit.as_ref().unwrap().t {
                        hit = temp_hit;
                    }
                } else {
                    hit = temp_hit;
                }
            }
        }
        // let final_duration = final_start.elapsed().as_nanos();
        // let method_duration = bvh_start.elapsed().as_nanos();
        // let total_duration = final_duration + bvh_duration;
        // if hit.is_some() || true{
        //     println!("BVH traversed in: {}ns, {} possible hits found", bvh_duration, hits_found);
        //     println!("Final tests took {}ns, total time {}ns, method time {}ns\n", final_duration, total_duration,method_duration);
        // }
        hit
    }

    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

// Materials
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
            roughness: 0.0,
        }
    }
}

impl Material for Glossy {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray> {
        Some(Ray {
            o: hit.pos + hit.norm * EPSILON,
            d: ray.d - 2.0 * (ray.d.dot(hit.norm)) * hit.norm,
        })
    }
    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3 {
        self.col
    }
}

impl Material for Diffuse {
    fn scatter(&self, _ray: &Ray, hit: &Hit) -> Option<Ray> {
        // Get random unit vector on sphere surface
        let mut random_unit: uv::Vec3;
        loop {
            let x = 2.0 * fastrand::f32() - 1.0;
            let y = 2.0 * fastrand::f32() - 1.0;
            let z = 2.0 * fastrand::f32() - 1.0;
            random_unit = uv::Vec3::new(x, y, z);
            if random_unit.mag_sq() <= 1.0 {
                break;
            }
        }

        Some(Ray {
            o: hit.pos + hit.norm * EPSILON,
            d: (hit.norm * (1.0 + EPSILON) + random_unit.normalized()).normalized(),
        })
    }

    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3 {
        self.col
    }
}

impl Material for Emmisive {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray> {
        None
    }
    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3 {
        self.col
    }
}

// BVH AABBx8
fn aabb_hit_8(r: &RayCx8, min: uv::Vec3x8, max: uv::Vec3x8) -> uv::f32x8 {
    let t1 = (min.x - r.o.x) * r.i.x;
    let t2 = (max.x - r.o.x) * r.i.x;
    let t3 = (min.y - r.o.y) * r.i.y;
    let t4 = (max.y - r.o.y) * r.i.y;
    let t5 = (min.z - r.o.z) * r.i.z;
    let t6 = (max.z - r.o.z) * r.i.z;

    let tmin = uv::f32x8::max(
        uv::f32x8::max(uv::f32x8::min(t1, t2), uv::f32x8::min(t3, t4)),
        uv::f32x8::min(t5, t6),
    );
    let tmax = uv::f32x8::min(
        uv::f32x8::min(uv::f32x8::max(t1, t2), uv::f32x8::max(t3, t4)),
        uv::f32x8::max(t5, t6),
    );

    tmax.cmp_ge(uv::f32x8::ZERO) & tmax.cmp_ge(tmin)
}

// BVH AABB
fn aabb_hit(r: &Ray, min: uv::Vec3, max: uv::Vec3) -> bool {
    let dirfrac_x = 1.0 / r.d.x;
    let dirfrac_y = 1.0 / r.d.y;
    let dirfrac_z = 1.0 / r.d.z;
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    let t1 = (min.x - r.o.x) * dirfrac_x;
    let t2 = (max.x - r.o.x) * dirfrac_x;
    let t3 = (min.y - r.o.y) * dirfrac_y;
    let t4 = (max.y - r.o.y) * dirfrac_y;
    let t5 = (min.z - r.o.z) * dirfrac_z;
    let t6 = (max.z - r.o.z) * dirfrac_z;

    let tmin = f32::max(
        f32::max(f32::min(t1, t2), f32::min(t3, t4)),
        f32::min(t5, t6),
    );
    let tmax = f32::min(
        f32::min(f32::max(t1, t2), f32::max(t3, t4)),
        f32::max(t5, t6),
    );

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0.0) {
        return false;
    }
    if (tmin > tmax) {
        return false;
    }
    return true;
}

fn sample_sky(ray: &Ray) -> uv::Vec3 {
    let apex = uv::Vec3::new(0.5, 0.7, 0.8);
    let horizon = uv::Vec3::new(1.0, 1.0, 1.0);
    let ground = uv::Vec3::new(0.0, 0.0, 0.0);

    let sun = uv::Vec3::new(1.0, 0.9, 0.9);
    let sun_dir = uv::Vec3::new(0.5, 1.0, 1.0).normalized();

    let sky_sample = horizon
        .lerp(apex, ray.d.y.clamp(0.0, 1.0))
        .lerp(ground, (-5.0 * ray.d.y).clamp(0.0, 1.0).powf(0.5));
    let sun_sample = if ray.d.dot(sun_dir) < 0.9 {
        uv::Vec3::new(0.0, 0.0, 0.0)
    } else {
        sun
    };

    2.0 * (sky_sample + 2.0 * sun_sample)
}

fn trace_ray(ray: &Ray, scene: &dyn Hittable, depth: i32) -> uv::Vec3 {
    let mut col = uv::Vec3::new(1.0, 1.0, 1.0);
    let mut working_ray: Ray = Ray { o: ray.o, d: ray.d };

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
    h = ray.d.cross(edge2);
    a = edge1.dot(h);
    if a > -EPSILON && a < EPSILON {
        return f32::MAX;
    } // This ray is parallel to this triangle.

    f = 1.0 / a;
    s = ray.o - tri.p0;
    u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return f32::MAX;
    }

    q = s.cross(edge1);
    v = f * ray.d.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return f32::MAX;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = f * edge2.dot(q);
    if t > EPSILON
    // ray intersection
    {
        return t;
    }
    // This means that there is a line intersection but not a ray intersection.
    f32::MAX
}

fn tri_intersect_8(tri: &Trix8, ray_single: &Ray) -> uv::f32x8 {
    let epsilon_x8: uv::f32x8 = uv::f32x8::splat(EPSILON);
    let ray: Rayx8 = Rayx8::splat(ray_single);
    let (edge1, edge2, h, s, q): (uv::Vec3x8, uv::Vec3x8, uv::Vec3x8, uv::Vec3x8, uv::Vec3x8);
    let (a, f, u, v): (uv::f32x8, uv::f32x8, uv::f32x8, uv::f32x8);
    edge1 = tri.p1 - tri.p0;
    edge2 = tri.p2 - tri.p0;
    h = ray.d.cross(edge2);
    a = edge1.dot(h);

    let invalid = a.cmp_gt(-epsilon_x8) & a.cmp_lt(epsilon_x8);
    // if -EPSILON < a && a < EPSILON
    //     { return f32::MAX; }   // This ray is parallel to this triangle.

    f = uv::f32x8::ONE / a;
    s = ray.o - tri.p0;
    u = f * s.dot(h);

    let invalid = invalid | u.cmp_lt(uv::f32x8::ZERO) | u.cmp_gt(uv::f32x8::ONE);
    // if u < 0.0 || u > 1.0
    //    { return f32::MAX; }

    q = s.cross(edge1);
    v = f * ray.d.dot(q);

    let invalid = invalid | v.cmp_lt(uv::f32x8::ZERO) | (u + v).cmp_gt(uv::f32x8::ONE);
    // if v < 0.0 || u + v > 1.0
    //    { return f32::MAX; }

    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = f * edge2.dot(q);

    // This means that there is a line intersection but not a ray intersection.
    let invalid = invalid | t.cmp_le(epsilon_x8) | (tri.p0.x * uv::f32x8::ZERO);
    // if t <= EPSILON
    //    { return f32::MAX; }

    invalid.blend(uv::f32x8::splat(f32::MAX), t)
}

fn tri_vec_intersect(input: (&Vec<Tri>, &Ray)) -> Vec<f32> {
    let mut res: Vec<f32> = Vec::with_capacity(input.0.len());
    for tri in input.0 {
        res.push(tri_intersect(tri, input.1));
    }
    return res;
}

fn tri_vec_intersect_8(input: (&Trix8, &Ray)) -> uv::f32x8 {
    tri_intersect_8(input.0, input.1)
}

fn model_loader(model_string: &str) -> Vec<Trix8> {
    println!("Loading Model");
    // Create a path to the file // sponza_simple
    let path = Path::new(model_string);
    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(&path) {
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

    println!("Creating Clusters");
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
    println!("Packing Cluster Data");
    let mut triangle_clusters: Vec<Trix8> = Vec::new();
    for (n, cluster) in clusters.iter().enumerate() {
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
    println!("Loading Done!");

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
#[clap(author = "Robert Chrisite", about = "Simple raytracer written in rust")]
struct CliArguments {
    #[clap(short = 'w', long, default_value = "512")]
    width: u32,

    #[clap(short = 'h', long, default_value = "512")]
    height: u32,

    #[clap(short = 's', long, default_value = "32")]
    samples: u32,

    #[clap(short = 's', long, default_value = "render.png")]
    output: String,
}

fn main() {
    let args = CliArguments::parse();
    let output_path = Path::new(&args.output);
    ctrlc::set_handler(move || {
        let mut stdout = stdout();
        stdout.execute(cursor::Show).unwrap();
        panic!();
    })
    .unwrap();

    // Scene creation
    let test_floor = PlaneRenderObject {
        plane: Plane {
            o: uv::Vec3::new(0.0, 0.0, 0.0),
            n: uv::Vec3::new(0.0, 1.0, 0.0),
        },
        mat: Rc::new(Diffuse {
            col: uv::Vec3::new(1.0, 1.0, 1.0),
            roughness: 1.0,
        }),
    };

    //let teapot_tris = model_loader("models/teapot.obj");
    let sponza_tris = model_loader("models/sponza_simple.obj");

    let mut scene_model: Vec<Box<dyn Hittable>> = Vec::with_capacity(sponza_tris.len());

    let sponza_mat: Rc<dyn Material> = Rc::new(Diffuse {
        col: uv::Vec3::new(0.7, 0.7, 0.7),
        roughness: 1.0,
    });

    // for tri_cluster in sponza_tris.into_iter() {
    //     let tau: [f32; 8] = uv::f32x8::TAU.into();
    //     let tau = tau[0];
    //     let hue = fastrand::f32() * tau;
    //     let col = 1.25
    //         * uv::Vec3::new(
    //             0.5 + 0.5 * f32::sin(hue),
    //             0.5 + 0.5 * f32::sin(hue + tau / 3.0),
    //             0.5 + 0.5 * f32::sin(hue + 2.0 * tau / 3.0),
    //         );
    //     scene_model.push(Box::new(TriClusterRenderObject {
    //         tris: Trix8 {
    //             p0: tri_cluster.p0 / uv::f32x8::splat(2.0),
    //             p1: tri_cluster.p1 / uv::f32x8::splat(2.0),
    //             p2: tri_cluster.p2 / uv::f32x8::splat(2.0),
    //             n: tri_cluster.n,
    //         },
    //         mat: Rc::new(Emmisive { col: 2.0 * col }), //mat: Rc::clone(&tea_pot_mat)
    //     }));
    // }

    for tri_cluster in sponza_tris.into_iter() {
        // let tau: [f32; 8] = uv::f32x8::TAU.into();
        // let tau = tau[0];
        // let hue = fastrand::f32() * tau;

        scene_model.push(Box::new(TriClusterRenderObject {
            tris: tri_cluster,
            mat: Rc::clone(&sponza_mat),
        }));
    }

    let mut render_object_bvh = RenderObjectBVH {
        objects: scene_model,
        nodes: vec![],
        mat: Rc::new(Emmisive {
            col: uv::Vec3::new(fastrand::f32(), fastrand::f32(), fastrand::f32()),
        }),
    };

    render_object_bvh.update_bvh();

    let render_object_list = RenderObjectList {
        objects: vec![Box::new(render_object_bvh), Box::new(test_floor)],
    };

    let mut img_buf = image::ImageBuffer::new(args.width, args.height);

    println!("Rendering");
    let mut stdout = stdout();

    stdout.execute(cursor::Hide).unwrap();
    let start_time = Instant::now();
    for y in 0..args.height {
        for x in 0..args.width {
            let ray: Ray = Ray {
                o: uv::Vec3::new(5.0, 2.0, 0.0),
                d: uv::Vec3::new(
                    -1.0,
                    1.0 - 2.0 * (y as f32) / (args.height as f32),
                    2.0 * (x as f32) / (args.width as f32) - 1.0,
                )
                .normalized(),
            };
            let mut col: uv::Vec3 = uv::Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..args.samples {
                col += trace_ray(&ray, &render_object_list, 4) / (args.samples as f32);
            }

            let pixel = img_buf.get_pixel_mut(x, y);
            *pixel = image::Rgb([
                (col.x * 255.0) as u8,
                (col.y * 255.0) as u8,
                (col.z * 255.0) as u8,
            ]);
        }

        let elapsed = start_time.elapsed().as_secs_f32();
        let total = (elapsed / y as f32) * args.height as f32;

        let term_width = terminal::size().map(|(w, _)| w).unwrap_or(32) as usize;
        let msg_time = format!("Rendering: {elapsed:0.2}s / {total:0.2}s");
        let progress_width = (term_width - msg_time.len() - 3).max(8);
        let render_progress =
            (progress_width as f32 * y as f32 / args.height as f32).ceil() as usize;

        let msg_progress = (0..progress_width)
            .map(|i| if i <= render_progress { '#' } else { ' ' })
            .collect::<String>();

        stdout.queue(cursor::SavePosition).unwrap();
        stdout
            .write_all(format!("{msg_time} [{msg_progress}]").as_bytes())
            .unwrap();
        stdout.queue(cursor::RestorePosition).unwrap();
        stdout.flush().unwrap();

        stdout.queue(cursor::RestorePosition).unwrap();
        stdout
            .queue(terminal::Clear(terminal::ClearType::FromCursorDown))
            .unwrap();
    }
    let duration = start_time.elapsed().as_secs_f32();
    stdout.execute(cursor::Show).unwrap();
    print!("\rRendered in {duration:0.2}s");

    img_buf.save(output_path).unwrap();
}
