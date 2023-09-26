use std::sync::Arc;

use ultraviolet::{f32x8, Vec3};

use crate::{primatives::*, Material};

pub use self::hitable::{Hit, Hittable};

pub mod bvh;
pub mod hitable;

struct NullRenderObject;

impl Hittable for NullRenderObject {
    fn ray_test(&self, _ray: &Ray) -> Option<Hit> {
        None
    }
    fn bounding_volume(&self) -> Aabb {
        Default::default()
    }
}

pub struct PlaneRenderObject {
    pub plane: Plane,
    pub mat: Arc<(dyn Material + Send + Sync)>,
}

impl PlaneRenderObject {
    fn normal(&self, _pos: Vec3) -> Vec3 {
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

    fn normal(&self, pos: Vec3) -> Vec3 {
        (pos - self.sphere.origin).normalized()
    }
}

impl Hittable for SphereRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = self.ray_sphere_intersect(ray);

        if t == f32::MAX {
            return None;
        }

        let pos: Vec3 = ray.origin + ray.direction * t;
        let norm: Vec3 = self.normal(pos);

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
    tri: Triangle,
    mat: Arc<dyn Material + Send + Sync>,
}

impl Hittable for TriRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = self.tri.intersect(ray);
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

pub struct TriClusterRenderObject {
    pub tris: Triangle8,
    pub mat: Arc<dyn Material + Send + Sync>,
}

impl Hittable for TriClusterRenderObject {
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32x8 = self.tris.intersect(ray);
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
        // let norm: Vec3 = self.tris.n[i];

        let norm = {
            let Triangle8 {
                p0,
                p1,
                p2,
                n0,
                n1,
                n2,
            } = self.tris;

            let (a, b, c, n0, n1, n2) = (
                Into::<[Vec3; 8]>::into(p0)[i],
                Into::<[Vec3; 8]>::into(p1)[i],
                Into::<[Vec3; 8]>::into(p2)[i],
                n0[i],
                n1[i],
                n2[i],
            );
            let p = pos;
            let v0 = b - a;
            let v1 = c - a;
            let v2 = p - a;

            let d00: f32 = v0.dot(v0);
            let d01: f32 = v0.dot(v1);
            let d11: f32 = v1.dot(v1);
            let d20: f32 = v2.dot(v0);
            let d21: f32 = v2.dot(v1);
            let denom: f32 = d00 * d11 - d01 * d01;

            // Degenerate triangle?
            if denom == 0.0 {
                return None;
            }

            let v = (d11 * d20 - d01 * d21) / denom;
            let w = (d00 * d21 - d01 * d20) / denom;
            let u = 1.0f32 - v - w;
            n0 * u + n1 * v + n2 * w
        }
        .normalized();

        Some(Hit {
            t,
            pos,
            norm,
            mat: &*self.mat,
        })

        // if !hit.norm.x.is_finite() || !hit.norm.y.is_finite() || !hit.norm.z.is_finite() {
        //     dbg!(hit.norm);
        //     None
        // } else {
        //     Some(hit)
        // }
    }

    fn bounding_volume(&self) -> Aabb {
        let p0: [Vec3; 8] = self.tris.p0.into();
        let p1: [Vec3; 8] = self.tris.p1.into();
        let p2: [Vec3; 8] = self.tris.p2.into();

        let mut pts = p0.to_vec();
        pts.append(&mut p1.to_vec());
        pts.append(&mut p2.to_vec());
        let filtered = pts.iter().filter(|n| !n.x.is_nan()).collect::<Vec<&Vec3>>();

        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for v in filtered {
            min = min.min_by_component(v.clone());
            max = max.max_by_component(v.clone());
        }
        (min, max)
    }
}

pub struct RenderObjectList {
    pub objects: Vec<Box<dyn Hittable + Send + Sync>>,
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
