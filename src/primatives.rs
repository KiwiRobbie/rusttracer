use ultraviolet::Vec3x8;
use ultraviolet::{f32x8, Vec3};

use crate::constants::EPSILON;

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

pub struct Ray8 {
    pub origin: Vec3x8,
    pub direction: Vec3x8,
}

impl Ray8 {
    pub fn splat(ray: &Ray) -> Ray8 {
        Ray8 {
            origin: Vec3x8::splat(ray.origin),
            direction: Vec3x8::splat(ray.direction),
        }
    }
}

pub struct RayInverse8 {
    pub origin: Vec3x8,
    pub inverse_direction: Vec3x8,
}

impl RayInverse8 {
    pub fn splat(ray: &Ray) -> RayInverse8 {
        RayInverse8 {
            origin: Vec3x8::splat(ray.origin),
            inverse_direction: Vec3x8::splat(Vec3::new(
                1.0 / ray.direction.x,
                1.0 / ray.direction.y,
                1.0 / ray.direction.z,
            )),
        }
    }
}

pub struct Triangle {
    pub p0: Vec3,
    pub p1: Vec3,
    pub p2: Vec3,
    pub n: Vec3,
}
impl Triangle {
    pub fn intersect(&self, ray: &Ray) -> f32 {
        let (edge1, edge2, h, s, q): (Vec3, Vec3, Vec3, Vec3, Vec3);
        let (a, f, u, v): (f32, f32, f32, f32);
        edge1 = self.p1 - self.p0;
        edge2 = self.p2 - self.p0;
        h = ray.direction.cross(edge2);
        a = edge1.dot(h);
        if a > -EPSILON && a < EPSILON {
            // This ray is parallel to this triangle.
            return f32::MAX;
        }

        f = 1.0 / a;
        s = ray.origin - self.p0;
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
}

pub struct Triangle8 {
    pub p0: Vec3x8,
    pub p1: Vec3x8,
    pub p2: Vec3x8,
    pub n0: [Vec3; 8],
    pub n1: [Vec3; 8],
    pub n2: [Vec3; 8],
}

impl Triangle8 {
    pub fn intersect(&self, ray_single: &Ray) -> f32x8 {
        let tri = self;
        let epsilon_x8: f32x8 = f32x8::splat(EPSILON);
        let ray: Ray8 = Ray8::splat(ray_single);
        let (edge1, edge2, h, s, q): (Vec3x8, Vec3x8, Vec3x8, Vec3x8, Vec3x8);
        let (a, f, u, v): (f32x8, f32x8, f32x8, f32x8);
        edge1 = tri.p1 - tri.p0;
        edge2 = tri.p2 - tri.p0;
        h = ray.direction.cross(edge2);
        a = edge1.dot(h);

        let invalid = a.cmp_gt(-epsilon_x8) & a.cmp_lt(epsilon_x8);

        f = f32x8::ONE / a;
        s = ray.origin - tri.p0;
        u = f * s.dot(h);

        let invalid = invalid | u.cmp_lt(f32x8::ZERO) | u.cmp_gt(f32x8::ONE);

        q = s.cross(edge1);
        v = f * ray.direction.dot(q);

        let invalid = invalid | v.cmp_lt(f32x8::ZERO) | (u + v).cmp_gt(f32x8::ONE);

        // At this stage we can compute t to find out where the intersection point is on the line.
        let t = f * edge2.dot(q);

        // This means that there is a line intersection but not a ray intersection.
        let invalid = invalid | t.cmp_le(epsilon_x8) | (tri.p0.x * f32x8::ZERO);

        invalid.blend(f32x8::splat(f32::MAX), t)
    }
}

pub struct Sphere {
    pub origin: Vec3,
    pub radius_squared: f32,
}

pub struct Plane {
    pub origin: Vec3,
    pub normal: Vec3,
}

pub type Aabb = (Vec3, Vec3);

#[derive(Clone)]
pub struct Aabb8(pub Vec3x8, pub Vec3x8);
impl Aabb8 {
    pub fn hit_test(&self, ray: &RayInverse8) -> f32x8 {
        let min = self.0;
        let max = self.1;
        let t1 = (min.x - ray.origin.x) * ray.inverse_direction.x;
        let t2 = (max.x - ray.origin.x) * ray.inverse_direction.x;
        let t3 = (min.y - ray.origin.y) * ray.inverse_direction.y;
        let t4 = (max.y - ray.origin.y) * ray.inverse_direction.y;
        let t5 = (min.z - ray.origin.z) * ray.inverse_direction.z;
        let t6 = (max.z - ray.origin.z) * ray.inverse_direction.z;

        let tmin = f32x8::max(
            f32x8::max(f32x8::min(t1, t2), f32x8::min(t3, t4)),
            f32x8::min(t5, t6),
        );
        let tmax = f32x8::min(
            f32x8::min(f32x8::max(t1, t2), f32x8::max(t3, t4)),
            f32x8::max(t5, t6),
        );

        tmax.cmp_lt(f32x8::ZERO) | tmax.cmp_lt(tmin) | tmin
    }
}
