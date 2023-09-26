use ultraviolet::Vec3;

use crate::{
    primatives::{Aabb, Ray},
    Material,
};

pub trait Hittable {
    fn ray_test(&self, ray: &Ray) -> Option<Hit>;
    fn bounding_volume(&self) -> Aabb;
}

pub struct Hit<'a> {
    pub t: f32,
    pub pos: Vec3,
    pub norm: Vec3,
    pub mat: &'a (dyn Material + Send + Sync),
}
