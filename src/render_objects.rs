struct SphereRenderObject{
    sphere: Sphere,
    mat: Box<dyn Material>
}


struct NullRenderObject;

impl SphereRenderObject{
 fn ray_sphere_intersect(
        &self,
        ray: &Ray,
        ) -> f32 {
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
}

impl Hittable for SphereRenderObject{
   
    fn normal(&self, pos: uv::Vec3) -> uv::Vec3 {
        return (pos - self.sphere.o).normalized() 
    }


    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = self.ray_sphere_intersect(&ray);

        if t==f32::MAX{
            return None;
        }

        let pos: uv::Vec3 = ray.o+ray.d*t; 
        let norm: uv::Vec3 = self.normal(pos);

        Some(Hit{
            t, 
            pos,
            norm,
            mat: &*self.mat
        })
    }
}



impl Hittable for NullRenderObject{
    fn normal(&self, _pos: uv::Vec3) -> uv::Vec3 {
        uv::Vec3::default()
    }

    fn ray_test(&self, _ray: &Ray) -> Option<Hit> {
        None
    }
}

struct Hit<'a>{
    t: f32,
    pos: uv::Vec3,
    norm: uv::Vec3,
    mat: &'a dyn Material
}

trait Hittable{
    fn ray_test(&self, ray: &Ray) -> Option<Hit>;
    fn normal(&self, pos: uv::Vec3) -> uv::Vec3;
}

