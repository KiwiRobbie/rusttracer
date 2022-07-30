include!("render_objects.rs");


trait Material{
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray>;
    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3;
}

struct Diffuse{
    col: uv::Vec3, 
    roughness: f32,
}

struct Glossy{
    col: uv::Vec3,
    roughness: f32
}

struct Emmisive{
    col: uv::Vec3
}


impl Default for Diffuse{
    fn default() -> Self {
        Diffuse{ 
            col: uv::Vec3::new(0.0,0.0,0.0), 
            roughness: 0.0, 
        }
    }
}


impl Material for Diffuse{   
    fn scatter(&self, _ray: &Ray, hit: &Hit) -> Option<Ray>
    {
        // Get random unit vector on sphere surface
        let mut random_unit: uv::Vec3;
        loop{
            let x = fastrand::f32();
            let y = fastrand::f32();
            let z = fastrand::f32();
            random_unit = uv::Vec3::new(x,y,z);
            if random_unit.mag_sq() <= 1.0 {
                break;
            }
        }
        
        Some(Ray{
            o: hit.pos+hit.norm*f32::MIN,
            d: (hit.norm + random_unit.normalized()).normalized()
        })
    }

    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3 {
        self.col
    }
}


impl Material for Glossy{
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray> {
        Some(Ray{
            o: hit.pos + hit.norm * f32::MIN,
            d: hit.norm
        })
    }
    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3 {
        self.col
    }
}


impl Material for Emmisive{
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray> {
        None
    }
    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3 {
        self.col
    }
    
}
