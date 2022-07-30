use ultraviolet as uv;
use fastrand;
use uv::Lerp;

// Basic data structs
struct Ray{
    o: uv::Vec3,
    d: uv::Vec3,
}

struct Sphere{
    o: uv::Vec3,
    r_sq: f32,
}

struct Plane{
    o: uv::Vec3,
    n: uv::Vec3
}

// Render Objects 
trait Hittable{
    fn ray_test(&self, ray: &Ray) -> Option<Hit>;
}

struct Hit<'a>{
    t: f32,
    pos: uv::Vec3,
    norm: uv::Vec3,
    mat: &'a dyn Material
}


struct NullRenderObject;

impl Hittable for NullRenderObject{
    fn ray_test(&self, _ray: &Ray) -> Option<Hit> {
        None
    }
}

struct PlaneRenderObject{
    plane: Plane,
    mat: Box<dyn Material>
}

impl PlaneRenderObject{
    fn normal(&self, pos: uv::Vec3) -> uv::Vec3 {
        self.plane.n
    }
}

impl Hittable for PlaneRenderObject{
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let div: f32 = ray.d.dot(self.plane.n);
        let t = (self.plane.o-ray.o).dot(self.plane.n)/div;
        if t>0.0{
            let pos = ray.o+ray.d*t;

            return Some(Hit{
                t,
                pos,
                norm: self.normal(pos),
                mat: &*self.mat
            })
        }
        None
    }
}

struct SphereRenderObject{
    sphere: Sphere,
    mat: Box<dyn Material>
}
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
    
    fn normal(&self, pos: uv::Vec3) -> uv::Vec3 {
        (pos - self.sphere.o).normalized() 
    }
}

impl Hittable for SphereRenderObject{ 
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = self.ray_sphere_intersect(ray);

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


struct RenderObjectList{
    objects: Vec<Box<dyn Hittable>> 
}

impl Hittable for RenderObjectList{
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        
        let mut hit: Option<Hit> = None;

        for (i, object) in self.objects.iter().enumerate(){
            let temp_hit = (&*object).ray_test(ray);
           
            if hit.is_some() { 
                if let Some(u_t_hit) = temp_hit {
                    let u_hit   = hit.as_ref().unwrap();
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
}


// Materials
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

impl Material for Glossy{
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<Ray> {
        Some(Ray{
            o: hit.pos + hit.norm * 0.000001,
            d: ray.d-2.0*(ray.d.dot(hit.norm))*hit.norm
        })
    }
    fn sample(&self, ray: &Ray, hit: &Hit) -> uv::Vec3 {
        self.col
    }
}

impl Material for Diffuse{   
    fn scatter(&self, _ray: &Ray, hit: &Hit) -> Option<Ray>
    {
        // Get random unit vector on sphere surface
        let mut random_unit: uv::Vec3;
        loop{
            let x = 2.0*fastrand::f32()-1.0;
            let y = 2.0*fastrand::f32()-1.0;
            let z = 2.0*fastrand::f32()-1.0;
            random_unit = uv::Vec3::new(x,y,z);
            if random_unit.mag_sq() <= 1.0 {
                break;
            }
        }
        
        Some(Ray{
            o: hit.pos+hit.norm*0.000001,
            d: (hit.norm + random_unit.normalized()).normalized()
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


fn sample_sky(ray: &Ray) -> uv::Vec3{
    let apex = uv::Vec3::new(0.5,0.7,0.8);
    let horizon = uv::Vec3::new(1.0,1.0,1.0);
    let ground = uv::Vec3::new(0.0, 0.0, 0.0);
    
    let sun = uv::Vec3::new(1.0,0.9,0.9);
    let sun_dir = uv::Vec3::new(0.5,1.0,1.0).normalized();


    let sky_sample = horizon.lerp(apex, ray.d.y.clamp(0.0,1.0)).lerp(ground, (-5.0 * ray.d.y).clamp(0.0, 1.0).powf(0.5));
    let sun_sample = if ray.d.dot(sun_dir) < 0.9 { uv::Vec3::new(0.0,0.0,0.0)}else{sun} ;

    sky_sample + 2.0 * sun_sample
}

fn trace_ray(ray: &Ray, scene:&dyn Hittable, depth: i32) -> uv::Vec3{ 
    let mut col = uv::Vec3::new(1.0,1.0,1.0);
    let mut working_ray: Ray = Ray{o:ray.o, d:ray.d}; 

    for _ in 0..depth{
        let hit: Option<Hit> = scene.ray_test(&working_ray);
        if hit.is_none(){
            return col*sample_sky(&working_ray);
        }
        let hit: Hit = hit.unwrap();

        col *= hit.mat.sample(&working_ray, &hit);
        let scatter_ray = hit.mat.scatter(&working_ray, &hit);
       
        if scatter_ray.is_none()
        {
            return col;
        }
        working_ray = scatter_ray.unwrap();
    }

    uv::Vec3::new(0.0,0.0,0.0) 
}



fn main(){
    // Scene creation
    let test_floor = PlaneRenderObject{
        plane: Plane{
            o: uv::Vec3::new(0.0,-1.0,0.0),
            n: uv::Vec3::new(0.0,1.0,0.0)
        },
        mat: Box::new(Diffuse{
            col: uv::Vec3::new(0.7,0.3,0.3),
            roughness: 1.0
        })
    };

    let test_sphere = SphereRenderObject{
        sphere: Sphere { o: uv::Vec3::new(0.0, 0.0, -3.0), r_sq: 1.0 },
        mat: Box::new(Glossy{
            col: uv::Vec3::new(0.7,0.7,0.7),
            roughness: 1.0
        })
    };

    let test_sphere_2 = SphereRenderObject{
        sphere: Sphere { o: uv::Vec3::new(1.6, 1.6, -3.0), r_sq: 1.0 },
        mat: Box::new(Diffuse{
            col: uv::Vec3::new(0.5,0.5,1.0),
            roughness: 1.0
        })
    };
    
    let renderObjectList = RenderObjectList{
        objects: vec![
            Box::new(test_floor),
            Box::new(test_sphere),
            Box::new(test_sphere_2)
        ]
    };

    let imgx = 512;
    let imgy = 512;
    let samples = 128;

    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);

    for y in 0..imgy {
        for x in 0..imgx {
            let ray: Ray = Ray{
                o: uv::Vec3::new(0.0, 0.0, 0.0),
                d: uv::Vec3::new(2.0*(x as f32)/(imgx as f32)-1.0, 1.0-2.0*(y as f32)/(imgy as f32), -1.0).normalized()
            };
            let mut col: uv::Vec3 = uv::Vec3::new(0.0,0.0,0.0);
            for  _ in 0..samples {
                col += trace_ray(&ray, &renderObjectList, 4)/(samples as f32);
            }


            let pixel = imgbuf.get_pixel_mut(x, y);
            *pixel = image::Rgb([(col.x*255.0) as u8, (col.y*255.0) as u8, (col.z*255.0) as u8]);
        }
        println!("{}",  y);
    }

    // Save the image as “fractal.png”, the format is deduced from the path
    imgbuf.save("render.png").unwrap();
}
