#![feature(test)]

use ultraviolet as uv;
use fastrand;
use uv::Lerp;

extern crate test;
use test::Bencher;
use uv::Vec3;

const EPSILON:  f32 = 0.00001;


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

type Aabb = (uv::Vec3, uv::Vec3);

// Render Objects 
trait Hittable{
    fn ray_test(&self, ray: &Ray) -> Option<Hit>;
}

trait Bounded{
    fn bounding_volume(&self) -> Aabb;
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

struct TriRenderObject{
    tri: Tri, 
    mat: Box<dyn Material>
}

impl Hittable for TriRenderObject{
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: f32 = tri_intersect(&self.tri, ray);
        if t==f32::MAX
            { return None }
        let pos = ray.o+ray.d*t;
        let norm = self.tri.n;

        Some(Hit{
            t,
            pos,
            norm,
            mat: &*self.mat
        })
    }
}

struct TriClusterRenderObject{
    tris: Trix8, 
    mat: Box<dyn Material>
}

impl Hittable for TriClusterRenderObject{
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        let t: uv::f32x8 = tri_intersect_8(&self.tris, ray);
        let t_in: [f32; 8] = t.into();
        let (i, t) = t_in.into_iter().enumerate().reduce(
            |accum, item| {
                if accum.1 <= item.1 { accum } else { item }
            }
        ).unwrap();
        
        if t==f32::MAX
            { return None; }

        let pos = ray.o+ray.d*t;
        let norm: uv::Vec3 =self.tris.n[i];

        Some(Hit{
            t,
            pos,
            norm,
            mat: &*self.mat
        })
    }
}

impl Bounded for TriClusterRenderObject{
    fn bounding_volume(&self) -> Aabb {
        let start = self.tris.p0.
            min_by_component(self.tris.p1).
            min_by_component(self.tris.p2);
        
        let end = self.tris.p0.
            max_by_component(self.tris.p1).
            max_by_component(self.tris.p2);

        
        ( uv::Vec3::new(
                <[f32; 8]>::from(start.x).into_iter().reduce(f32::min).unwrap()-EPSILON,
                <[f32; 8]>::from(start.y).into_iter().reduce(f32::min).unwrap()-EPSILON,
                <[f32; 8]>::from(start.z).into_iter().reduce(f32::min).unwrap()-EPSILON
            ),uv::Vec3::new(
                <[f32; 8]>::from(end.x).into_iter().reduce(f32::max).unwrap()+EPSILON,
                <[f32; 8]>::from(end.y).into_iter().reduce(f32::max).unwrap()+EPSILON,
                <[f32; 8]>::from(end.z).into_iter().reduce(f32::max).unwrap()+EPSILON
        ))
        
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



struct BvhNode{
    bound: Option<Aabb>,
    subnodes: Option<Vec<BvhNode>>,
    leaf: Option<u32>
}

impl BvhNode{
    fn bin_children(mut self) -> Self{
        let (a,e) = RenderObjectBVH::split_nodes(self.subnodes.unwrap(), 0);
        let (a,c) = RenderObjectBVH::split_nodes(a, 1);
        let (a,b) = RenderObjectBVH::split_nodes(a, 2);
        let (c,d) = RenderObjectBVH::split_nodes(c, 2);

        let (e,g) = RenderObjectBVH::split_nodes(e, 1);
        let (e,f) = RenderObjectBVH::split_nodes(e, 2);
        let (g,h) = RenderObjectBVH::split_nodes(g, 2);
 
        self.subnodes = Some(Vec::new());


        for nodes in [a,b,c,d,e,f,g,h]{
            if nodes.len() > 0{
                self.subnodes.as_mut().unwrap().push(
                    BvhNode{
                        bound: None,
                        subnodes: Some(nodes),
                        leaf: None
                    }
                )
            }
        }
        self
    }
}

impl Bounded for BvhNode{
    fn bounding_volume(self: &BvhNode) -> Aabb {
        let mut min = uv::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = uv::Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        for subnode in self.subnodes.as_ref().unwrap().iter() {
            if subnode.bound.is_none() {
                subnode.bounding_volume();
            }
            min = min.min_by_component(subnode.bound.unwrap().0);
            max = max.min_by_component(subnode.bound.unwrap().0);
        }
        (min, max)
    }
}

struct RenderObjectBVH{
    objects: Vec<Box<dyn Bounded>>,
    nodes: Vec<BvhNode>,
}

impl RenderObjectBVH{

    fn split_nodes(mut nodes: Vec<BvhNode>, axis: usize) -> (Vec<BvhNode>, Vec<BvhNode>){
        nodes.sort_by(|a, b|{
            (a.bound.unwrap().0[axis]+a.bound.unwrap().1[axis]).
            partial_cmp(&(b.bound.unwrap().0[axis]+b.bound.unwrap().1[axis])).
            unwrap()
        });

        let split  = nodes.split_off(nodes.len()/2);
        (nodes,split)

    }



    fn update_bvh(&mut self){
        let mut leaves: Vec<BvhNode> = Vec::new();
        for (i, object) in self.objects.iter().enumerate(){
            leaves.push(
                BvhNode { 
                    bound: Some(object.bounding_volume()),
                    leaf: Some(i as u32),
                    subnodes: None
                }
            )
        }

        self.root = BvhNode{
            bound: None,
            subnodes: Some(leaves),
            leaf: None
        };

        let mut remaining_nodes = vec![&self.root];

        while remaining_nodes.len() > 0 {
            let parent = remaining_nodes.pop().unwrap();
            parent.bin_children();
            for node in parent.subnodes.unwrap().iter(){
                if node.subnodes.as_ref().unwrap().len()>1{
                    remaining_nodes.push(node);
                }
            }
        }
        
        self.root.bounding_volume();
    }
}

impl Hittable for RenderObjectBVH{
    fn ray_test(&self, ray: &Ray) -> Option<Hit> {
        None
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
            o: hit.pos + hit.norm * 0.00001,
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
            o: hit.pos+hit.norm*0.00001,
            d: (hit.norm*1.00001 + random_unit.normalized()).normalized()
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

// BVH AABB 
fn aabb_hit(r: Ray, t_min: f32, t_max: f32, min: uv::Vec3, max: uv::Vec3) -> bool {
    for a in 0..3 {
        let invD = 1.0 / r.d[a];
        let mut t0 = (min[a] - r.o[a]) * invD;
        let mut t1 = (max[a] - r.o[a]) * invD;
        
        if invD < 0.0
        {
            (t0, t1) = (t1,t0);
        }

        let t_min = if t0 > t_min {t0} else {t_min};
        let t_max = if t1 < t_max {t1} else {t_max};
        if t_max <= t_min {
            return false;
        }
    }
    true
}



fn sample_sky(ray: &Ray) -> uv::Vec3{
    let apex = uv::Vec3::new(0.5,0.7,0.8);
    let horizon = uv::Vec3::new(1.0,1.0,1.0);
    let ground = uv::Vec3::new(0.0, 0.0, 0.0);
    
    let sun = uv::Vec3::new(1.0,0.9,0.9);
    let sun_dir = uv::Vec3::new(0.5,1.0,1.0).normalized();


    let sky_sample = horizon.lerp(apex, ray.d.y.clamp(0.0,1.0)).lerp(ground, (-5.0 * ray.d.y).clamp(0.0, 1.0).powf(0.5));
    let sun_sample = if ray.d.dot(sun_dir) < 0.9 { uv::Vec3::new(0.0,0.0,0.0)}else{sun} ;

    0.0*sky_sample + 2.0 * 0.0 *sun_sample
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



struct Tri{
    p0: uv::Vec3,
    p1: uv::Vec3,
    p2: uv::Vec3,
    n: uv::Vec3
}
struct Trix8{
    p0: uv::Vec3x8,
    p1: uv::Vec3x8,
    p2: uv::Vec3x8, 
    n:  [uv::Vec3;8]
}

struct Rayx8{
    o: uv::Vec3x8,
    d: uv::Vec3x8
}

trait Splat{
    fn splat(ray: &Ray) -> Rayx8;
}

impl Splat for Rayx8{
    fn splat(ray: &Ray) -> Rayx8{
        Rayx8 { 
            o: uv::Vec3x8::splat(ray.o),
            d: uv::Vec3x8::splat(ray.d)
        }
    }
}

fn tri_intersect(tri: &Tri, ray: &Ray) -> f32 {
    let (edge1, edge2, h, s, q): (uv::Vec3, uv::Vec3, uv::Vec3, uv::Vec3, uv::Vec3);
    let (a,f,u,v): (f32, f32, f32, f32);
    edge1 = tri.p1 - tri.p0;
    edge2 = tri.p2 - tri.p0;
    h = ray.d.cross(edge2);
    a = edge1.dot(h);
    if a > -EPSILON && a < EPSILON
        { return f32::MAX; }   // This ray is parallel to this triangle.
    
    f = 1.0/a;
    s = ray.o - tri.p0;
    u = f * s.dot(h);
    if u < 0.0 || u > 1.0
        { return f32::MAX; }
    
    q = s.cross(edge1);
    v = f * ray.d.dot(q);
    if v < 0.0 || u + v > 1.0
        { return f32::MAX; }
    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = f * edge2.dot(q);
    if t > EPSILON // ray intersection
    {
        return t;
    }
    // This means that there is a line intersection but not a ray intersection.
    f32::MAX
}

fn tri_intersect_8(tri: &Trix8, ray_single: &Ray) -> uv::f32x8 {
    let EPSILONx8:  uv::f32x8 = uv::f32x8::splat(0.00001);
    let ray: Rayx8= Rayx8::splat(ray_single);
    let (edge1, edge2, h, s, q): (uv::Vec3x8, uv::Vec3x8, uv::Vec3x8, uv::Vec3x8, uv::Vec3x8);
    let (a,f,u,v): (uv::f32x8, uv::f32x8, uv::f32x8, uv::f32x8);
    edge1 = tri.p1 - tri.p0;
    edge2 = tri.p2 - tri.p0;
    h = ray.d.cross(edge2);
    a = edge1.dot(h);


    let invalid = a.cmp_gt(-EPSILONx8) & a.cmp_lt(EPSILONx8);
    // if -EPSILON < a && a < EPSILON
    //     { return f32::MAX; }   // This ray is parallel to this triangle.
    
    f = uv::f32x8::ONE/a;
    s = ray.o - tri.p0;
    u = f * s.dot(h);

    let invalid = invalid | 
        u.cmp_lt(uv::f32x8::ZERO) | 
        u.cmp_gt(uv::f32x8::ONE);
    // if u < 0.0 || u > 1.0
    //    { return f32::MAX; }
    
    q = s.cross(edge1);
    v = f * ray.d.dot(q);

    let invalid = invalid | 
        v.cmp_lt(uv::f32x8::ZERO) |
        (u+v).cmp_gt(uv::f32x8::ONE);
    // if v < 0.0 || u + v > 1.0
    //    { return f32::MAX; }

    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = f * edge2.dot(q);

    // This means that there is a line intersection but not a ray intersection.
    let invalid = invalid | t.cmp_le(EPSILONx8);
    // if t <= EPSILON 
    //    { return f32::MAX; }
    
    invalid.blend(uv::f32x8::splat(f32::MAX), t)
}

fn tri_vec_intersect( 
    input: (&Vec<Tri>, &Ray)

) -> Vec<f32>{
    let mut res: Vec<f32> = Vec::with_capacity(input.0.len());
    for tri in input.0{
        res.push(tri_intersect(tri, input.1));
    }
    return res;
}

fn tri_vec_intersect_8(
    input: (&Trix8, & Ray)
) -> uv::f32x8{
    tri_intersect_8(input.0, input.1)
}

//#[bench]
// fn bench_non_simd_tris(b: &mut Bencher){
//     let test_ray: Ray = Ray { 
//         o: uv::Vec3::new( 0.0, 0.0, 0.0),
//         d: uv::Vec3::new( 0.0, 0.0,-1.0)
//     };

//     let test_tris: Vec<Tri> = vec![
//         Tri{
//             p0: uv::Vec3::new(0.0, 0.0, -1.0),
//             p1: uv::Vec3::new(1.0, 0.0, -1.0),
//             p2: uv::Vec3::new(0.0, 1.0, -1.0)
//         },
//         Tri{
//             p0: uv::Vec3::new(1.0, 0.0, -1.0),
//             p1: uv::Vec3::new(1.0, 1.0, -1.0),
//             p2: uv::Vec3::new(0.0, 1.0, -1.0)
//         },
//         Tri{
//             p0: uv::Vec3::new(1.0, 0.0, -1.0),
//             p1: uv::Vec3::new(2.0, 0.0, -1.0),
//             p2: uv::Vec3::new(1.0, 1.0, -1.0)
//         },
//         Tri{
//             p0: uv::Vec3::new(2.0, 0.0, -1.0),
//             p1: uv::Vec3::new(2.0, 1.0, -1.0),
//             p2: uv::Vec3::new(1.0, 1.0, -1.0)
//         },        
//         Tri{
//             p0: uv::Vec3::new(0.0, 0.0, -2.0),
//             p1: uv::Vec3::new(1.0, 0.0, -2.0),
//             p2: uv::Vec3::new(0.0, 1.0, -2.0)
//         },
//         Tri{
//             p0: uv::Vec3::new(1.0, 0.0, -2.0),
//             p1: uv::Vec3::new(1.0, 1.0, -2.0),
//             p2: uv::Vec3::new(0.0, 1.0, -2.0)
//         },
//         Tri{
//             p0: uv::Vec3::new(1.0, 0.0, -2.0),
//             p1: uv::Vec3::new(2.0, 0.0, -2.0),
//             p2: uv::Vec3::new(1.0, 1.0, -2.0)
//         },
//         Tri{
//             p0: uv::Vec3::new(2.0, 0.0, -2.0),
//             p1: uv::Vec3::new(2.0, 1.0, -2.0),
//             p2: uv::Vec3::new(1.0, 1.0, -2.0)
//         }
//         ];

//         b.iter(||{
//             Some((&test_tris, &test_ray)).into_iter().map(tri_vec_intersect).collect::<Vec<Vec<f32>>>()
//         });

// }

fn create_test_tri_cluster() -> Trix8 {
    let test_tris: Vec<Tri> = vec![
        Tri{
            p0: uv::Vec3::new(0.0, 0.0, -2.0), 
            p1: uv::Vec3::new(1.0, 0.0, -2.0), 
            p2: uv::Vec3::new(0.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },
        Tri{
            p0: uv::Vec3::new(1.0, 0.0, -2.0), 
            p1: uv::Vec3::new(1.0, 1.0, -2.0), 
            p2: uv::Vec3::new(0.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },
        Tri{
            p0: uv::Vec3::new(1.0, 0.0, -2.0), 
            p1: uv::Vec3::new(2.0, 0.0, -2.0), 
            p2: uv::Vec3::new(1.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },
        Tri{
            p0: uv::Vec3::new(0.0, 1.0, -2.0), 
            p1: uv::Vec3::new(1.0, 1.0, -2.0), 
            p2: uv::Vec3::new(0.0, 2.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },        
        Tri{
            p0: uv::Vec3::new(0.0, 0.0, -2.0), 
            p1: uv::Vec3::new(1.0, 0.0, -2.0), 
            p2: uv::Vec3::new(0.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },
        Tri{
            p0: uv::Vec3::new(0.0, 0.0, -2.0), 
            p1: uv::Vec3::new(1.0, 0.0, -2.0), 
            p2: uv::Vec3::new(0.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },
        Tri{
            p0: uv::Vec3::new(0.0, 0.0, -2.0), 
            p1: uv::Vec3::new(1.0, 0.0, -2.0), 
            p2: uv::Vec3::new(0.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },
        Tri{
            p0: uv::Vec3::new(0.0, 0.0, -2.0), 
            p1: uv::Vec3::new(1.0, 0.0, -2.0), 
            p2: uv::Vec3::new(0.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        }
        ];


        let mut p0: [uv::Vec3; 8] = Default::default();
        let mut p1: [uv::Vec3; 8] = Default::default();
        let mut p2: [uv::Vec3; 8] = Default::default();
        let mut n:  [uv::Vec3; 8] = Default::default();
        for (i, tri) in test_tris.iter().enumerate(){
            p0[i] = tri.p0;
            p1[i] = tri.p1;
            p2[i] = tri.p2;
            n[i]  = tri.n;
        }

        Trix8{
            p0: uv::Vec3x8::from(p0),
            p1: uv::Vec3x8::from(p1),
            p2: uv::Vec3x8::from(p2),
            n
        }
}

// #[bench]
// fn bench_simd_tris(b: &mut Bencher){
//     let test_ray: Ray = Ray { 
//         o: uv::Vec3::new( 0.0, 0.0, 0.0),
//         d: uv::Vec3::new( 0.0, 0.0,-1.0)
//     };
//         b.iter(||{
//             Some((&create_test_tri_cluster(), &test_ray)).into_iter().map(tri_vec_intersect_8).collect::<Vec<uv::f32x8>>()
//         });
// }

use std::default;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::collections::hash_set::HashSet;

fn model_loader() -> Vec<Trix8>
{
    println!("Loading Model");
    // Create a path to the file
    let path = Path::new("models/teapot.obj");
    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };

    let reader = BufReader::new(file);
    
    let mut faces: Vec<(u32, u32, u32)> = Vec::new();
    let mut verts: Vec<uv::Vec3> = Vec::new();
    let mut faces_on_vert:Vec<Vec<u32>> = Vec::new();

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
            },
            Some("f") => {
                let attached_verts = (
                    line_iter.next().unwrap().parse::<u32>().unwrap()-1,
                    line_iter.next().unwrap().parse::<u32>().unwrap()-1,
                    line_iter.next().unwrap().parse::<u32>().unwrap()-1,
                );
                let i = faces.len() as u32;
                faces_on_vert[attached_verts.0 as usize].push(i);
                faces_on_vert[attached_verts.1 as usize].push(i);
                faces_on_vert[attached_verts.2 as usize].push(i);
                faces.push(attached_verts);
            },
            Some(&_) => {},
            None => {}
        }
    }

    println!("Creating Clusters");
    let mut remaining_faces: HashSet<u32> = (0..faces.len() as u32).collect();
    let mut clusters: Vec<Vec<u32>> = Vec::new();
    while remaining_faces.len() > 0 {
        let mut i: u32 = 0;
        
        let mut cluster: Vec<u32> = vec![remaining_faces.take(&remaining_faces.iter().next().cloned().unwrap()).unwrap()];

        let mut connected_faces = Vec::new();
            


        'cluster_loop: while cluster.len() < 8{
            while connected_faces.len() == 0 {
                if i==cluster.len() as u32 {
                    break 'cluster_loop;
                }
                let face_verts = faces[cluster[i as usize] as usize];
                connected_faces.append(&mut faces_on_vert[face_verts.0 as usize].clone());
                connected_faces.append(&mut faces_on_vert[face_verts.1 as usize].clone());
                connected_faces.append(&mut faces_on_vert[face_verts.2 as usize].clone());
                i+=1;
 
                
            }
            let face: u32 = connected_faces.pop().unwrap();
            if connected_faces.contains(&face) || !remaining_faces.contains(&face)
                { continue;}
            remaining_faces.remove(&face);
            cluster.push(face)
        }
        clusters.push(cluster)
    }
    println!("Packing Cluster Data");
    let mut triangle_clusters: Vec<Trix8> = Vec::new();
    for cluster in clusters.iter() {
        let mut p0: [uv::Vec3; 8] = Default::default();
        let mut p1: [uv::Vec3; 8] = Default::default();
        let mut p2: [uv::Vec3; 8] = Default::default();
        let mut n:  [uv::Vec3; 8] = Default::default();
        for (i, v) in cluster.iter().enumerate() {
            let face = faces[*v as usize];
            let offset = uv::Vec3::new(0.0,-2.0,-5.0);
            p0[i] = verts[face.0 as usize]+offset;
            p1[i] = verts[face.1 as usize]+offset;
            p2[i] = verts[face.2 as usize]+offset;
            n[i]  = (p1[i]-p0[i]).cross(p2[i]-p0[i]).normalized();
        }
        triangle_clusters.push(Trix8{
            p0: uv::Vec3x8::from(p0),
            p1: uv::Vec3x8::from(p1),
            p2: uv::Vec3x8::from(p2),
            n
        });
    }
    println!("Loading Done!");
    triangle_clusters

}



fn main(){
    // Scene creation
    let test_floor = PlaneRenderObject{
        plane: Plane{
            o: uv::Vec3::new(0.0,-2.0,0.0),
            n: uv::Vec3::new(0.0,1.0,0.0)
        },
        mat: Box::new(Diffuse{
            col: uv::Vec3::new(1.0,1.0,1.0),
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
    
    let test_tri = TriRenderObject{
        tri: Tri { 
            p0: uv::Vec3::new(0.0, 0.0, -2.0), 
            p1: uv::Vec3::new(1.0, 0.0, -2.0), 
            p2: uv::Vec3::new(0.0, 1.0, -2.0), 
            n:  uv::Vec3::new(0.0,0.0,1.0)
        },
        mat: Box::new(Diffuse{
            col: uv::Vec3::new(0.5,1.0,0.5),
            roughness: 1.0
        })
    };
    let test_tri_cluster = TriClusterRenderObject{
        tris: create_test_tri_cluster(),
        mat: Box::new(Diffuse{
            col: uv::Vec3::new(0.5,1.0,0.5),
            roughness: 1.0
        })
    };

    let model_tris = model_loader();

    let mut test_model: Vec<Box<dyn Hittable>> = Vec::with_capacity(model_tris.len());
    for tri_cluster in model_tris.into_iter() {
        test_model.push(Box::new(
            TriClusterRenderObject { 
            tris: tri_cluster, 
            mat: Box::new(Emmisive{
                col: uv::Vec3::new(fastrand::f32(),fastrand::f32(),fastrand::f32()),
                }) 
            }
        ));
    }
    test_model.push(Box::new(test_floor));

    let renderObjectList = RenderObjectList{
            objects: test_model
        // objects: vec![
        //     //Box::new(test_floor),
        //     //Box::new(test_sphere),
        //     //Box::new(test_sphere_2),
        //     Box::new(test_model)]
    };

    let imgx = 512;
    let imgy = 512;
    let samples = 32;

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
