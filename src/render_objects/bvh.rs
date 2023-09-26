use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use ultraviolet::Vec3;

use crate::primatives::*;

use super::{Hit, Hittable};

struct BvhNodeBuilder {
    bound: Option<Aabb>,
    subnodes: Option<Vec<usize>>,
    subnode_bounds: Option<Aabb8>,
    leaf: Option<usize>,
}

#[derive(Clone, Copy)]
pub enum BvhNodeIndex {
    Node(usize),
    Leaf(usize),
    None,
}

#[derive(Clone)]
pub struct BvhNode {
    subnodes: [BvhNodeIndex; 8],
    subnode_bounds: Aabb8,
}

pub struct RenderObjectBvhBuilder {
    objects: Vec<Box<dyn Hittable + Send + Sync>>,
    nodes: Vec<BvhNodeBuilder>,
}

pub struct RenderObjectBVH {
    objects: Vec<Box<dyn Hittable + Send + Sync>>,
    nodes: Vec<BvhNode>,
}

impl RenderObjectBvhBuilder {
    pub fn new(objects: Vec<Box<dyn Hittable + Send + Sync>>) -> Self {
        Self {
            objects,
            nodes: vec![],
        }
    }

    pub fn build(self) -> RenderObjectBVH {
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
                    subnode_bounds: old_node.subnode_bounds.clone().unwrap(),
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
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        let mut s_min: [Vec3; 8] = [Vec3::new(f32::MAX, f32::MAX, f32::MAX); 8];
        let mut s_max: [Vec3; 8] = [Vec3::new(f32::MIN, f32::MIN, f32::MIN); 8];

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
        target_node.subnode_bounds = Some(Aabb8(s_min.into(), s_max.into()));
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

    pub fn update_bvh(&mut self) {
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

    pub fn dump_bvh(&self, file: &Path) {
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
            let hit_data: [f32; 8] = bvh_test_node.subnode_bounds.hit_test(&ray_inverse).into();

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
