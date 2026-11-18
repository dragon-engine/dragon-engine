use std::collections::HashMap;

use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};
use slotmap::{SecondaryMap, new_key_type};

use crate::effect::EffectAttachment;

new_key_type! {
    /// Generational handle identifying an actor across AoS, SoA, and Rapier storage.
    pub struct ActorId;
}

/// Rigid-body simulation category for a spawned actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyKind {
    Dynamic,
    Kinematic,
    Fixed,
}

/// Axis-aligned bounding box in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb3 {
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

impl Aabb3 {
    pub fn from_center_half_extents(center: glam::Vec3, half: glam::Vec3) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn contains_point(&self, p: glam::Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

/// Result of a raycast against the physics world.
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    pub actor: Option<ActorId>,
    pub distance: f32,
    pub point: glam::Vec3,
    pub normal: glam::Vec3,
}

/// Deferred physics and visual mutations flushed at the end of each simulation step.
#[derive(Debug, Clone)]
pub enum PhysicsCommand {
    ApplyImpulse {
        actor: ActorId,
        impulse: glam::Vec3,
    },
    SetLinearVelocity {
        actor: ActorId,
        velocity: glam::Vec3,
    },
    SetPosition {
        actor: ActorId,
        position: glam::Vec3,
    },
    SetRotation {
        actor: ActorId,
        rotation: glam::Quat,
    },
    SetMeshColor {
        actor: ActorId,
        color: [f32; 3],
    },
    SetMeshEffect {
        actor: ActorId,
        effect: Option<EffectAttachment>,
    },
}

/// Read-only spatial query surface exposed to gameplay scripts during simulation polling.
pub trait SpatialQuery {
    fn raycast(
        &self,
        origin: glam::Vec3,
        dir: glam::Vec3,
        max_dist: f32,
        ignore: Option<ActorId>,
    ) -> Option<RaycastHit>;

    fn overlap_sphere(
        &self,
        center: glam::Vec3,
        radius: f32,
        ignore: Option<ActorId>,
    ) -> Vec<ActorId>;

    fn query_octree(&self, region: Aabb3) -> Vec<ActorId>;

    fn actor_position(&self, actor: ActorId) -> Option<glam::Vec3>;

    fn actor_bounds(&self, actor: ActorId) -> Option<Aabb3>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SoaMesh {
    pub half_extents: glam::Vec3,
    pub color: [f32; 3],
    pub visible: bool,
    pub effect: Option<EffectAttachment>,
}

/// Contiguous Structure-of-Arrays storage for spatial, physics, and render data.
/// Kept private inside `engine`.
pub(crate) struct SoaStorage {
    pub actor_ids: Vec<ActorId>,
    pub positions: Vec<glam::Vec3>,
    pub rotations: Vec<glam::Quat>,
    pub half_extents: Vec<glam::Vec3>,
    pub bounds: Vec<Aabb3>,
    pub colliders: Vec<ColliderHandle>,
    pub rigid_bodies: Vec<RigidBodyHandle>,
    pub meshes: Vec<SoaMesh>,
    pub body_kinds: Vec<BodyKind>,
    pub actor_to_index: SecondaryMap<ActorId, usize>,
}

impl SoaStorage {
    pub(crate) fn new() -> Self {
        Self {
            actor_ids: Vec::new(),
            positions: Vec::new(),
            rotations: Vec::new(),
            half_extents: Vec::new(),
            bounds: Vec::new(),
            colliders: Vec::new(),
            rigid_bodies: Vec::new(),
            meshes: Vec::new(),
            body_kinds: Vec::new(),
            actor_to_index: SecondaryMap::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push(
        &mut self,
        actor: ActorId,
        position: glam::Vec3,
        rotation: glam::Quat,
        half_extents: glam::Vec3,
        collider: ColliderHandle,
        rigid_body: RigidBodyHandle,
        color: [f32; 3],
        effect: Option<EffectAttachment>,
        body_kind: BodyKind,
    ) {
        let idx = self.actor_ids.len();
        self.actor_ids.push(actor);
        self.positions.push(position);
        self.rotations.push(rotation);
        self.half_extents.push(half_extents);
        self.bounds.push(
            Aabb3::from_center_half_extents(
                position,
                half_extents,
            ),
        );
        self.colliders.push(collider);
        self.rigid_bodies.push(rigid_body);
        self.meshes.push(
            SoaMesh {
                half_extents,
                color,
                visible: true,
                effect,
            },
        );
        self.body_kinds.push(body_kind);
        self.actor_to_index.insert(
            actor, idx,
        );
    }

    pub(crate) fn swap_remove(
        &mut self,
        actor: ActorId,
    ) -> Option<(
        RigidBodyHandle,
        ColliderHandle,
    )> {
        let idx = self.actor_to_index.remove(actor)?;
        let last = self.actor_ids.len() - 1;
        let removed_rb = self.rigid_bodies[idx];
        let removed_col = self.colliders[idx];

        self.actor_ids.swap_remove(idx);
        self.positions.swap_remove(idx);
        self.rotations.swap_remove(idx);
        self.half_extents.swap_remove(idx);
        self.bounds.swap_remove(idx);
        self.colliders.swap_remove(idx);
        self.rigid_bodies.swap_remove(idx);
        self.meshes.swap_remove(idx);
        self.body_kinds.swap_remove(idx);

        if idx < last {
            let moved_actor = self.actor_ids[idx];
            self.actor_to_index.insert(
                moved_actor,
                idx,
            );
        }

        Some(
            (
                removed_rb,
                removed_col,
            ),
        )
    }

    pub(crate) fn index_of(&self, actor: ActorId) -> Option<usize> {
        self.actor_to_index.get(actor).copied()
    }
}

/// Lightweight octree built over SoA `bounds` for spatial culling and broad queries.
#[derive(Debug, Clone)]
pub(crate) struct Octree {
    root: Option<OctreeNode>,
}

#[derive(Debug, Clone)]
struct OctreeNode {
    region: Aabb3,
    items: Vec<usize>,
    children: Option<Box<[OctreeNode; 8]>>,
}

impl Octree {
    const MAX_DEPTH: usize = 5;
    const MAX_ITEMS_PER_LEAF: usize = 8;

    pub(crate) fn rebuild(bounds: &[Aabb3]) -> Self {
        if bounds.is_empty() {
            return Self { root: None };
        }
        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        for b in bounds {
            min = min.min(b.min);
            max = max.max(b.max);
        }
        // Pad slightly to avoid degenerate zero-volume octrees.
        let root_region = Aabb3 {
            min: min - glam::Vec3::splat(1.0),
            max: max + glam::Vec3::splat(1.0),
        };
        let indices: Vec<usize> = (0..bounds.len()).collect();
        let root = Self::build_node(
            root_region,
            indices,
            bounds,
            0,
        );
        Self { root: Some(root) }
    }

    fn build_node(
        region: Aabb3,
        indices: Vec<usize>,
        bounds: &[Aabb3],
        depth: usize,
    ) -> OctreeNode {
        if indices.len() <= Self::MAX_ITEMS_PER_LEAF || depth >= Self::MAX_DEPTH {
            return OctreeNode {
                region,
                items: indices,
                children: None,
            };
        }

        let mid = (region.min + region.max) * 0.5;
        let child_regions = [
            Aabb3 {
                min: region.min,
                max: mid,
            },
            Aabb3 {
                min: glam::Vec3::new(
                    mid.x,
                    region.min.y,
                    region.min.z,
                ),
                max: glam::Vec3::new(
                    region.max.x,
                    mid.y,
                    mid.z,
                ),
            },
            Aabb3 {
                min: glam::Vec3::new(
                    region.min.x,
                    mid.y,
                    region.min.z,
                ),
                max: glam::Vec3::new(
                    mid.x,
                    region.max.y,
                    mid.z,
                ),
            },
            Aabb3 {
                min: glam::Vec3::new(
                    mid.x,
                    mid.y,
                    region.min.z,
                ),
                max: glam::Vec3::new(
                    region.max.x,
                    region.max.y,
                    mid.z,
                ),
            },
            Aabb3 {
                min: glam::Vec3::new(
                    region.min.x,
                    region.min.y,
                    mid.z,
                ),
                max: glam::Vec3::new(
                    mid.x,
                    mid.y,
                    region.max.z,
                ),
            },
            Aabb3 {
                min: glam::Vec3::new(
                    mid.x,
                    region.min.y,
                    mid.z,
                ),
                max: glam::Vec3::new(
                    region.max.x,
                    mid.y,
                    region.max.z,
                ),
            },
            Aabb3 {
                min: glam::Vec3::new(
                    region.min.x,
                    mid.y,
                    mid.z,
                ),
                max: glam::Vec3::new(
                    mid.x,
                    region.max.y,
                    region.max.z,
                ),
            },
            Aabb3 {
                min: mid,
                max: region.max,
            },
        ];

        let mut buckets: [Vec<usize>; 8] = Default::default();
        let mut staying = Vec::new();

        for idx in indices {
            let item_box = bounds[idx];
            let mut matched_child = None;
            for (c_idx, c_reg) in child_regions.iter().enumerate() {
                if c_reg.contains_point(item_box.min) && c_reg.contains_point(item_box.max) {
                    matched_child = Some(c_idx);
                    break;
                }
            }
            if let Some(c_idx) = matched_child {
                buckets[c_idx].push(idx);
            } else {
                staying.push(idx);
            }
        }

        let children_vec: Vec<OctreeNode> = child_regions
            .into_iter()
            .zip(buckets)
            .map(
                |(reg, items)| {
                    Self::build_node(
                        reg,
                        items,
                        bounds,
                        depth + 1,
                    )
                },
            )
            .collect();
        let children: [OctreeNode; 8] = children_vec.try_into().unwrap_or_else(
            |v: Vec<OctreeNode>| {
                panic!(
                    "Octree must build 8 children, got {}",
                    v.len()
                )
            },
        );

        OctreeNode {
            region,
            items: staying,
            children: Some(Box::new(children)),
        }
    }

    pub(crate) fn query(&self, region: &Aabb3, out: &mut Vec<usize>) {
        if let Some(root) = &self.root {
            Self::query_node(
                root, region, out,
            );
        }
    }

    fn query_node(node: &OctreeNode, region: &Aabb3, out: &mut Vec<usize>) {
        if !node.region.intersects(region) {
            return;
        }
        out.extend_from_slice(&node.items);
        if let Some(children) = &node.children {
            for child in children.iter() {
                Self::query_node(
                    child, region, out,
                );
            }
        }
    }
}

/// Private Rapier3D physics world coupled with `SoaStorage` and `Octree`.
pub(crate) struct PhysicsWorld {
    pipeline: PhysicsPipeline,
    gravity: Vector,
    integration_parameters: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    soft_bodies: SoftBodySet,
    ccd_solver: CCDSolver,
    collider_to_actor: HashMap<ColliderHandle, ActorId>,
    pub soa: SoaStorage,
    pub octree: Octree,
}

impl PhysicsWorld {
    pub(crate) fn new() -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            gravity: Vector::new(
                0.0, -9.81, 0.0,
            ),
            integration_parameters: IntegrationParameters::default(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            soft_bodies: SoftBodySet::new(),
            ccd_solver: CCDSolver::new(),
            collider_to_actor: HashMap::new(),
            soa: SoaStorage::new(),
            octree: Octree::rebuild(&[]),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spawn_body(
        &mut self,
        actor: ActorId,
        position: glam::Vec3,
        rotation: glam::Quat,
        half_extents: glam::Vec3,
        color: [f32; 3],
        effect: Option<EffectAttachment>,
        body_kind: BodyKind,
        mass: f32,
    ) {
        let rb_builder = match body_kind {
            BodyKind::Dynamic => RigidBodyBuilder::dynamic().lock_rotations(),
            BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
            BodyKind::Fixed => RigidBodyBuilder::fixed(),
        };
        let rigid_body = rb_builder
            .pose(
                Pose::from_parts(
                    Vector::new(
                        position.x, position.y, position.z,
                    ),
                    Rotation::from_xyzw(
                        rotation.x, rotation.y, rotation.z, rotation.w,
                    ),
                ),
            )
            .linear_damping(4.0)
            .additional_mass(mass.max(0.1))
            .build();

        let rb_handle = self.bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(
            half_extents.x,
            half_extents.y,
            half_extents.z,
        )
        .friction(0.3)
        .restitution(0.05)
        .build();
        let col_handle = self.colliders.insert_with_parent(
            collider,
            rb_handle,
            &mut self.bodies,
        );

        self.collider_to_actor.insert(
            col_handle, actor,
        );
        self.soa.push(
            actor,
            position,
            rotation,
            half_extents,
            col_handle,
            rb_handle,
            color,
            effect,
            body_kind,
        );
        self.octree = Octree::rebuild(&self.soa.bounds);
    }

    pub(crate) fn despawn_body(&mut self, actor: ActorId) {
        if let Some((rb_handle, col_handle)) = self.soa.swap_remove(actor) {
            self.collider_to_actor.remove(&col_handle);
            self.bodies.remove(
                rb_handle,
                &mut self.islands,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.soft_bodies,
                true,
            );
            self.octree = Octree::rebuild(&self.soa.bounds);
        }
    }

    pub(crate) fn step(&mut self, dt_secs: f32) {
        if dt_secs > 0.0 {
            self.integration_parameters.dt = dt_secs.min(1.0 / 20.0);
            self.pipeline.step(
                self.gravity,
                &self.integration_parameters,
                &mut self.islands,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.soft_bodies,
                &mut self.ccd_solver,
                &(),
                &(),
            );
        }

        // Sync Rapier rigid-body poses into contiguous SoA arrays.
        for idx in 0..self.soa.actor_ids.len() {
            let rb_handle = self.soa.rigid_bodies[idx];
            if let Some(body) = self.bodies.get(rb_handle) {
                let pos = body.translation();
                let rot = body.rotation();
                let p = glam::Vec3::new(
                    pos.x, pos.y, pos.z,
                );
                let q = glam::Quat::from_xyzw(
                    rot.x, rot.y, rot.z, rot.w,
                );
                self.soa.positions[idx] = p;
                self.soa.rotations[idx] = q;
                self.soa.bounds[idx] = Aabb3::from_center_half_extents(
                    p,
                    self.soa.half_extents[idx],
                );
            }
        }

        self.octree = Octree::rebuild(&self.soa.bounds);
    }

    pub(crate) fn flush_commands(&mut self, commands: &mut Vec<PhysicsCommand>) {
        let mut bounds_dirty = false;
        for cmd in commands.drain(..) {
            match cmd {
                PhysicsCommand::ApplyImpulse { actor, impulse } => {
                    if let Some(idx) = self.soa.index_of(actor)
                        && let Some(body) = self.bodies.get_mut(self.soa.rigid_bodies[idx])
                    {
                        body.apply_impulse(
                            Vector::new(
                                impulse.x, impulse.y, impulse.z,
                            ),
                            true,
                        );
                    }
                }
                PhysicsCommand::SetLinearVelocity { actor, velocity } => {
                    if let Some(idx) = self.soa.index_of(actor)
                        && let Some(body) = self.bodies.get_mut(self.soa.rigid_bodies[idx])
                    {
                        body.set_linvel(
                            Vector::new(
                                velocity.x, velocity.y, velocity.z,
                            ),
                            true,
                        );
                    }
                }
                PhysicsCommand::SetPosition { actor, position } => {
                    if let Some(idx) = self.soa.index_of(actor) {
                        if let Some(body) = self.bodies.get_mut(self.soa.rigid_bodies[idx]) {
                            body.set_translation(
                                Vector::new(
                                    position.x, position.y, position.z,
                                ),
                                true,
                            );
                        }
                        self.soa.positions[idx] = position;
                        self.soa.bounds[idx] = Aabb3::from_center_half_extents(
                            position,
                            self.soa.half_extents[idx],
                        );
                        bounds_dirty = true;
                    }
                }
                PhysicsCommand::SetRotation { actor, rotation } => {
                    if let Some(idx) = self.soa.index_of(actor) {
                        if let Some(body) = self.bodies.get_mut(self.soa.rigid_bodies[idx]) {
                            body.set_rotation(
                                Rotation::from_xyzw(
                                    rotation.x, rotation.y, rotation.z, rotation.w,
                                ),
                                true,
                            );
                        }
                        self.soa.rotations[idx] = rotation;
                    }
                }
                PhysicsCommand::SetMeshColor { actor, color } => {
                    if let Some(idx) = self.soa.index_of(actor) {
                        self.soa.meshes[idx].color = color;
                    }
                }
                PhysicsCommand::SetMeshEffect { actor, effect } => {
                    if let Some(idx) = self.soa.index_of(actor) {
                        self.soa.meshes[idx].effect = effect;
                    }
                }
            }
        }
        if bounds_dirty {
            self.octree = Octree::rebuild(&self.soa.bounds);
        }
    }
}

impl SpatialQuery for PhysicsWorld {
    fn raycast(
        &self,
        origin: glam::Vec3,
        dir: glam::Vec3,
        max_dist: f32,
        ignore: Option<ActorId>,
    ) -> Option<RaycastHit> {
        let dir_norm = dir.normalize_or_zero();
        if dir_norm.length_squared() < f32::EPSILON {
            return None;
        }
        let ignore_col = ignore
            .and_then(|id| self.soa.index_of(id))
            .map(|idx| self.soa.colliders[idx]);

        let filter = match ignore_col {
            Some(col) => QueryFilter::default().exclude_collider(col),
            None => QueryFilter::default(),
        };

        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            filter,
        );

        let ray = Ray::new(
            Vector::new(
                origin.x, origin.y, origin.z,
            ),
            Vector::new(
                dir_norm.x, dir_norm.y, dir_norm.z,
            ),
        );

        query_pipeline
            .cast_ray_and_get_normal(
                &ray, max_dist, true,
            )
            .map(
                |(col_handle, hit)| {
                    let pt = ray.point_at(hit.time_of_impact);
                    RaycastHit {
                        actor: self.collider_to_actor.get(&col_handle).copied(),
                        distance: hit.time_of_impact,
                        point: glam::Vec3::new(
                            pt.x, pt.y, pt.z,
                        ),
                        normal: glam::Vec3::new(
                            hit.normal.x,
                            hit.normal.y,
                            hit.normal.z,
                        ),
                    }
                },
            )
    }

    fn overlap_sphere(
        &self,
        center: glam::Vec3,
        radius: f32,
        ignore: Option<ActorId>,
    ) -> Vec<ActorId> {
        let ignore_col = ignore
            .and_then(|id| self.soa.index_of(id))
            .map(|idx| self.soa.colliders[idx]);
        let filter = match ignore_col {
            Some(col) => QueryFilter::default().exclude_collider(col),
            None => QueryFilter::default(),
        };
        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            filter,
        );

        let shape = Ball::new(radius);
        let pose = Pose::translation(
            center.x, center.y, center.z,
        );
        let mut results = Vec::new();
        for (col_handle, _) in query_pipeline.intersect_shape(
            pose, &shape,
        ) {
            if let Some(&actor_id) = self.collider_to_actor.get(&col_handle) {
                results.push(actor_id);
            }
        }
        results
    }

    fn query_octree(&self, region: Aabb3) -> Vec<ActorId> {
        let mut indices = Vec::new();
        self.octree.query(
            &region,
            &mut indices,
        );
        indices
            .into_iter()
            .filter_map(|idx| self.soa.actor_ids.get(idx).copied())
            .collect()
    }

    fn actor_position(&self, actor: ActorId) -> Option<glam::Vec3> {
        let idx = self.soa.index_of(actor)?;
        self.soa.positions.get(idx).copied()
    }

    fn actor_bounds(&self, actor: ActorId) -> Option<Aabb3> {
        let idx = self.soa.index_of(actor)?;
        self.soa.bounds.get(idx).copied()
    }
}
