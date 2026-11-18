use crate::executor::TaskHandle;

/// Private layer inside `ModeManager` that owns mode-scoped `TaskHandle`s.
/// Popping a `ModeLayer` drops all attached `TaskHandle`s and cancels their scripts.
struct ModeLayer<S> {
    mode: S,
    // Kept solely for Drop-based task cancellation when a mode layer pops.
    #[allow(dead_code)]
    tasks: Vec<TaskHandle>,
}

/// Manages flat application states alongside a LIFO modal stack without losing stack history.
pub struct ModeManager<F, S> {
    flat: F,
    stack: Vec<ModeLayer<S>>,
    world_active: fn(&F, &[S]) -> bool,
}

impl<F: Clone, S: Clone> ModeManager<F, S> {
    /// Creates a `ModeManager` with an initial flat mode, root stack mode, and world-clock predicate.
    pub fn new(initial_flat: F, root_stack: S, world_active: fn(&F, &[S]) -> bool) -> Self {
        Self {
            flat: initial_flat,
            stack: vec![ModeLayer {
                mode: root_stack,
                tasks: Vec::new(),
            }],
            world_active,
        }
    }

    /// Returns the current flat mode.
    pub fn flat(&self) -> &F {
        &self.flat
    }

    /// Switches the flat mode without clearing the modal stack.
    pub fn set_flat(&mut self, flat: F) {
        self.flat = flat;
    }

    /// Returns the top mode on the stack, if non-empty.
    pub fn stack_top(&self) -> Option<&S> {
        self.stack.last().map(|layer| &layer.mode)
    }

    /// Returns a snapshot of all stack modes from bottom to top.
    pub fn stack_modes(&self) -> Vec<S> {
        self.stack.iter().map(|layer| layer.mode.clone()).collect()
    }

    /// Pushes a stack mode without an attached script.
    pub fn push(&mut self, mode: S) {
        self.stack.push(
            ModeLayer {
                mode,
                tasks: Vec::new(),
            },
        );
    }

    /// Pushes a stack mode and binds a private `TaskHandle` to its lifetime.
    pub(crate) fn push_with_handle(&mut self, mode: S, handle: TaskHandle) {
        self.stack.push(
            ModeLayer {
                mode,
                tasks: vec![handle],
            },
        );
    }

    /// Pops the top stack mode (preserving at least the root layer when `stack.len() > 1`),
    /// dropping its `ModeLayer` and cancelling any mode-bound scripts.
    pub fn pop(&mut self) -> Option<S> {
        if self.stack.len() <= 1 {
            return None;
        }
        self.stack.pop().map(|layer| layer.mode)
    }

    /// Evaluates whether `ClockDomain::World` should advance in the current modality.
    pub fn is_world_active(&self) -> bool {
        let modes = self.stack_modes();
        (self.world_active)(
            &self.flat, &modes,
        )
    }
}
