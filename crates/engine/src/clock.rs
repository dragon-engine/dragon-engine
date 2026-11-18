use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::task::Waker;

/// Identifies which simulation clock domain governs a timer or script.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockDomain {
    /// Pauses when gameplay is paused (for example, inside menus or save checkpoints).
    World,
    /// Advances continuously at 1.0x speed across all modes.
    Ui,
}

#[derive(Debug)]
struct TimerEntry {
    deadline_ms: u64,
    id: u64,
    waker: Waker,
}

impl PartialEq for TimerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline_ms == other.deadline_ms && self.id == other.id
    }
}

impl Eq for TimerEntry {}

impl PartialOrd for TimerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimerEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.deadline_ms
            .cmp(&other.deadline_ms)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Manages independent `World` and `Ui` clock domains and their timer heaps.
pub(crate) struct DualClocks {
    world_time_ms: u64,
    ui_time_ms: u64,
    world_speed: f32,
    world_paused_for_checkpoint: bool,
    next_timer_id: u64,
    world_timers: BinaryHeap<Reverse<TimerEntry>>,
    ui_timers: BinaryHeap<Reverse<TimerEntry>>,
}

impl DualClocks {
    pub(crate) fn new() -> Self {
        Self {
            world_time_ms: 0,
            ui_time_ms: 0,
            world_speed: 0.0,
            world_paused_for_checkpoint: false,
            next_timer_id: 1,
            world_timers: BinaryHeap::new(),
            ui_timers: BinaryHeap::new(),
        }
    }

    pub(crate) fn world_time_ms(&self) -> u64 {
        self.world_time_ms
    }

    pub(crate) fn ui_time_ms(&self) -> u64 {
        self.ui_time_ms
    }

    pub(crate) fn set_world_time_ms(&mut self, time_ms: u64) {
        self.world_time_ms = time_ms;
    }

    pub(crate) fn set_world_speed(&mut self, speed: f32) {
        self.world_speed = speed.max(0.0);
    }

    pub(crate) fn world_speed(&self) -> f32 {
        if self.world_paused_for_checkpoint {
            0.0
        } else {
            self.world_speed
        }
    }

    pub(crate) fn pause_world_for_checkpoint(&mut self) {
        self.world_paused_for_checkpoint = true;
    }

    pub(crate) fn resume_world_from_checkpoint(&mut self) {
        self.world_paused_for_checkpoint = false;
    }

    /// Advances `Ui` at 1.0x and `World` by `world_speed`, returning elapsed world seconds.
    pub(crate) fn advance(&mut self, real_dt_ms: u64) -> f32 {
        self.ui_time_ms = self.ui_time_ms.saturating_add(real_dt_ms);
        Self::wake_expired(
            self.ui_time_ms,
            &mut self.ui_timers,
        );

        let effective_speed = self.world_speed();
        let world_dt_ms = if effective_speed > 0.0 {
            ((real_dt_ms as f64) * (effective_speed as f64)).round() as u64
        } else {
            0
        };

        if world_dt_ms > 0 {
            self.world_time_ms = self.world_time_ms.saturating_add(world_dt_ms);
            Self::wake_expired(
                self.world_time_ms,
                &mut self.world_timers,
            );
        }

        (world_dt_ms as f32) / 1000.0
    }

    pub(crate) fn domain_time_ms(&self, domain: ClockDomain) -> u64 {
        match domain {
            ClockDomain::World => self.world_time_ms,
            ClockDomain::Ui => self.ui_time_ms,
        }
    }

    pub(crate) fn register_timer(&mut self, domain: ClockDomain, deadline_ms: u64, waker: Waker) {
        let id = self.next_timer_id;
        self.next_timer_id = self.next_timer_id.wrapping_add(1);
        let entry = Reverse(
            TimerEntry {
                deadline_ms,
                id,
                waker,
            },
        );
        match domain {
            ClockDomain::World => self.world_timers.push(entry),
            ClockDomain::Ui => self.ui_timers.push(entry),
        }
    }

    fn wake_expired(now_ms: u64, heap: &mut BinaryHeap<Reverse<TimerEntry>>) {
        while let Some(Reverse(top)) = heap.peek() {
            if top.deadline_ms <= now_ms {
                let Reverse(expired) = heap.pop().expect("peeked timer must exist");
                expired.waker.wake();
            } else {
                break;
            }
        }
    }
}
