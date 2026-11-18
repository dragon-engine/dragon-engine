use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

/// Ordered keyboard press or release transition captured on the main thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyTransition {
    pub key: KeyCode,
    pub state: ElementState,
    pub repeat: bool,
}

/// Ordered mouse button press or release transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MouseTransition {
    pub button: MouseButton,
    pub state: ElementState,
}

/// Lossless batch of input events collected on the main thread between simulation steps.
#[derive(Debug, Clone, Default)]
pub(crate) struct InputBatch {
    pub mouse_delta: glam::Vec2,
    pub cursor_pos: Option<glam::Vec2>,
    pub key_events: Vec<KeyTransition>,
    pub mouse_buttons: Vec<MouseTransition>,
}

impl InputBatch {
    pub(crate) fn is_empty(&self) -> bool {
        self.mouse_delta == glam::Vec2::ZERO
            && self.key_events.is_empty()
            && self.mouse_buttons.is_empty()
    }

    /// Converts the lossless batch into an ordered stream of discrete `GameInput` events
    /// so sequential processing against `ModeManager` respects mid-batch mode transitions.
    pub(crate) fn into_ordered_inputs(self) -> Vec<GameInput> {
        let mut out = Vec::with_capacity(self.key_events.len() + self.mouse_buttons.len() + 1);
        for key in self.key_events {
            out.push(GameInput::Key(key));
        }
        for btn in self.mouse_buttons {
            out.push(GameInput::MouseButton(btn));
        }
        if self.mouse_delta != glam::Vec2::ZERO {
            out.push(GameInput::MouseDelta(self.mouse_delta));
        }
        out
    }
}

/// Discrete input item delivered sequentially to `GameApp::on_input`.
#[derive(Debug, Clone, Copy)]
pub enum GameInput {
    Key(KeyTransition),
    MouseButton(MouseTransition),
    MouseDelta(glam::Vec2),
}
