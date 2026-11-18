use engine::{ActionSender, RenderSchedule, RenderSnapshot, UiBlitter, UiOverlay, UiRenderTarget};
use iced_core::{
    Alignment, Border, Color, Element, Length, Padding, Pixels, Size, Theme, clipboard, mouse,
};
use iced_graphics::Viewport;
use iced_runtime::user_interface::{self, UserInterface};
use iced_tiny_skia::Renderer as IcedRenderer;
use iced_widget::{Space, button, column, container, mouse_area, row, stack, text};
use iced_winit::conversion;
use tiny_skia::{Mask, Pixmap};
use winit::event::{ElementState, Modifiers, WindowEvent};
use winit::window::Window;

use crate::app::{FlatMode, RpgAction, RpgSnapshot, StackMode};

#[derive(Debug, Clone)]
pub(crate) enum UiMessage {
    Noop,
    Attack,
    UsePotion,
    OpenInventory,
    CloseModal,
    PromptDiscardPotion,
    ConfirmDiscardPotion,
    ToggleMenu,
    SetSchedule(RenderSchedule),
    Save,
    Load,
    Quit,
}

struct UiViewState {
    snapshot: RpgSnapshot,
    world_time_ms: u64,
    ui_time_ms: u64,
    world_speed: f32,
    schedule: RenderSchedule,
}

impl Default for UiViewState {
    fn default() -> Self {
        Self {
            snapshot: RpgSnapshot::default(),
            world_time_ms: 0,
            ui_time_ms: 0,
            world_speed: 1.0,
            schedule: RenderSchedule::Sequential,
        }
    }
}

pub(crate) struct RpgUi {
    renderer: IcedRenderer,
    cache: Option<user_interface::Cache>,
    events: Vec<iced_core::Event>,
    cursor: mouse::Cursor,
    modifiers: Modifiers,
    view_state: UiViewState,
    pixmap: Option<Pixmap>,
    clip_mask: Option<Mask>,
    blitter: Option<UiBlitter>,
    last_physical_size: (
        u32,
        u32,
    ),
    last_scale: f32,
    cursor_grabbed: bool,
}

impl RpgUi {
    pub(crate) fn new() -> Self {
        Self {
            renderer: IcedRenderer::new(
                iced_core::Font::DEFAULT,
                Pixels(15.0),
            ),
            cache: Some(user_interface::Cache::new()),
            events: Vec::new(),
            cursor: mouse::Cursor::Unavailable,
            modifiers: Modifiers::default(),
            view_state: UiViewState::default(),
            pixmap: None,
            clip_mask: None,
            blitter: None,
            last_physical_size: (
                1280, 720,
            ),
            last_scale: 1.0,
            cursor_grabbed: false,
        }
    }
}

fn build_view(state: &UiViewState) -> Element<'_, UiMessage, Theme, IcedRenderer> {
    let top_hud = container(
        row![
            column![
                text(
                    format!(
                        "Knight HP: {}/{}",
                        state.snapshot.player_hp, state.snapshot.player_max_hp
                    )
                )
                .size(16),
                text(
                    format!(
                        "Potions: {} | Regen: {} ms | Goblin: {} | Tree: {}",
                        state.snapshot.player_potions,
                        state.snapshot.player_regen_ms,
                        state.snapshot.goblin_hp.map_or_else(
                            || String::from("Dead"),
                            |hp| hp.to_string()
                        ),
                        state.snapshot.tree_hp.map_or_else(
                            || String::from("Purified"),
                            |hp| hp.to_string()
                        ),
                    )
                )
                .size(13),
                text(
                    format!(
                        "Quest: {}",
                        state.snapshot.quest_stage.label()
                    )
                )
                .size(14),
            ]
            .spacing(4),
            Space::new().width(Length::Fill),
            column![
                text(
                    format!(
                        "World: {} ms (x{:.1}) | UI: {} ms",
                        state.world_time_ms, state.world_speed, state.ui_time_ms
                    )
                )
                .size(13),
                text(
                    format!(
                        "Mode: {:?} | Stack: {:?}",
                        state.snapshot.flat_mode, state.snapshot.stack_modes
                    )
                )
                .size(13),
                text(
                    format!(
                        "Status: {}",
                        state.snapshot.status_banner
                    )
                )
                .size(13),
            ]
            .align_x(Alignment::End)
            .spacing(4),
        ]
        .padding(10),
    )
    .width(Length::Fill)
    .style(
        |_theme: &Theme| container::Style {
            background: Some(
                Color::from_rgba(
                    0.06, 0.08, 0.12, 0.78,
                )
                .into(),
            ),
            text_color: Some(Color::WHITE),
            ..Default::default()
        },
    );

    let sched_label = match state.schedule {
        RenderSchedule::Sequential => "Schedule: Low Latency (Seq)",
        RenderSchedule::Parallel => "Schedule: High Frame Rate (Par)",
    };
    let next_sched = match state.schedule {
        RenderSchedule::Sequential => RenderSchedule::Parallel,
        RenderSchedule::Parallel => RenderSchedule::Sequential,
    };

    let bottom_bar = container(
        row![
            button(text("Attack [Space]").size(13)).on_press(UiMessage::Attack),
            button(text("Use Potion [H]").size(13)).on_press(UiMessage::UsePotion),
            button(text("Inventory [I]").size(13)).on_press(UiMessage::OpenInventory),
            button(text("Menu [Esc]").size(13)).on_press(UiMessage::ToggleMenu),
            button(text(sched_label).size(13)).on_press(UiMessage::SetSchedule(next_sched)),
            button(text("Save [F5]").size(13)).on_press(UiMessage::Save),
            button(text("Load [F9]").size(13)).on_press(UiMessage::Load),
        ]
        .spacing(8)
        .padding(10),
    )
    .width(Length::Fill)
    .style(
        |_theme: &Theme| container::Style {
            background: Some(
                Color::from_rgba(
                    0.06, 0.08, 0.12, 0.80,
                )
                .into(),
            ),
            ..Default::default()
        },
    );

    let mut layers: Vec<Element<'_, UiMessage, Theme, IcedRenderer>> = vec![
        column![top_hud, Space::new().height(Length::Fill), bottom_bar]
            .width(Length::Fill)
            .height(Length::Fill)
            .into(),
    ];

    if state.snapshot.stack_modes.last() == Some(&StackMode::Inventory) {
        let panel = modal_card(
            column![
                text("Inventory (World Paused)").size(20),
                text(
                    format!(
                        "Health Potions: {}",
                        state.snapshot.player_potions
                    )
                )
                .size(15),
                text("Using a potion heals AoS HP now and queues regen on World clock.").size(13),
                row![
                    button("Drink Potion").on_press(UiMessage::UsePotion),
                    button("Discard 1 Potion").on_press(UiMessage::PromptDiscardPotion),
                    button("Close").on_press(UiMessage::CloseModal),
                ]
                .spacing(10),
            ]
            .spacing(12),
        );
        layers.push(modal_backdrop(panel));
    }

    if state.snapshot.stack_modes.last() == Some(&StackMode::ConfirmDiscard) {
        let panel = modal_card(
            column![
                text("Confirm Discard").size(20),
                text("Discard 1 Health Potion? (Auto-cancels in 6s UI time)").size(14),
                row![
                    button("Confirm Discard").on_press(UiMessage::ConfirmDiscardPotion),
                    button("Cancel").on_press(UiMessage::CloseModal),
                ]
                .spacing(10),
            ]
            .spacing(12),
        );
        layers.push(modal_backdrop(panel));
    }

    if state.snapshot.flat_mode == FlatMode::MainMenu {
        let panel = modal_card(
            column![
                text("Main Menu (Stack Preserved)").size(22),
                text(
                    format!(
                        "Underlying Stack: {:?}",
                        state.snapshot.stack_modes
                    )
                )
                .size(14),
                row![
                    button("Resume Game [Esc]").on_press(UiMessage::ToggleMenu),
                    button("Save Checkpoint").on_press(UiMessage::Save),
                    button("Load Checkpoint").on_press(UiMessage::Load),
                    button("Quit").on_press(UiMessage::Quit),
                ]
                .spacing(10),
            ]
            .spacing(12),
        );
        layers.push(modal_backdrop(panel));
    }

    stack(layers)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn modal_card<'a>(
    content: impl Into<Element<'a, UiMessage, Theme, IcedRenderer>>,
) -> Element<'a, UiMessage, Theme, IcedRenderer> {
    container(content)
        .padding(Padding::from(20))
        .style(
            |_theme: &Theme| container::Style {
                background: Some(
                    Color::from_rgb(
                        0.12, 0.15, 0.22,
                    )
                    .into(),
                ),
                text_color: Some(Color::WHITE),
                border: Border {
                    color: Color::from_rgb(
                        0.45, 0.55, 0.72,
                    ),
                    width: 2.0,
                    radius: 6.0.into(),
                },
                ..Default::default()
            },
        )
        .into()
}

fn modal_backdrop<'a>(
    card: Element<'a, UiMessage, Theme, IcedRenderer>,
) -> Element<'a, UiMessage, Theme, IcedRenderer> {
    let overlay = container(card)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .style(
            |_theme: &Theme| container::Style {
                background: Some(
                    Color::from_rgba(
                        0.0, 0.0, 0.0, 0.55,
                    )
                    .into(),
                ),
                ..Default::default()
            },
        );
    mouse_area(overlay).on_press(UiMessage::Noop).into()
}

impl UiOverlay<RpgSnapshot, RpgAction> for RpgUi {
    fn handle_event(&mut self, event: &WindowEvent, window: &Window) -> bool {
        let scale = window.scale_factor() as f32;
        self.last_scale = scale;
        let inner = window.inner_size();
        if inner.width > 0 && inner.height > 0 {
            self.last_physical_size = (
                inner.width,
                inner.height,
            );
        }

        if let WindowEvent::ModifiersChanged(new_mods) = event {
            self.modifiers = *new_mods;
        }
        if self.cursor_grabbed {
            self.cursor = mouse::Cursor::Unavailable;
        } else if let WindowEvent::CursorMoved { position, .. } = event {
            self.cursor = mouse::Cursor::Available(
                conversion::cursor_position(
                    *position, scale,
                ),
            );
        } else if let WindowEvent::CursorLeft { .. } = event {
            self.cursor = mouse::Cursor::Unavailable;
        }

        if let Some(iced_ev) = conversion::window_event(
            event.clone(),
            scale,
            self.modifiers.state(),
        ) {
            self.events.push(iced_ev);
        }

        // Capture pointer clicks only when a modal layer is active or pointer hits bottom bar / top HUD.
        let modal_open = self.view_state.snapshot.flat_mode == FlatMode::MainMenu
            || self.view_state.snapshot.stack_modes.last() != Some(&StackMode::Play);

        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                if modal_open {
                    return true;
                }
                if self.cursor_grabbed {
                    return false;
                }
                if let Some(pos) = self.cursor.position() {
                    let logical_h = inner.height as f32 / scale.max(0.1);
                    return pos.y > logical_h - 60.0 || pos.y < 85.0;
                }
                false
            }
            _ => false,
        }
    }

    fn sync(&mut self, snapshot: &RenderSnapshot<RpgSnapshot>, actions: &ActionSender<RpgAction>) {
        self.cursor_grabbed = snapshot.cursor_grab;
        if self.cursor_grabbed {
            self.cursor = mouse::Cursor::Unavailable;
        }
        self.view_state.snapshot = snapshot.game.clone();
        self.view_state.world_time_ms = snapshot.world_time_ms;
        self.view_state.ui_time_ms = snapshot.ui_time_ms;
        self.view_state.world_speed = snapshot.world_speed;
        self.view_state.schedule = snapshot.schedule;

        let logical_size = Size::new(
            self.last_physical_size.0 as f32 / self.last_scale.max(0.1),
            self.last_physical_size.1 as f32 / self.last_scale.max(0.1),
        );

        let cache = self.cache.take().unwrap_or_default();
        let mut messages = Vec::new();
        let mut ui = UserInterface::build(
            build_view(&self.view_state),
            logical_size,
            cache,
            &mut self.renderer,
        );
        let events = std::mem::take(&mut self.events);
        let _ = ui.update(
            &events,
            self.cursor,
            &mut self.renderer,
            &mut clipboard::Null,
            &mut messages,
        );
        self.cache = Some(ui.into_cache());

        for msg in messages {
            let mapped = match msg {
                UiMessage::Noop => None,
                UiMessage::Attack => Some(RpgAction::Attack),
                UiMessage::UsePotion => Some(RpgAction::UsePotion),
                UiMessage::OpenInventory => Some(RpgAction::ToggleInventory),
                UiMessage::CloseModal => Some(RpgAction::CancelModal),
                UiMessage::PromptDiscardPotion => Some(RpgAction::DiscardPotionPrompt),
                UiMessage::ConfirmDiscardPotion => Some(RpgAction::ConfirmDiscardPotion),
                UiMessage::ToggleMenu => Some(RpgAction::ToggleMainMenu),
                UiMessage::SetSchedule(s) => Some(RpgAction::SetSchedule(s)),
                UiMessage::Save => Some(RpgAction::SaveCheckpoint),
                UiMessage::Load => Some(RpgAction::LoadCheckpoint),
                UiMessage::Quit => Some(RpgAction::Quit),
            };
            if let Some(action) = mapped
                && let Err(err) = actions.try_send(action)
            {
                tracing::warn!(
                    ?err,
                    "Failed to enqueue UI action"
                );
            }
        }
    }

    fn render(&mut self, target: &mut UiRenderTarget<'_>, _ui_time_ms: u64) {
        let width = target.width.max(1);
        let height = target.height.max(1);
        self.last_physical_size = (
            width, height,
        );
        self.last_scale = target.scale_factor;

        let viewport = Viewport::with_physical_size(
            Size::new(
                width, height,
            ),
            target.scale_factor,
        );

        let cache = self.cache.take().unwrap_or_default();
        let mut ui = UserInterface::build(
            build_view(&self.view_state),
            viewport.logical_size(),
            cache,
            &mut self.renderer,
        );
        ui.draw(
            &mut self.renderer,
            &Theme::Dark,
            &iced_core::renderer::Style::default(),
            self.cursor,
        );
        self.cache = Some(ui.into_cache());

        let needs_alloc = self
            .pixmap
            .as_ref()
            .is_none_or(|p| p.width() != width || p.height() != height);
        if needs_alloc {
            self.pixmap = Pixmap::new(
                width, height,
            );
            self.clip_mask = Mask::new(
                width, height,
            );
        }

        if let (Some(pixmap), Some(mask)) = (
            self.pixmap.as_mut(),
            self.clip_mask.as_mut(),
        ) {
            pixmap.fill(tiny_skia::Color::TRANSPARENT);
            let mut pixmap_mut = pixmap.as_mut();
            let damage = [iced_core::Rectangle::with_size(viewport.logical_size())];
            self.renderer.draw(
                &mut pixmap_mut,
                mask,
                &viewport,
                &damage,
                Color::TRANSPARENT,
            );

            let blitter = self.blitter.get_or_insert_with(|| UiBlitter::new(target));
            blitter.composite_rgba8(
                target,
                pixmap.data(),
            );
        }
    }
}
