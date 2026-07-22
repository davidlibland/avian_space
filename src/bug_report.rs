//! One-key bug reporting for tester builds (cargo feature `bugreport`).
//!
//! F8 (anywhere, any state) or the "Report a problem" button in the help
//! window captures everything a report needs with zero typing: a screenshot,
//! a copy of the current pilot's save, the recent log tail (via a tracing
//! layer that mirrors log lines into a ring buffer), and a context header
//! (version, play state, system/planet/interior). Bundles land in
//! `<user data dir>/bug_reports/<timestamp>/`; an optional note window lets
//! the tester add a sentence afterwards. `scripts/sync_bug_reports.sh` turns
//! bundles into GitHub issues.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Fired by UI buttons to trigger a capture (F8 is read directly).
#[derive(Message)]
pub struct FileBugReport;

// ── Log ring buffer (fed by the LogPlugin custom layer) ───────────────────

const LOG_CAP: usize = 400;
static LOG_RING: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

/// Plug into `LogPlugin { custom_layer }`: mirrors every log event into the
/// in-memory ring so reports can include the recent log tail.
pub fn log_capture_layer(_app: &mut App) -> Option<bevy::log::BoxedLayer> {
    Some(Box::new(RingLayer))
}

struct RingLayer;

impl bevy::log::tracing_subscriber::Layer<bevy::log::tracing_subscriber::Registry> for RingLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: bevy::log::tracing_subscriber::layer::Context<
            '_,
            bevy::log::tracing_subscriber::Registry,
        >,
    ) {
        struct MsgVisitor(String);
        impl tracing::field::Visit for MsgVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    use std::fmt::Write;
                    let _ = write!(self.0, "{value:?}");
                }
            }
        }
        let mut visitor = MsgVisitor(String::new());
        event.record(&mut visitor);
        let meta = event.metadata();
        let line = format!(
            "{:5} {}: {}",
            meta.level().as_str(),
            meta.target(),
            visitor.0
        );
        if let Ok(mut ring) = LOG_RING.lock() {
            if ring.len() >= LOG_CAP {
                ring.pop_front();
            }
            ring.push_back(line);
        }
    }
}

// ── Report bundle ─────────────────────────────────────────────────────────

/// Everything the report header records, gathered by the trigger system.
pub(crate) struct ReportMeta {
    pub stamp: String,
    pub pilot: String,
    pub play_state: String,
    pub system: String,
    pub planet: String,
    pub interior: String,
}

fn bug_reports_dir() -> PathBuf {
    crate::game_save::user_data_dir().join("bug_reports")
}

/// Days-since-epoch → (year, month, day). Howard Hinnant's civil-date
/// algorithm; avoids pulling a date crate for one timestamp.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let rem = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}_{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Write `report.md` + `log.txt` into `dir` and copy the pilot save next to
/// them. Split from the system for testability.
pub(crate) fn write_report_files(dir: &std::path::Path, meta: &ReportMeta) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let report = format!(
        "# Bug report {}\n\n\
         | field | value |\n|---|---|\n\
         | version | {} |\n\
         | pilot | {} |\n\
         | state | {} |\n\
         | system | {} |\n\
         | planet | {} |\n\
         | interior | {} |\n\
         | os | {} |\n",
        meta.stamp,
        env!("CARGO_PKG_VERSION"),
        meta.pilot,
        meta.play_state,
        meta.system,
        meta.planet,
        meta.interior,
        std::env::consts::OS,
    );
    std::fs::write(dir.join("report.md"), report)?;
    let log: Vec<String> = LOG_RING
        .lock()
        .map(|r| r.iter().cloned().collect())
        .unwrap_or_default();
    std::fs::write(dir.join("log.txt"), log.join("\n"))?;
    let save = crate::game_save::pilots_dir().join(format!("{}.yaml", meta.pilot));
    if save.is_file() {
        let _ = std::fs::copy(&save, dir.join("pilot.yaml"));
    }
    Ok(())
}

// ── Systems ───────────────────────────────────────────────────────────────

/// Note-window state: which fresh report (if any) is awaiting an optional note.
#[derive(Resource, Default)]
pub struct BugReportUi {
    pending: Option<PathBuf>,
    note: String,
    /// The note field grabs keyboard focus once per capture.
    focus_given: bool,
}

impl BugReportUi {
    /// True while the tester is (potentially) typing a note — game keyboard
    /// input is suppressed and egui keeps its focus for the duration.
    pub fn is_noting(&self) -> bool {
        self.pending.is_some()
    }
}

/// While the note window is open, blank the game's view of the keyboard so
/// typing "i" can't open the mission log (egui reads its own input stream
/// and is unaffected). Runs right after input collection each frame.
fn swallow_keys_while_noting(ui: Res<BugReportUi>, mut keys: ResMut<ButtonInput<KeyCode>>) {
    if ui.is_noting() {
        keys.reset_all();
    }
}

#[allow(clippy::too_many_arguments)]
fn trigger_bug_report(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut requests: MessageReader<FileBugReport>,
    mut ui: ResMut<BugReportUi>,
    play_state: Res<State<crate::PlayState>>,
    game_state: Res<crate::game_save::PlayerGameState>,
    current_system: Res<crate::CurrentStarSystem>,
    landed: Res<crate::planet_ui::LandedContext>,
    interior: Option<Res<crate::surface::interiors::InteriorContext>>,
    mut comms: ResMut<crate::hud::CommsChannel>,
) {
    let requested = requests.read().count() > 0 || keys.just_pressed(KeyCode::F8);
    if !requested {
        return;
    }
    let meta = ReportMeta {
        stamp: timestamp(),
        pilot: game_state.pilot_name.clone(),
        play_state: format!("{:?}", play_state.get()),
        system: current_system.0.clone(),
        planet: landed.planet_name.clone().unwrap_or_else(|| "-".into()),
        interior: interior
            .map(|c| format!("{:?} level {}", c.kind, c.level))
            .unwrap_or_else(|| "-".into()),
    };
    let dir = bug_reports_dir().join(&meta.stamp);
    match write_report_files(&dir, &meta) {
        Ok(()) => {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(dir.join("screenshot.png")));
            info!("bug report captured at {dir:?}");
            comms.send("Bug report saved — thanks!");
            ui.pending = Some(dir);
            ui.note.clear();
            ui.focus_given = false;
        }
        Err(e) => {
            error!("failed to write bug report to {dir:?}: {e}");
            comms.send("Could not save bug report (see log).");
        }
    }
}

/// After a capture, offer an optional free-text note; appends to report.md.
fn note_ui(mut egui_contexts: EguiContexts, mut ui_state: ResMut<BugReportUi>) {
    if ui_state.pending.is_none() {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let mut done = false;
    egui::Window::new("Bug report saved!")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label("Got it — screenshot and game state are saved.");
            ui.label("Want to say what happened? (you can skip this)");
            let resp = ui.add(
                egui::TextEdit::multiline(&mut ui_state.note)
                    .hint_text("e.g. \"I clicked the door and the screen went black\"")
                    .desired_rows(3)
                    .desired_width(320.0),
            );
            if !ui_state.focus_given {
                resp.request_focus();
                ui_state.focus_given = true;
            }
            if ui.button("Done").clicked() {
                done = true;
            }
        });
    if done {
        if let Some(dir) = ui_state.pending.take() {
            let note = ui_state.note.trim();
            if !note.is_empty() {
                let _ = std::fs::OpenOptions::new()
                    .append(true)
                    .open(dir.join("report.md"))
                    .and_then(|mut f| {
                        use std::io::Write;
                        writeln!(f, "\n## Player note\n\n{note}")
                    });
            }
        }
        ui_state.note.clear();
    }
}

pub fn bug_report_plugin(app: &mut App) {
    app.add_message::<FileBugReport>()
        .init_resource::<BugReportUi>()
        .add_systems(
            PreUpdate,
            swallow_keys_while_noting.after(bevy::input::InputSystems),
        )
        .add_systems(Update, trigger_bug_report)
        .add_systems(EguiPrimaryContextPass, note_ui);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_date_conversion_is_correct() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // leap year start
        assert_eq!(civil_from_days(20_656), (2026, 7, 22));
        assert_eq!(civil_from_days(-1), (1969, 12, 31)); // era boundary
    }

    #[test]
    fn report_bundle_contains_header_and_log() {
        let dir = std::env::temp_dir().join(format!("bugreport_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let meta = ReportMeta {
            stamp: "2026-07-22_120000".into(),
            pilot: "Test Pilot".into(),
            play_state: "Exploring".into(),
            system: "sol".into(),
            planet: "mars".into(),
            interior: "-".into(),
        };
        write_report_files(&dir, &meta).unwrap();
        let report = std::fs::read_to_string(dir.join("report.md")).unwrap();
        assert!(report.contains("# Bug report 2026-07-22_120000"));
        assert!(report.contains("| pilot | Test Pilot |"));
        assert!(report.contains("| system | sol |"));
        assert!(dir.join("log.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
