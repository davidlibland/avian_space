#!/usr/bin/env bash
# Build a standalone macOS .app bundle for inference-only play.
#
# Assets are baked into the binary via the `bundle` cargo feature
# (bevy_embedded_assets + include_dir), so the .app is fully self-contained
# — no Resources/assets folder ships alongside it.
#
# When launched from Finder the binary detects it is inside a .app (no CLI
# args passed) and defaults to `--inference` mode.
#
# Output: target/macos/AvianSpace.app
#
# Usage:
#   scripts/build_macos_app.sh                # release build
#   SKIP_BUILD=1 scripts/build_macos_app.sh   # repackage existing binary
#
# Notes:
#   * Built without the `dev` feature on purpose — `dev` enables Bevy
#     dynamic_linking, which doesn't relocate cleanly into a bundle.
#   * Drop an icon at scripts/AppIcon.icns to embed it; otherwise Finder
#     will show the generic app icon.

set -euo pipefail

cd "$(dirname "$0")/.."

BUNDLE_NAME="AvianSpace.app"
BUNDLE_DIR="target/macos/${BUNDLE_NAME}"
BIN_NAME="avian_space"
DISPLAY_NAME="Avian Space"
BUNDLE_ID="com.dlibland.avianspace"
VERSION="$(grep '^version' Cargo.toml | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "[1/5] cargo build --release --features bundle"
  cargo build --release --features bundle
fi

BIN_PATH="target/release/${BIN_NAME}"
if [[ ! -x "${BIN_PATH}" ]]; then
  echo "error: release binary not found at ${BIN_PATH}" >&2
  exit 1
fi

echo "[2/5] assembling ${BUNDLE_DIR}"
rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}/Contents/MacOS"
mkdir -p "${BUNDLE_DIR}/Contents/Resources"

cp "${BIN_PATH}" "${BUNDLE_DIR}/Contents/MacOS/${BIN_NAME}"
chmod +x "${BUNDLE_DIR}/Contents/MacOS/${BIN_NAME}"

echo "[3/5] (assets are embedded in the binary — nothing to copy)"

ICON_KEY=""
if [[ -f scripts/AppIcon.icns ]]; then
  cp scripts/AppIcon.icns "${BUNDLE_DIR}/Contents/Resources/AppIcon.icns"
  ICON_KEY="<key>CFBundleIconFile</key><string>AppIcon</string>"
fi

echo "[4/5] writing Info.plist"
cat > "${BUNDLE_DIR}/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>${DISPLAY_NAME}</string>
  <key>CFBundleDisplayName</key><string>${DISPLAY_NAME}</string>
  <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleExecutable</key><string>${BIN_NAME}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  ${ICON_KEY}
</dict>
</plist>
PLIST

echo "[5/5] ad-hoc signing the bundle"
# Ad-hoc signing (identity "-") doesn't satisfy notarization, but it stamps a
# consistent self-signature across the whole bundle. Without it, recipients
# often see Gatekeeper's "is damaged and can't be opened" error after the
# quarantine attribute is applied during transfer; with it, the failure mode
# becomes the recoverable "unidentified developer" path.
codesign --force --deep --sign - "${BUNDLE_DIR}"
codesign --verify --deep --strict "${BUNDLE_DIR}"

echo
echo "built ${BUNDLE_DIR}"
echo "run with: open '${BUNDLE_DIR}'"
echo
echo "recipients on another Mac may still need to strip the quarantine attr:"
echo "  xattr -dr com.apple.quarantine /path/to/${BUNDLE_NAME}"
