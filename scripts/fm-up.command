#!/bin/zsh
# Double-click entry point, or add it as a Login Item to start at login: macOS opens .command
# files in Terminal.app, which `fm serve` needs for Private Cloud Compute. Keep it next to fm-up.
exec "${0:A:h}/fm-up"
