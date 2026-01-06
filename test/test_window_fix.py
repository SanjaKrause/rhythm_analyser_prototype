"""
Test different window configurations to find onsets near bar boundaries.

The goal: find onsets that are truly at the 1/16th position (bar boundary),
not onsets from earlier in the previous bar.
"""

import numpy as np

STEPS_PER_BAR = 16

# User's example
prev_bar_start = 108.66938775510204
current_bar_start = 108.831927
next_bar_start = 109.04816287499999

prev_bar_duration = current_bar_start - prev_bar_start
current_bar_duration = next_bar_start - current_bar_start

prev_step_duration = prev_bar_duration / STEPS_PER_BAR
current_step_duration = current_bar_duration / STEPS_PER_BAR

# Target is 1/16th (tick 0) of current bar (which is the downbeat)
grid_time = current_bar_start
prev_bar_15_16th = prev_bar_start + (15.0 / STEPS_PER_BAR) * prev_bar_duration

print(f"Bar durations: prev={prev_bar_duration * 1000:.2f}ms, current={current_bar_duration * 1000:.2f}ms")
print(f"Step durations: prev={prev_step_duration * 1000:.2f}ms, current={current_step_duration * 1000:.2f}ms")
print(f"Target (1/16th): {grid_time:.6f}")
print(f"Prev 15/16th: {prev_bar_15_16th:.6f}")
print(f"Distance between: {(grid_time - prev_bar_15_16th) * 1000:.2f}ms")
print()

print("=" * 70)
print("CURRENT IMPLEMENTATION (PROBLEMATIC)")
print("=" * 70)

# Current implementation
SEARCH_WINDOW_START_PHASE = 0.5
SEARCH_WINDOW_END_PHASE = 0.75

window_start_old = prev_bar_15_16th - SEARCH_WINDOW_START_PHASE * prev_step_duration
window_end_old = grid_time + SEARCH_WINDOW_END_PHASE * current_step_duration

print(f"Window start: {window_start_old:.6f}")
print(f"  = prev_bar_15_16th - 0.5 * prev_step")
print(f"  = {prev_bar_15_16th:.6f} - {0.5 * prev_step_duration:.6f}")
print(f"  = tick {(window_start_old - prev_bar_start) / prev_step_duration:.2f} of prev bar")
print()
print(f"Window end: {window_end_old:.6f}")
print(f"  = grid_time + 0.75 * current_step")
print(f"  = tick {(window_end_old - current_bar_start) / current_step_duration:.2f} of current bar")
print()
print(f"Total window: {(window_end_old - window_start_old) * 1000:.2f}ms")
print(f"Distance before target: {(grid_time - window_start_old) * 1000:.2f}ms ({(grid_time - window_start_old) / current_step_duration:.2f} steps)")
print()

print("=" * 70)
print("PROPOSED FIX: Symmetric window around boundary")
print("=" * 70)

# Proposed: search from 0.5 steps before the target to 0.75 steps after
# But use the CURRENT bar's step duration for consistency
window_start_new = grid_time - 0.5 * current_step_duration
window_end_new = grid_time + 0.75 * current_step_duration

print(f"Window start: {window_start_new:.6f}")
print(f"  = grid_time - 0.5 * current_step")
print(f"  = {grid_time:.6f} - {0.5 * current_step_duration:.6f}")
print(f"  = tick {(window_start_new - current_bar_start) / current_step_duration:.2f} of current bar")
print(f"  = tick {(window_start_new - prev_bar_start) / prev_step_duration:.2f} of prev bar")
print()
print(f"Window end: {window_end_new:.6f}")
print(f"  = grid_time + 0.75 * current_step")
print(f"  = tick {(window_end_new - current_bar_start) / current_step_duration:.2f} of current bar")
print()
print(f"Total window: {(window_end_new - window_start_new) * 1000:.2f}ms")
print(f"Distance before target: {(grid_time - window_start_new) * 1000:.2f}ms ({(grid_time - window_start_new) / current_step_duration:.2f} steps)")
print()

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Old window looks back: {(grid_time - window_start_old) * 1000:.2f}ms ({(grid_time - window_start_old) / current_step_duration:.2f} steps)")
print(f"New window looks back: {(grid_time - window_start_new) * 1000:.2f}ms ({(grid_time - window_start_new) / current_step_duration:.2f} steps)")
print()
print("The new window is more symmetric and doesn't reach back into")
print("earlier ticks of the previous bar.")
