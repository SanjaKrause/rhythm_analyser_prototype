"""
Test the reference onset search window to understand the -0.5 threshold bug.

The issue: the search window is finding onsets too far away from the target.
"""

import numpy as np

# From raster.py configuration
SEARCH_WINDOW_START_PHASE = 0.5  # Search 0.5 steps before 15/16th
SEARCH_WINDOW_END_PHASE = 0.75   # Search 0.75 steps after 1/16th
STEPS_PER_BAR = 16

# User's example calculations suggest these times:
# 108.66938775510204 - 108.831927 = -0.1625 seconds
# 108.831927 - 109.04816287499999 = -0.2162 seconds

# Let's simulate a scenario
prev_bar_start = 108.66938775510204
current_bar_start = 108.831927
next_bar_start = 109.04816287499999

prev_bar_duration = current_bar_start - prev_bar_start
current_bar_duration = next_bar_start - current_bar_start

print(f"Previous bar: {prev_bar_start:.3f} to {current_bar_start:.3f}")
print(f"Previous bar duration: {prev_bar_duration:.6f} seconds ({prev_bar_duration * 1000:.2f} ms)")
print(f"Current bar: {current_bar_start:.3f} to {next_bar_start:.3f}")
print(f"Current bar duration: {current_bar_duration:.6f} seconds ({current_bar_duration * 1000:.2f} ms)")
print()

# Calculate step durations
prev_step_duration = prev_bar_duration / STEPS_PER_BAR
current_step_duration = current_bar_duration / STEPS_PER_BAR

print(f"Previous bar step duration: {prev_step_duration:.6f} seconds ({prev_step_duration * 1000:.2f} ms)")
print(f"Current bar step duration: {current_step_duration:.6f} seconds ({current_step_duration * 1000:.2f} ms)")
print()

# Target is 1/16th (tick 0) of current bar
target_tick = 0
grid_time = current_bar_start + (target_tick * current_step_duration)

print(f"Target grid time (1/16th of current bar): {grid_time:.6f}")
print()

# Calculate 15/16th position of previous bar
prev_bar_15_16th = prev_bar_start + (15.0 / STEPS_PER_BAR) * prev_bar_duration

print(f"Previous bar 15/16th position: {prev_bar_15_16th:.6f}")
print(f"Distance from prev 15/16th to current 1/16th: {(grid_time - prev_bar_15_16th) * 1000:.2f} ms")
print()

# Calculate search window
window_start = prev_bar_15_16th - SEARCH_WINDOW_START_PHASE * prev_step_duration
window_end = grid_time + SEARCH_WINDOW_END_PHASE * current_step_duration

print("=" * 70)
print("SEARCH WINDOW CALCULATION")
print("=" * 70)
print(f"window_start = prev_bar_15_16th - {SEARCH_WINDOW_START_PHASE} * prev_step_duration")
print(f"             = {prev_bar_15_16th:.6f} - {SEARCH_WINDOW_START_PHASE} * {prev_step_duration:.6f}")
print(f"             = {prev_bar_15_16th:.6f} - {SEARCH_WINDOW_START_PHASE * prev_step_duration:.6f}")
print(f"             = {window_start:.6f}")
print()
print(f"window_end   = grid_time + {SEARCH_WINDOW_END_PHASE} * current_step_duration")
print(f"             = {grid_time:.6f} + {SEARCH_WINDOW_END_PHASE} * {current_step_duration:.6f}")
print(f"             = {grid_time:.6f} + {SEARCH_WINDOW_END_PHASE * current_step_duration:.6f}")
print(f"             = {window_end:.6f}")
print()

print(f"Window: [{window_start:.6f}, {window_end:.6f}]")
print(f"Window size: {(window_end - window_start) * 1000:.2f} ms")
print()

# How far back does the window go from the target?
distance_back = (grid_time - window_start) * 1000
print(f"Window extends {distance_back:.2f} ms BEFORE the target grid time")
print(f"This is {distance_back / (current_step_duration * 1000):.2f} steps before the target")
print()

# How far back from prev_bar_15_16th?
distance_from_15_16th = (prev_bar_15_16th - window_start) * 1000
print(f"Window starts {distance_from_15_16th:.2f} ms BEFORE the prev bar's 15/16th position")
print()

# What if we had an onset at the very start of the window?
print("=" * 70)
print("PROBLEM SCENARIO")
print("=" * 70)
print(f"If there's an onset at window_start = {window_start:.6f}:")
print(f"  - It's {(grid_time - window_start) * 1000:.2f} ms BEFORE the target (1/16th of current bar)")
print(f"  - It's {(prev_bar_15_16th - window_start) * 1000:.2f} ms BEFORE the 15/16th of prev bar")
print(f"  - It's at approximately tick {(window_start - prev_bar_start) / prev_step_duration:.1f} of the previous bar")
print()

# Calculate what tick position this would be in the previous bar
prev_bar_tick = (window_start - prev_bar_start) / prev_step_duration
print(f"The window_start corresponds to tick {prev_bar_tick:.2f} of the PREVIOUS bar")
print(f"This means we're looking as far back as tick {int(prev_bar_tick)} of the previous bar!")
