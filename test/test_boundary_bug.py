"""
Test script to debug the boundary assignment bug.

This reproduces the issue where onset 118.537... is assigned to both:
- bar 1, tick 15
- bar 2, tick 0
"""

import numpy as np

# Configuration (from raster.py)
MAX_MATCH_FRAC_BEFORE = 0.49
MAX_MATCH_FRAC_AFTER = 0.51
STEPS_PER_BAR = 16

# Test data from the bug report
onset_time = 118.53786848072562

# Simulated bar boundaries (we need to infer these from the grid times)
# From the CSV: bar 1 grid_time = 118.4711108125, bar 2 grid_time = 118.607528
# These look like uncorrected grid times for tick 0

# Let's assume 4/4 time signature with ~2 second bars
# Bar 1 starts around 118.4711108125 - correction
# Bar 2 starts around 118.607528 - correction

# From the phase values, we can estimate:
# phase_uncorrected for bar 1, tick 15: 0.9680852535194744
# This means: (onset - bar_start) / bar_duration = 0.968...
# So: onset - bar_start = 0.968 * bar_duration

# Let's work backwards from the corrected grid times
# Uncorrected grid time bar 1 tick 0: 118.4711108125
# Uncorrected grid time bar 2 tick 0: 118.607528
# So bar 1 duration = 118.607528 - 118.4711108125 = 0.136417...

bar_1_start_uncorrected = 118.4711108125
bar_2_start_uncorrected = 118.607528
bar_1_duration = bar_2_start_uncorrected - bar_1_start_uncorrected

print(f"Bar 1 start (uncorrected): {bar_1_start_uncorrected}")
print(f"Bar 2 start (uncorrected): {bar_2_start_uncorrected}")
print(f"Bar 1 duration: {bar_1_duration}")
print(f"Bar 1 end: {bar_1_start_uncorrected + bar_1_duration}")
print()

# Now let's estimate the correction offset
# phase_per_snippet = 0.9361704083437676 (from bar 1, tick 15)
# phase_uncorrected = 0.9680852535194744
# The correction shifted the phase, so we can calculate the offset

# From grid_time_per_snippet = 118.54077054719389 (bar 1, tick 0)
# From grid_time_uncorrected = 118.4711108125 (bar 1, tick 0)
ref_offset_s = 118.54077054719389 - 118.4711108125

print(f"Reference offset: {ref_offset_s} seconds ({ref_offset_s * 1000} ms)")
print()

# Now simulate the assignment logic
step_duration = bar_1_duration / STEPS_PER_BAR

print("=" * 60)
print("BAR 1 PROCESSING")
print("=" * 60)

# Bar 1: Filter using uncorrected boundaries
bar_1_onsets = [onset_time] if bar_1_start_uncorrected <= onset_time < bar_2_start_uncorrected else []
print(f"Bar 1 filtering: {bar_1_start_uncorrected} <= {onset_time} < {bar_2_start_uncorrected}")
print(f"Onset in bar 1: {len(bar_1_onsets) > 0}")

if len(bar_1_onsets) > 0:
    # Calculate corrected phase
    corrected_bar_1_start = bar_1_start_uncorrected + ref_offset_s
    phase = (onset_time - corrected_bar_1_start) / bar_1_duration
    print(f"Corrected bar 1 start: {corrected_bar_1_start}")
    print(f"Phase: {phase}")

    # Assign to nearest tick
    nearest_tick = int(round(phase * STEPS_PER_BAR))
    nearest_tick = max(0, min(STEPS_PER_BAR - 1, nearest_tick))
    print(f"Nearest tick (before clamp): {round(phase * STEPS_PER_BAR)}")
    print(f"Nearest tick (after clamp): {nearest_tick}")

    # Check tolerance
    grid_time = corrected_bar_1_start + (nearest_tick / STEPS_PER_BAR) * bar_1_duration
    distance = abs(onset_time - grid_time)

    if onset_time < grid_time:
        tolerance = MAX_MATCH_FRAC_BEFORE * step_duration
    else:
        tolerance = MAX_MATCH_FRAC_AFTER * step_duration

    print(f"Grid time for tick {nearest_tick}: {grid_time}")
    print(f"Distance: {distance}")
    print(f"Tolerance: {tolerance}")
    print(f"Within tolerance: {distance <= tolerance}")

    if distance <= tolerance:
        print(f"✓ ASSIGNED to bar 1, tick {nearest_tick}")
    else:
        print(f"✗ NOT ASSIGNED (outside tolerance)")

print()
print("=" * 60)
print("BAR 2 PROCESSING")
print("=" * 60)

# Bar 2: Assume same duration for simplicity
bar_2_duration = bar_1_duration
bar_2_end_uncorrected = bar_2_start_uncorrected + bar_2_duration

# Filter using uncorrected boundaries
bar_2_onsets = [onset_time] if bar_2_start_uncorrected <= onset_time < bar_2_end_uncorrected else []
print(f"Bar 2 filtering: {bar_2_start_uncorrected} <= {onset_time} < {bar_2_end_uncorrected}")
print(f"Onset in bar 2: {len(bar_2_onsets) > 0}")

if len(bar_2_onsets) > 0:
    # Calculate corrected phase
    corrected_bar_2_start = bar_2_start_uncorrected + ref_offset_s
    phase = (onset_time - corrected_bar_2_start) / bar_2_duration
    print(f"Corrected bar 2 start: {corrected_bar_2_start}")
    print(f"Phase: {phase}")

    # Assign to nearest tick
    nearest_tick = int(round(phase * STEPS_PER_BAR))
    nearest_tick = max(0, min(STEPS_PER_BAR - 1, nearest_tick))
    print(f"Nearest tick (before clamp): {round(phase * STEPS_PER_BAR)}")
    print(f"Nearest tick (after clamp): {nearest_tick}")

    # Check tolerance
    grid_time = corrected_bar_2_start + (nearest_tick / STEPS_PER_BAR) * bar_2_duration
    distance = abs(onset_time - grid_time)

    if onset_time < grid_time:
        tolerance = MAX_MATCH_FRAC_BEFORE * step_duration
    else:
        tolerance = MAX_MATCH_FRAC_AFTER * step_duration

    print(f"Grid time for tick {nearest_tick}: {grid_time}")
    print(f"Distance: {distance}")
    print(f"Tolerance: {tolerance}")
    print(f"Within tolerance: {distance <= tolerance}")

    if distance <= tolerance:
        print(f"✓ ASSIGNED to bar 2, tick {nearest_tick}")
    else:
        print(f"✗ NOT ASSIGNED (outside tolerance)")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("If the onset is assigned to BOTH bars, that's the bug!")
