"""
Test script to understand the merge bug.

The issue: when grid correction shifts boundaries, the same onset
can be assigned to different (bar, tick) positions in different methods.
"""

import pandas as pd
import numpy as np

# Simulate the three methods assigning the SAME onset to different positions

# Method 1: Uncorrected - assigns onset to bar 1, tick 15
df_uncorrected = pd.DataFrame([
    {'bar_number': 1, 'tick_16th': 15, 'onset_time': 118.537, 'phase_uncorrected': 0.968}
])

# Method 2: Per-snippet - assigns the SAME onset to bar 2, tick 0 (due to correction)
df_per_snippet = pd.DataFrame([
    {'bar_number': 2, 'tick_16th': 0, 'onset_time': 118.537, 'phase_per_snippet': 0.0106}
])

# Create comprehensive grid (all bar/tick combinations)
grid_rows = []
for bar in range(4):
    for tick in range(16):
        grid_rows.append({'bar_number': bar, 'tick_16th': tick})

df_comprehensive = pd.DataFrame(grid_rows)

print("Comprehensive grid (first 20 rows):")
print(df_comprehensive.head(20))
print()

# Merge uncorrected
print("After merging uncorrected:")
df_comprehensive = df_comprehensive.merge(
    df_uncorrected[['bar_number', 'tick_16th', 'onset_time', 'phase_uncorrected']],
    on=['bar_number', 'tick_16th'],
    how='left'
)
print(df_comprehensive[(df_comprehensive['bar_number'].isin([1, 2])) &
                        (df_comprehensive['tick_16th'].isin([0, 15]))].to_string())
print()

# Merge per-snippet
print("After merging per-snippet:")
df_comprehensive = df_comprehensive.merge(
    df_per_snippet[['bar_number', 'tick_16th', 'phase_per_snippet']],
    on=['bar_number', 'tick_16th'],
    how='left'
)
print(df_comprehensive[(df_comprehensive['bar_number'].isin([1, 2])) &
                        (df_comprehensive['tick_16th'].isin([0, 15]))].to_string())
print()

print("=" * 80)
print("PROBLEM IDENTIFIED:")
print("=" * 80)
print("Row (1, 15): has onset_time + phase_uncorrected, but NO phase_per_snippet")
print("Row (2, 0):  has NO onset_time, but HAS phase_per_snippet")
print()
print("This happens because the same onset was assigned to DIFFERENT grid positions")
print("after correction! The onset at 118.537 is:")
print("  - At bar 1, tick 15 in uncorrected method (phase = 0.968)")
print("  - At bar 2, tick 0 in corrected method (phase = 0.0106)")
print()
print("This is the bug you're seeing!")
