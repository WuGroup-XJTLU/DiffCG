"""Common physical constants and unit conversions for diffcg.

All constants are expressed in kJ, mol, K, nm, and bar where applicable to
match the rest of the codebase conventions.
"""

# Boltzmann constant in kJ / (mol * K)
BOLTZMANN_KJMOLK: float = 0.0083145107

# Pressure conversion: kJ / (mol * nm^3) to bar
# 1 kJ mol^-1 nm^-3 = 16.6054 bar
PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR: float = 16.6054


