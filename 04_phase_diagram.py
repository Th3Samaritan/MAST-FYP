"""Fe–Fe3C reference diagram for the manuscript, drawn by the same code the tool uses (tool/tool_core.py).

Schematic of the binary metastable Fe–Fe3C system through standard tabulated invariant points
(not a CALPHAD calculation). Marks the in-house 0.40 wt.% C steel, its normalising temperature
(860 °C), the binary A3 and the Andrews (1965) Ae3 for its measured composition.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "tool"))
from tool_core import fe_c_diagram  # noqa: E402

inhouse = dict(C=0.40, Si=0.968, Mn=0.64, P=0.020, Ni=0.039, Cr=0.01, Cu=0.156, Mo=0.0, V=0.024)
fig, ae3 = fe_c_diagram(inhouse, NT=860, label="In-house steel, 0.40 wt.% C (measured composition)")
fig.savefig(HERE / "figures" / "fig_FeC_diagram_corrected.png", dpi=300)
print(f"Andrews Ae3 = {ae3:.1f} °C")
