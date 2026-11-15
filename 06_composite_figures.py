"""Assemble multi-panel manuscript figures from the individual PNGs."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FIG = Path(__file__).parent / "figures"
try:
    FONT = ImageFont.truetype("arialbd.ttf", 90)
except OSError:
    FONT = ImageFont.load_default()


def row(files, out, labels="abcdef"):
    ims = [Image.open(FIG / f).convert("RGB") for f in files]
    h = min(i.height for i in ims)
    ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    canvas = Image.new("RGB", (sum(i.width for i in ims), h), "white")
    x = 0
    for i, (im, lab) in enumerate(zip(ims, labels)):
        canvas.paste(im, (x, 0))
        ImageDraw.Draw(canvas).text((x + 20, 10), f"({lab})", fill="black", font=FONT)
        x += im.width
    canvas.save(FIG / out, dpi=(300, 300))


row(["fig_shap_summary_Tensile.png", "fig_shap_summary_Hardness.png"], "composite_shap_summary.png")
row(["fig_shap_dependence_Tensile_C.png", "fig_shap_dependence_Tensile_Cr.png",
     "fig_shap_dependence_Tensile_TT.png"], "composite_shap_dependence_tensile.png")
print("ok")
