import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from cycler import cycler
import os


STYLE_ENV_VAR = "PLOT_STYLE"
STYLE_CHOICES = (
    "paper",
    "conference",
    "conference_dark",
    "conference_light",
    "dark_transparent",
    "light_transparent",
    "light",
    "dark",
    "talk",
)


_STYLE_ALIASES = {
    "publication": "paper",
    "pub": "paper",
    "journal": "paper",
    "conference": "conference_dark",
    "conf": "conference_dark",
    "conference_dark": "conference_dark",
    "conference-dark": "conference_dark",
    "dark_conference": "conference_dark",
    "dark-conference": "conference_dark",
    "dark_transparent": "conference_dark",
    "dark-transparent": "conference_dark",
    "transparent": "conference_dark",
    "conference_light": "conference_light",
    "conference-light": "conference_light",
    "light_conference": "conference_light",
    "light-conference": "conference_light",
    "light_transparent": "conference_light",
    "light-transparent": "conference_light",
    # 'talk' == conference_dark geometry (transparent, light ink) but in the
    # phd-talk-design sans face at lecture-theatre sizes. See TALK_* below.
    "talk": "talk",
    "seminar": "talk",
    "slides": "talk",
    "conference_talk": "talk",
    "conference-talk": "talk",
}

# --- talk-style tokens (dark-deck theme) -----------------------------------
# Ink is the deck's primary/secondary text so figures sit on #0B0A1A without
# looking like a pasted-in paper figure. Backgrounds stay transparent.
TALK_INK = "#F2EFF7"
TALK_INK_DIM = "#A9A2C0"
TALK_GRID = "#5A5478"
# Aptos first (present on the user's Mac via Office), then the usual humanist
# sans fallbacks. Carlito is the closest face available on bare Linux/CI, so
# sandbox renders stay visually near the real thing.
SANS_FONT_STACK = [
    "Aptos", "Aptos Display", "Helvetica Neue", "Helvetica", "Arial",
    "Carlito", "Liberation Sans", "DejaVu Sans",
]
TALK_FONT_STACK = SANS_FONT_STACK   # back-compat alias

# Styles that render in a sans face rather than the paper's Times chain.
# A projected figure is read at distance and usually sits on a slide whose
# body text is sans, so a serif figure reads as a pasted-in paper crop. The
# print styles keep the serif chain -- that is what the journal expects.
SANS_STYLES = frozenset({"conference_dark", "conference_light", "talk"})


# --------------------------------------------------------------------------- #
# PER-STYLE SIZING
# --------------------------------------------------------------------------- #
# Every style gets three numbers, so a script can move from print to projection
# without hand-editing figsizes:
#
#   scale      multiplier on figure width AND height (aspect preserved).
#              Feed print inches through scaled_figsize() to pick it up.
#   fontsize   base_fontsize used when the caller passes None.
#   linewidth  lines.linewidth used when the caller passes None.
#
# 'paper' is the print baseline. The conference/light/dark styles are read off
# a slide, so they get a bigger canvas, heavier ink and larger type together --
# and type grows FASTER than the canvas (15/10 = 1.5x vs scale 1.35x), because
# scaling a figure up without scaling type only restores print proportions and
# nothing actually becomes more legible.
#
# 'talk' is deliberately left at scale 1.0: it is already tuned for lecture-
# theatre projection at 17pt on a print-width canvas, and enlarging the canvas
# under it would make its type RELATIVELY smaller than it is today. Raise
# STYLE_SIZING["talk"]["scale"] if you want the bigger canvas there too.
STYLE_SIZING = {
    "paper":            {"scale": 1.00, "fontsize": 10, "linewidth": 1.5},
    "light":            {"scale": 1.35, "fontsize": 15, "linewidth": 2.4},
    "dark":             {"scale": 1.35, "fontsize": 15, "linewidth": 2.4},
    "conference_light": {"scale": 1.35, "fontsize": 15, "linewidth": 2.4},
    "conference_dark":  {"scale": 1.35, "fontsize": 15, "linewidth": 2.4},
    "talk":             {"scale": 1.00, "fontsize": 17, "linewidth": 2.6},
}

# --------------------------------------------------------------------------- #
# PER-STYLE COLORMAP BRIGHTNESS FLOOR
# --------------------------------------------------------------------------- #
# The low end of plasma (and viridis, magma, inferno) is near-black purple. On
# a white page that is the highest-contrast end of the ramp; on a dark slide it
# disappears into the background. So on the dark styles we refuse to sample
# below a floor, compressing the whole ramp into its brighter half:
#
#     x' = floor + x * (1 - floor)
#
# This is applied to EVERY cmap sample -- the automatic prop_cycle AND explicit
# get_cmap_colors(start=..., end=...) / plasma_color(x) calls -- so a script
# that hand-picks plasma_color(0.2) for a paper figure still gets a visible
# colour under conference_dark without changing a line. floor 0.0 is the
# identity, so the light styles are completely unaffected.
STYLE_CMAP_FLOOR = {
    "paper":            0.00,
    "light":            0.00,
    "conference_light": 0.00,
    "dark":             0.35,
    "conference_dark":  0.35,
    "talk":             0.35,
}

# The style most recently installed by apply_plot_style/set_plot_style. This is
# what makes scaled_figsize() and the cmap floor dynamic: a script writes its
# print figsize and colour picks once, and the active style decides how big the
# canvas is and how bright the ramp gets.
_ACTIVE = {"style": "paper", "scale": 1.0, "cmap_floor": 0.0}


def _record_active(style_norm):
    """Remember which style is installed, so the dynamic helpers follow it."""
    _ACTIVE["style"] = style_norm
    _ACTIVE["scale"] = float(
        STYLE_SIZING.get(style_norm, STYLE_SIZING["paper"])["scale"])
    _ACTIVE["cmap_floor"] = float(STYLE_CMAP_FLOOR.get(style_norm, 0.0))


# --------------------------------------------------------------------------- #
# THEME-AWARE INK
# --------------------------------------------------------------------------- #
# Figure code is normally written against a light background, so it names
# concrete light-theme colours: "k" for primary ink, greyscale strings ("0.5",
# "0.8") for guide lines and faint shading, "white" for open marker faces and
# annotation-box fills. Those are exactly wrong on a dark slide -- black curves
# vanish into the background and white boxes glare.
#
# theme_ink() maps one such literal onto the ACTIVE style:
#
#   * under paper / light / conference_light it is the IDENTITY, so a print
#     figure set renders byte-for-byte as it always has;
#   * under dark / conference_dark / talk it inverts greyscale luminance, so
#     "k" -> near-white ink, "0.8" (faint light guide) -> "0.2" (faint dark
#     guide), and "white" -> the style's surface colour.
#
# Named accent colours (hex literals, matplotlib colour names, RGBA tuples) are
# passed through untouched -- they carry meaning and are chosen per-figure.
DARK_STYLES = frozenset({"dark", "conference_dark", "talk"})

# Primary ink and surface (panel/legend/marker fill) per style. The dark values
# match the rcParams that set_plot_style installs, so remapped literals sit in
# the same family as the axis furniture drawn from the theme.
STYLE_INK = {
    "talk":            {"ink": TALK_INK, "surface": "#0B0A1A"},
    "dark":            {"ink": "#EAEAEA", "surface": "#111217"},
    "conference_dark": {"ink": "#EAEAEA", "surface": "#111217"},
}
_LIGHT_INK = {"ink": "k", "surface": "white"}


def style_is_dark(style=None):
    """True if the given (or active) style paints on a dark background."""
    name = normalise_plot_style(style, default="paper") if style is not None \
        else _ACTIVE["style"]
    return name in DARK_STYLES


def current_ink():
    """Primary ink colour of the active style ('k' on light backgrounds)."""
    return STYLE_INK.get(_ACTIVE["style"], _LIGHT_INK)["ink"]


def current_surface():
    """Panel/marker-fill colour of the active style ('white' on light)."""
    return STYLE_INK.get(_ACTIVE["style"], _LIGHT_INK)["surface"]


def theme_ink(c):
    """Map a light-theme ink/grey/white literal onto the active style."""
    if not style_is_dark():
        return c
    if c in ("k", "black"):
        return current_ink()
    if c in ("w", "white"):
        return current_surface()
    try:
        v = float(c)
    except (TypeError, ValueError):
        return c                      # named/hex accent colour -- leave alone
    # Invert luminance, clamped so nothing lands on pure black or pure white.
    return f"{min(max(1.0 - v, 0.08), 0.92):.2f}"


def relative_luminance(c):
    """WCAG relative luminance of any matplotlib colour spec, in [0, 1]."""
    r, g, b = mcolors.to_rgb(c)
    def _lin(u):
        return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def theme_accent(c, *, min_luminance=0.30):
    """Keep a NAMED accent colour legible against the active background.

    theme_ink() handles greys and black/white, but figures also carry deliberate
    hex accents -- a dark gold "#8A6A00" caption, a brown "#B04B00" label -- that
    are chosen for contrast against WHITE. On a dark slide they disappear.

    On a dark style this lifts such a colour along its own hue (raising HSV
    value, then desaturating toward white if that is not enough) until it clears
    min_luminance. Colours that are already bright enough, and every colour on a
    light style, are returned untouched.
    """
    if not style_is_dark():
        return c
    if relative_luminance(c) >= min_luminance:
        return c
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(c))
    for _ in range(24):
        if v < 1.0:
            v = min(1.0, v + 0.06)
        else:
            s = max(0.0, s - 0.06)          # out of value: desaturate to white
        out = mcolors.hsv_to_rgb((h, s, v))
        if relative_luminance(out) >= min_luminance:
            return mcolors.to_hex(out)
        if v >= 1.0 and s <= 0.0:
            break
    return mcolors.to_hex(mcolors.hsv_to_rgb((h, s, v)))


def theme_legend_kw(**overrides):
    """Legend styling (facecolor/edgecolor/framealpha) for the active style.

    Splat into ax.legend()/fig.legend() so legend boxes follow the theme
    instead of painting a white panel onto a dark slide.
    """
    if style_is_dark():
        out = dict(frameon=True, framealpha=0.55,
                   facecolor=current_surface(), edgecolor="0.45")
    else:
        out = dict(frameon=True, framealpha=0.6,
                   facecolor="white", edgecolor="0.7")
    out.update(overrides)
    return out


def current_cmap_floor():
    """Lowest cmap position the active style will sample (0.0 for print)."""
    return float(_ACTIVE["cmap_floor"])


def lift_cmap_x(x):
    """Compress a cmap position into the active style's visible range."""
    floor = current_cmap_floor()
    if floor <= 0.0:
        return float(x)
    return floor + float(x) * (1.0 - floor)


def style_sizing(style=None, **overrides):
    """Return {'scale', 'fontsize', 'linewidth'} for a style, with overrides.

    Pass a style name, or None to use the environment / 'paper' default.
    Keyword overrides win, so a script can keep its own print tuning:

        sizing = style_sizing("paper", linewidth=1.4)
    """
    style_norm = normalise_plot_style(style, default="paper")
    out = dict(STYLE_SIZING.get(style_norm, STYLE_SIZING["paper"]))
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def current_style():
    """Canonical name of the style installed by the last apply/set call."""
    return _ACTIVE["style"]


def current_scale():
    """Canvas scale of the active style (1.0 for print)."""
    return float(_ACTIVE["scale"])


def scaled_figsize(width, height, *, style=None):
    """Scale a PRINT figsize onto the active (or given) style's canvas.

    Aspect ratio is preserved, so a script declares its figure once in print
    inches and every style renders it at the right physical size:

        fig, ax = plt.subplots(figsize=scaled_figsize(COL_W, 2.8))
    """
    s = style_sizing(style)["scale"] if style is not None else current_scale()
    return (float(width) * s, float(height) * s)


def normalise_plot_style(style=None, *, default="paper", env_var=STYLE_ENV_VAR):
    """Return the canonical style name, with environment override support."""
    requested = os.environ.get(env_var) or os.environ.get("PLOT_STYLE") or style or default
    style_norm = str(requested).strip().lower()
    style_norm = _STYLE_ALIASES.get(style_norm, style_norm)
    valid = {"paper", "conference_dark", "conference_light", "light", "dark", "talk"}
    if style_norm not in valid:
        allowed = ", ".join(STYLE_CHOICES)
        raise ValueError(
            f"Unknown plot style '{requested}'. Use one of: {allowed}. "
            f"Set {STYLE_ENV_VAR}=paper|conference_dark|conference_light to override scripts."
        )
    return style_norm


def add_style_argument(parser, *, default=None):
    """Add the standard repository-wide plotting style flag to an ArgumentParser."""
    parser.add_argument(
        "--style",
        choices=STYLE_CHOICES,
        default=default,
        help=(
            "Plot style. 'paper' is light/opaque; 'conference' and "
            "'conference_dark' are dark transparent; 'conference_light' is "
            "light transparent. Can also be set with PLOT_STYLE."
        ),
    )
    return parser


def apply_plot_style(
    style=None,
    *,
    font="Times New Roman",
    base_fontsize=None,
    linewidth=None,
    n_colors=6,
    cmap_name="plasma",
    min_cycle=256,
):
    """Apply a paper/conference plotting style with sensible defaults.

    Font size and line width default to this style's STYLE_SIZING entry, so
    switching style moves type and ink together. Pair with scaled_figsize()
    to move the canvas as well.
    """
    style_norm = normalise_plot_style(style, default="paper")
    sizing = STYLE_SIZING.get(style_norm, STYLE_SIZING["paper"])
    if style_norm == "paper":
        return set_paper_style(
            style="paper",
            font=font,
            base_fontsize=sizing["fontsize"] if base_fontsize is None else base_fontsize,
            linewidth=sizing["linewidth"] if linewidth is None else linewidth,
            n_colors=n_colors,
            cmap_name=cmap_name,
            min_cycle=min_cycle,
        )
    if style_norm == "talk":
        # Larger base than conference_dark: these are read from the back of a
        # lecture theatre, not off a printed column.
        #
        # Scripts that hard-code apply_plot_style("paper", base_fontsize=10)
        # would otherwise drag the talk style back down to print sizing when
        # PLOT_STYLE=talk is set from the environment, so an explicit
        # env override wins over the caller's argument here.
        _env_fs = os.environ.get("PLOT_BASE_FONTSIZE")
        _env_lw = os.environ.get("PLOT_LINEWIDTH")
        return set_plot_style(
            style="talk",
            font=font,
            base_fontsize=(float(_env_fs) if _env_fs
                           else (sizing["fontsize"] if base_fontsize is None else base_fontsize)),
            linewidth=(float(_env_lw) if _env_lw
                       else (sizing["linewidth"] if linewidth is None else linewidth)),
            n_colors=n_colors,
            cmap_name=cmap_name,
            min_cycle=min_cycle,
        )
    return set_plot_style(
        style=style_norm,
        font=font,
        base_fontsize=sizing["fontsize"] if base_fontsize is None else base_fontsize,
        linewidth=sizing["linewidth"] if linewidth is None else linewidth,
        n_colors=n_colors,
        cmap_name=cmap_name,
        min_cycle=min_cycle,
    )


def _colors_from_cmap(*, cmap, n):
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [cmap(lift_cmap_x(0.5))]
    xs = np.linspace(0.15, 0.95, n, endpoint=True)
    return [cmap(lift_cmap_x(x)) for x in xs]


def _distinct_hsv_colors(n, *, s=0.65, v=0.90, a=1.0, h0=0.0):
    n = int(n)
    if n <= 0:
        return []
    phi = 0.6180339887498949
    hs = (h0 + phi * np.arange(n, dtype=float)) % 1.0
    return [
        (*mcolors.hsv_to_rgb((float(h), float(s), float(v))), float(a))
        for h in hs
    ]


def _make_long_color_cycle(*, n_colors, cmap_name="plasma", min_cycle=256):
    n_colors = int(n_colors)
    cycle_len = int(max(min_cycle, max(1, n_colors)))
    cmap = plt.get_cmap(str(cmap_name))
    base_len = int(max(0, n_colors))
    base = _colors_from_cmap(cmap=cmap, n=base_len)

    extra_len = int(max(0, cycle_len - base_len))
    extra = _distinct_hsv_colors(extra_len, h0=0.15)

    return list(base) + list(extra)


def set_plot_style(
    style="light",
    font="Times New Roman",
    base_fontsize=15,
    linewidth=2.0,
    n_colors=6,
    cmap_name="plasma",
    min_cycle=256,
):
    style_norm = normalise_plot_style(style, default="light")
    _record_active(style_norm)
    colors = _make_long_color_cycle(n_colors=n_colors, cmap_name=cmap_name, min_cycle=min_cycle)

    # --- If dark mode: overwrite the first color with white ---
    if style_norm in {"dark", "conference_dark", "talk"}:
        colors[0] = (1.0, 1.0, 1.0, 1.0)   # RGBA white

    light = dict(
        figure_facecolor="#FFFFFF",
        axes_facecolor="#FFFFFF",
        axes_edgecolor="#262626",
        text_color="#262626",
        tick_color="#262626",
        grid_color="#CCCCCC",
    )
    dark = dict(
        figure_facecolor="#111217",
        axes_facecolor="#111217",
        axes_edgecolor="#EAEAEA",
        text_color="#EAEAEA",
        tick_color="#EAEAEA",
        grid_color="#555555",
    )
    paper = dict(
        figure_facecolor="#FFFFFF",
        axes_facecolor="#FFFFFF",
        axes_edgecolor="#000000",
        text_color="#000000",
        tick_color="#000000",
        grid_color="#AAAAAA",
    )

    if style_norm in {"light"}:
        theme = light
    elif style_norm == "conference_light":
        theme = light.copy()
        theme["figure_facecolor"] = "none"
        theme["axes_facecolor"] = "none"
    elif style_norm in {"dark", "conference_dark"}:
        theme = dark.copy()
        if style_norm == "conference_dark":
            theme["figure_facecolor"] = "none"
            theme["axes_facecolor"] = "none"
    elif style_norm == "talk":
        theme = dict(
            figure_facecolor="none",
            axes_facecolor="none",
            axes_edgecolor=TALK_INK_DIM,
            text_color=TALK_INK,
            tick_color=TALK_INK_DIM,
            grid_color=TALK_GRID,
        )
    elif style_norm == "paper":
        theme = paper
    else:
        raise ValueError(
            "Unknown style. Use one of: 'paper', 'conference', 'conference_dark', 'conference_light', 'dark_transparent', 'light_transparent', 'light', 'dark'."
        )

    is_transparent = theme["figure_facecolor"] == "none"

    plt.rcParams.update({
        # Figure & font
        "figure.dpi": 150,
        "figure.autolayout": True,
        "font.family": "serif",
        # Fallback chain: honour the requested font first (Times New Roman on
        # macOS / MikTeX / TeXLive Mac builds), then STIX Two Text and STIX
        # General (Times-clone shipped with matplotlib, present in every
        # matplotlib install including sandbox Linux), TeX Gyre Termes (Times
        # metrics, ships with TeX Live), Liberation Serif (Times metrics,
        # ships with most Linux distros), and finally DejaVu Serif as the
        # absolute fallback. This keeps embedded PDF fonts consistent across
        # render environments — the paper set always looks Times-like.
        "font.serif": [font, "STIX Two Text", "STIXGeneral", "TeX Gyre Termes",
                        "Liberation Serif", "DejaVu Serif"],
        "font.size": base_fontsize,

        "mathtext.fontset": "cm",

        # Axes & text colors
        "axes.facecolor": theme["axes_facecolor"],
        "figure.facecolor": theme["figure_facecolor"],
        "axes.edgecolor": theme["axes_edgecolor"],
        "axes.labelcolor": theme["text_color"],
        "axes.titlecolor": theme["text_color"],
        "text.color": theme["text_color"],
        "xtick.color": theme["tick_color"],
        "ytick.color": theme["tick_color"],
        "grid.color": theme["grid_color"],

        # Sizes
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,

        # Lines & cycle
        "axes.prop_cycle": cycler(color=colors),
        "lines.linewidth": linewidth,
        "lines.markersize": 5,

        "patch.edgecolor": theme["text_color"],
        "patch.facecolor": theme["text_color"],

        # Grid
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.3,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": base_fontsize - 1,
        "legend.labelcolor": theme["text_color"],

        # Colormap
        "image.cmap": str(cmap_name),

        # Save
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": theme["figure_facecolor"],
        "savefig.edgecolor": theme["figure_facecolor"],
        "savefig.transparent": bool(is_transparent),
    })

    if style_norm in SANS_STYLES:
        # Slide styles render in a sans face. Math has to switch with it:
        # 'cm' is Computer Modern serif and clashes badly next to sans body
        # text, so the whole label reads as two different fonts spliced
        # together. 'dejavusans' is the sans-matched fontset and ships with
        # every matplotlib install, so this renders the same everywhere.
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": list(SANS_FONT_STACK),
            "mathtext.fontset": "dejavusans",
        })

    if style_norm == "talk":
        # Projection sizing. Ticks are bumped ABOVE the base rather than below
        # it (the paper/conference styles use base-1) because these are read at
        # distance; axis labels lead, tick labels are only 1pt down.
        plt.rcParams.update({
            "axes.titlesize": base_fontsize + 2,
            "axes.labelsize": base_fontsize + 2,
            "xtick.labelsize": base_fontsize,
            "ytick.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.major.size": 6.5,
            "ytick.major.size": 6.5,
            "xtick.minor.size": 3.5,
            "ytick.minor.size": 3.5,
            "lines.markersize": 7,
            "grid.alpha": 0.45,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.06,
            # Legends need a hint of body behind them on a dark slide, but must
            # not paint a light box: a low-alpha dark patch keeps text legible
            # where it overlaps data without breaking transparency.
            "legend.frameon": True,
            "legend.framealpha": 0.55,
            "legend.facecolor": "#0B0A1A",
            "legend.edgecolor": TALK_GRID,
            "legend.labelcolor": TALK_INK,
            "patch.edgecolor": TALK_INK,
            "patch.facecolor": TALK_INK,
        })

    plt.set_cmap(str(cmap_name))


def get_cmap_colors(*, cmap_name="plasma", n=6, start=0.15, end=0.95):
    """Sample n colours from a colormap, lifted into the active style's range.

    start/end are given in PRINT terms; on a dark style they are compressed
    above the brightness floor so nothing lands in the invisible near-black end.
    """
    n = int(n)
    if n <= 0:
        return []
    cmap = plt.get_cmap(str(cmap_name))
    if n == 1:
        return [cmap(lift_cmap_x(0.5 * (start + end)))]
    xs = np.linspace(float(start), float(end), n, endpoint=True)
    return [cmap(lift_cmap_x(x)) for x in xs]


def plasma_colors(*, n=6, start=0.15, end=0.95):
    return get_cmap_colors(cmap_name="plasma", n=n, start=start, end=end)


def plasma_color(x):
    """Sample plasma at x, lifted into the active style's visible range."""
    return plt.get_cmap("plasma")(lift_cmap_x(x))


def save_figure(fig, path_no_ext, *, dpi=200):
    fig.savefig(
        f"{path_no_ext}.png",
        dpi=dpi,
        transparent=bool(plt.rcParams.get("savefig.transparent", False)),
        facecolor=plt.rcParams.get("savefig.facecolor", "auto"),
        edgecolor=plt.rcParams.get("savefig.edgecolor", "auto"),
    )


def current_style_is_transparent():
    return bool(plt.rcParams.get("savefig.transparent", False))


def current_savefig_kwargs(**kwargs):
    """Return savefig kwargs consistent with the active plot style."""
    out = {
        "transparent": current_style_is_transparent(),
        "facecolor": plt.rcParams.get("savefig.facecolor", "auto"),
        "edgecolor": plt.rcParams.get("savefig.edgecolor", "auto"),
    }
    out.update(kwargs)
    return out


def set_paper_style(
    style=None,
    font="Times New Roman",
    base_fontsize=10,
    linewidth=1.5,
    n_colors=6,
    cmap_name="plasma",
    min_cycle=256,
    figure_width_inches=3.5,
):
    """
    Set publication-ready plot style with appropriate font sizes for journal figures.
    
    Parameters
    ----------
    font : str
        Font family (default: "Times New Roman")
    base_fontsize : float
        Base font size in points. For single-column figures (3.5"), use 8-10pt.
        For double-column figures (7"), use 9-11pt. (default: 10)
    linewidth : float
        Default line width (default: 1.5)
    n_colors : int
        Number of colors in the color cycle (default: 6)
    cmap_name : str
        Colormap name (default: "plasma")
    min_cycle : int
        Minimum color cycle length (default: 256)
    figure_width_inches : float
        Typical figure width for sizing reference. Single-column: 3.5", double-column: 7" (default: 3.5)
    
    Notes
    -----
    Font size guidelines for publication (assuming 3.5" single-column width):
    - Axis labels: 10-11 pt
    - Tick labels: 8-9 pt
    - Legend: 8-9 pt
    - Title (if used): 11-12 pt
    
    Colors are optimized for white backgrounds and print reproduction.
    """
    try:
        style_norm = normalise_plot_style(style, default="paper")
    except ValueError:
        # Backwards compatibility: before style selection was added, the first
        # positional argument to set_paper_style was the font name.
        if style is None:
            raise
        font = str(style)
        style_norm = normalise_plot_style(None, default="paper")
    _record_active(style_norm)
    if style_norm != "paper":
        set_plot_style(
            style=style_norm,
            font=font,
            base_fontsize=base_fontsize,
            linewidth=linewidth,
            n_colors=n_colors,
            cmap_name=cmap_name,
            min_cycle=min_cycle,
        )
        return

    set_plot_style(
        style=style_norm,
        font=font,
        base_fontsize=base_fontsize,
        linewidth=linewidth,
        n_colors=n_colors,
        cmap_name=cmap_name,
        min_cycle=min_cycle,
    )
    
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "savefig.dpi": 300,
    })


def annotate_auto_arrow(
    *,
    ax,
    text,
    xy,
    xytext,
    color=None,
    arrow_color=None,
    **kwargs,
):
    if color is None:
        color = plt.rcParams.get("text.color", "k")
    if arrow_color is None:
        arrow_color = color
    arrowprops = dict(kwargs.pop("arrowprops", {}) or {})
    arrowprops.setdefault("arrowstyle", "->")
    arrowprops.setdefault("color", arrow_color)
    arrowprops.setdefault("lw", plt.rcParams.get("lines.linewidth", 1.5))
    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color,
        arrowprops=arrowprops,
        **kwargs,
    )
