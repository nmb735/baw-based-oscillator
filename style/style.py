# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# "Belt DEE" signature style — IEEE-calibre plots for RF / antenna engineering.                                                        #
# Inspired by MATLAB, CST, HFSS, ADS, and IEEE Transactions aesthetics, elevated with a personal palette and typography.              #
# Import this module once at the top of any plotting script; all rcParams are applied globally on import.                             #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import seaborn                 as sns                     # Available for notebook use; not used for global style here.               #
import matplotlib              as mpl                     # Core matplotlib.                                                           #
import matplotlib.pyplot       as plt                     # Pyplot interface.                                                          #
from   cycler                  import cycler              # Property cycler for per-trace styles.                                      #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------- "BELT DEE" SIGNATURE COLOUR PALETTE -------------------------------------------- #
#                                                                                                                                      #
#  Designed for RF/antenna S-parameter plots:                                                                                          #
#    • Cobalt blue  (#0047AB) and Cardinal red (#C41E3A) lead the cycle —                                                              #
#      they pair perfectly for simulation-vs-measurement overlays.                                                                     #
#    • All 10 colours are distinguishable under Deuteranopia & Protanopia.                                                             #
#    • Adjacent colours maintain ≥3:1 contrast ratio on white (#FFFFFF).                                                               #
#                                                                                                                                      #
COLOR_CYCLE = [                                                                                                                        #
    '#0047AB',   # Cobalt blue       — primary   (simulation / reference)                                                             #
    '#C41E3A',   # Cardinal red      — secondary (measurement / comparison)                                                           #
    '#1D7340',   # Forest green      — tertiary                                                                                        #
    '#D97000',   # Dark amber        — quaternary                                                                                      #
    '#5B2D8E',   # Deep purple       — quinary                                                                                         #
    '#007BA7',   # Cerulean blue     — senary                                                                                          #
    '#706B6B',   # Warm grey         — septenary                                                                                       #
    '#8B4513',   # Saddle brown      — octonary                                                                                        #
    '#1A8476',   # Deep teal         — nonary                                                                                          #
    '#C0541A',   # Burnt sienna      — denary                                                                                          #
]                                                                                                                                      #
#                                                                                                                                      #
#  Linestyles paired 1-to-1 with colours: solid / dashed / dash-dot / dotted                                                          #
#  (4-cycle) keeps adjacent traces apart in black-and-white print.                                                                     #
#                                                                                                                                      #
LINESTYLE_CYCLE = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']                                                              #
MARKER_CYCLE    = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'p', 'h']                                                                 #
#                                                                                                                                      #
#  Additive cycler: trace i gets (color[i], linestyle[i]).                                                                             #
#  Markers are OFF by default ("lines.marker": "None") — S-param sweeps have                                                          #
#  thousands of points and markers would be illegible.  Enable per-plot if needed.                                                     #
#                                                                                                                                      #
custom_cycler = cycler(color=COLOR_CYCLE) + cycler(linestyle=LINESTYLE_CYCLE)                                                          #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ---------------------------------------------------------- SIZE HIERARCHY --------------------------------------------------------- #
#                                                                                                                                      #
#  Font sizes are calibrated for on-screen / high-resolution PDF work.                                                                 #
#  When targeting a printed IEEE single-column figure (3.5"), scale all                                                                #
#  sizes by ≈ 0.45 or use FIG_WIDTH_1COL / FIG_HEIGHT_1COL directly.                                                                  #
#                                                                                                                                      #
TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE = 24, 20, 18, 18                                                                       #
ANNOTATION_SIZE = 14                                                                                                                   #
SMALL_SIZE      = 12                                                                                                                   #
#                                                                                                                                      #
#  Screen / presentation (default figure canvas)                                                                                       #
FIG_WIDTH,  FIG_HEIGHT  = 10, 7                                                                                                        #
#                                                                                                                                      #
#  IEEE publication — single column (88 mm) and double column (178 mm)                                                                 #
FIG_WIDTH_1COL,  FIG_HEIGHT_1COL  = 3.5,  2.625                                                                                       #
FIG_WIDTH_2COL,  FIG_HEIGHT_2COL  = 7.0,  5.25                                                                                        #
#                                                                                                                                      #
#  Line / marker geometry                                                                                                               #
LINE_WIDTH        = 1.75                                                                                                               #
AXES_LINEWIDTH    = 1.0                                                                                                                #
MARKER_SIZE       = 7                                                                                                                  #
MARKER_EDGE_WIDTH = 1.2                                                                                                                #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ---------------------------------------------------------- COLOUR TOKENS --------------------------------------------------------- #
#                                                                                                                                      #
#  Named tokens used in plotters and importable by any module.                                                                         #
#                                                                                                                                      #
SPINE_COLOR      = '#1C1C1C'   # Near-black — crisper than pure #000000 on LCD.                                                       #
GRID_MAJOR_COLOR = '#C8C8C8'   # Mid-light grey for major gridlines.                                                                  #
GRID_MINOR_COLOR = '#E6E6E6'   # Very light grey for minor gridlines.                                                                  #
GRID_MAJOR_LW    = 0.65                                                                                                                #
GRID_MINOR_LW    = 0.35                                                                                                                #
TAG_NAME         = "Belt DEE"  # Style signature.                                                                                      #
#                                                                                                                                      #
#  Regulatory-band shading — a very light neutral grey wash marking an operating                                                       #
#  band behind the traces (see shade_band).  Desaturated and low-alpha so it                                                           #
#  never competes with the data or the grid.                                                                                           #
BAND_SHADE_COLOR = '#9E9E9E'   # Neutral grey; the alpha does the work, not the hue.                                                   #
BAND_SHADE_ALPHA = 0.32        # ~#D2D2D2 over white — darkened on request; the stripe is ~2 MHz on a 150-200 MHz axis.  #
#                                                                                                                                      #
#  Zoom-slice rules - the dashed verticals marking the frequency slice an inset                                                        #
#  magnifies.  Kept lighter than a data trace so they mark without competing.                                                          #
ZOOM_RULE_COLOR  = '#5C5C5C'   # Darkened on request; shared with the inset leaders.                                                   #
ZOOM_RULE_ALPHA  = 0.70                                                                                                                #
ZOOM_RULE_LW     = 1.0                                                                                                                 #
ZOOM_RULE_DASH   = (0, (4, 4))                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- REGULATORY BANDS ---------------------------------------------------------- #
#                                                                                                                                      #
#  European UHF RFID interrogator band, ETSI EN 302 208.  The four high-power                                                          #
#  channels are centred on 865.7 / 866.3 / 866.9 / 867.5 MHz, so the shaded span                                                       #
#  runs 865.7-867.5 MHz, inside the wider 865-868 MHz allocation.  Given in both                                                       #
#  units so a plotter can shade it whatever its frequency axis.                                                                        #
#                                                                                                                                      #
ETSI_UHF_BAND_MHZ = (865.7, 867.5)                                                                                                     #
ETSI_UHF_BAND_GHZ = (ETSI_UHF_BAND_MHZ[0] / 1e3, ETSI_UHF_BAND_MHZ[1] / 1e3)                                                           #
#                                                                                                                                      #
#  Full 865-868 MHz allocation — swap these in for a wider, more legible                                                               #
#  stripe; the caption wording must then follow.                                                                                       #
ETSI_UHF_ALLOC_MHZ = (865.0, 868.0)                                                                                                    #
ETSI_UHF_ALLOC_GHZ = (ETSI_UHF_ALLOC_MHZ[0] / 1e3, ETSI_UHF_ALLOC_MHZ[1] / 1e3)                                                        #
#                                                                                                                                      #
#  Ready-made caption sentence, so every figure that shades the band words it                                                          #
#  identically.                                                                                                                        #
ETSI_UHF_CAPTION  = ('The shaded vertical band indicates the European UHF RFID band '                                                  #
                     f'(ETSI, {ETSI_UHF_BAND_MHZ[0]:.1f}-{ETSI_UHF_BAND_MHZ[1]:.1f} MHz).')                                            #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- GLOBAL rcPARAMS --------------------------------------------------------- #
#                                                                                                                                      #
#  Applied once on import.  Individual plots may override any key via                                                                  #
#  plt.rcParams or as keyword arguments to the plotter methods.                                                                        #
#                                                                                                                                      #
plt.rcParams.update({                                                                                                                  #
                                                                                                                                       #
    # ── Figure ────────────────────────────────────────────────────────────── #
    "figure.figsize":              (FIG_WIDTH, FIG_HEIGHT),                                                                           #
    "figure.dpi":                  600,                                                                                                #
    "figure.titlesize":            TITLE_SIZE,                                                                                         #
    "figure.facecolor":            "white",                                                                                            #
    "figure.edgecolor":            "white",                                                                                            #
                                                                                                                                       #
    # ── Axes ──────────────────────────────────────────────────────────────── #
    "axes.facecolor":              "white",                                                                                            #
    "axes.edgecolor":              SPINE_COLOR,                                                                                        #
    "axes.linewidth":              AXES_LINEWIDTH,                                                                                     #
    "axes.titlesize":              TITLE_SIZE,                                                                                         #
    "axes.labelsize":              LABEL_SIZE,                                                                                         #
    "axes.labelpad":               6.0,                                                                                                #
    "axes.labelcolor":             SPINE_COLOR,                                                                                        #
    "axes.spines.top":             True,   # Full box — MATLAB / CST / HFSS convention.                                               #
    "axes.spines.right":           True,                                                                                               #
    "axes.prop_cycle":             custom_cycler,                                                                                      #
    "axes.axisbelow":              True,   # Data drawn above grid.                                                                    #
                                                                                                                                       #
    # ── Grid (rcParams sets the major-grid style;                           # #
    #         apply_minor_grid() handles the minor-grid style at axes level) # #
    "axes.grid":                   True,                                                                                               #
    "axes.grid.which":             "major",                                                                                            #
    "axes.grid.axis":              "both",                                                                                             #
    "grid.color":                  GRID_MAJOR_COLOR,                                                                                   #
    "grid.linestyle":              "-",                                                                                                #
    "grid.linewidth":              GRID_MAJOR_LW,                                                                                      #
    "grid.alpha":                  1.0,                                                                                                #
                                                                                                                                       #
    # ── Ticks — inward on all four sides (IEEE / CST / HFSS standard) ────── #
    "xtick.labelsize":             TICK_SIZE,                                                                                          #
    "ytick.labelsize":             TICK_SIZE,                                                                                          #
    "xtick.direction":             "in",                                                                                               #
    "ytick.direction":             "in",                                                                                               #
    "xtick.top":                   True,                                                                                               #
    "ytick.right":                 True,                                                                                               #
    "xtick.major.size":            5.5,                                                                                                #
    "xtick.minor.size":            3.0,                                                                                                #
    "ytick.major.size":            5.5,                                                                                                #
    "ytick.minor.size":            3.0,                                                                                                #
    "xtick.major.width":           AXES_LINEWIDTH,                                                                                     #
    "xtick.minor.width":           0.7,                                                                                                #
    "ytick.major.width":           AXES_LINEWIDTH,                                                                                     #
    "ytick.minor.width":           0.7,                                                                                                #
    "xtick.minor.visible":         True,                                                                                               #
    "ytick.minor.visible":         True,                                                                                               #
    "xtick.color":                 SPINE_COLOR,                                                                                        #
    "ytick.color":                 SPINE_COLOR,                                                                                        #
                                                                                                                                       #
    # ── Lines ─────────────────────────────────────────────────────────────── #
    "lines.linewidth":             LINE_WIDTH,                                                                                         #
    "lines.markersize":            MARKER_SIZE,                                                                                        #
    "lines.markeredgewidth":       MARKER_EDGE_WIDTH,                                                                                  #
    "lines.marker":                "None",  # No markers by default — enable per-line for sparse data. #                              #
    "lines.solid_capstyle":        "round",                                                                                            #
    "lines.solid_joinstyle":       "round",                                                                                            #
                                                                                                                                       #
    # ── Font and LaTeX ────────────────────────────────────────────────────── #
    "font.family":                 "serif",                                                                                            #
    "font.serif":                  ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],                                       #
    "font.size":                   LABEL_SIZE,                                                                                         #
    "text.usetex":                 True,                                                                                               #
    "text.latex.preamble":         r"\usepackage{amsmath,amssymb,amsfonts,mathrsfs,bm}",                                               #
                                                                                                                                       #
    # ── Legend ────────────────────────────────────────────────────────────── #
    "legend.fontsize":             LEGEND_SIZE,                                                                                        #
    "legend.framealpha":           0.90,                                                                                               #
    "legend.edgecolor":            "#AAAAAA",  # Subtle grey border.                                                                  #
    "legend.fancybox":             False,      # Square corners — professional in print.                                              #
    "legend.markerscale":          1.0,                                                                                                #
    "legend.labelspacing":         0.35,                                                                                               #
    "legend.handlelength":         2.5,                                                                                                #
    "legend.handleheight":         0.7,                                                                                                #
    "legend.borderpad":            0.5,                                                                                                #
    "legend.columnspacing":        1.2,                                                                                                #
    "legend.title_fontsize":       LABEL_SIZE,                                                                                         #
                                                                                                                                       #
    # ── Saving ────────────────────────────────────────────────────────────── #
    "savefig.dpi":                 600,                                                                                                #
    "savefig.bbox":                "tight",                                                                                            #
    "savefig.facecolor":           "white",                                                                                            #
    "savefig.transparent":         False,                                                                                               #
                                                                                                                                       #
    # ── PDF / PS font embedding (vector text survives zoom and printing) ─── #
    "pdf.fonttype":                42,                                                                                                 #
    "ps.fonttype":                 42,                                                                                                 #
})                                                                                                                                     #
                                                                                                                                       #
plt.rc("text", usetex=True)                                                                                                            #
plt.rc("font", family="serif")                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- AXES HELPER ------------------------------------------------------------ #
def apply_minor_grid(ax) -> None:                                                                                                      #
    """                                                                                                                                #
    Add a faint minor-grid layer on top of the major grid already drawn by                                                             #
    rcParams.  Call this inside _style_ax() in any plotter module.                                                                     #
                                                                                                                                       #
    Major grid (set by rcParams):  GRID_MAJOR_COLOR, GRID_MAJOR_LW                                                                    #
    Minor grid (set here):         GRID_MINOR_COLOR, GRID_MINOR_LW                                                                    #
    """                                                                                                                                #
    ax.grid(True,  which="minor",                                                                                                      #
            color=GRID_MINOR_COLOR, linewidth=GRID_MINOR_LW,                                                                          #
            linestyle="-", alpha=0.8)                                                                                                  #
# =================================================================================================================================== #




# =================================================================================================================================== #
# --------------------------------------------------------- BAND HELPER ------------------------------------------------------------- #
def shade_band(ax, band=ETSI_UHF_BAND_GHZ, color=BAND_SHADE_COLOR, alpha=BAND_SHADE_ALPHA) -> None:                                    #
    """                                                                                                                                #
    Wash a faint grey vertical stripe over a frequency interval to mark an                                                             #
    operating band — by default the European UHF RFID band (ETSI EN 302 208).                                                          #
                                                                                                                                       #
    The patch is drawn at zorder 0, beneath both the grid and the traces, and is                                                       #
    kept out of the legend, so it reads as background rather than as a series.                                                         #
    Declare the band in the figure caption; ETSI_UHF_CAPTION has the wording.                                                          #
                                                                                                                                       #
    Parameters:                                                                                                                        #
        - ax    [plt.Axes]   : The axes to shade.                                                                                      #
        - band  [tuple|None] : (f_lo, f_hi) in the units of the x-axis — GHz by                                                        #
                               default; pass ETSI_UHF_BAND_MHZ for an MHz axis.                                                        #
                               ``None`` is a no-op.                                                                                    #
        - color [str]        : Fill colour.                                                                                            #
        - alpha [float]      : Fill opacity.                                                                                           #
                                                                                                                                       #
    Returns:                                                                                                                           #
        - None                                                                                                                         #
    """                                                                                                                                #
    if band is None:                                                                                                                   #
        return                                                                                                                         #
    ax.axvspan(band[0], band[1],                                                                                                       #
               facecolor=color, alpha=alpha,                                                                                           #
               edgecolor='none', linewidth=0,                                                                                          #
               zorder=0, label='_nolegend_')                                                                                           #
# =================================================================================================================================== #




# =================================================================================================================================== #
# --------------------------------------------------------- ZOOM HELPERS ------------------------------------------------------------ #
def zoom_rules(ax, span, color=ZOOM_RULE_COLOR, alpha=ZOOM_RULE_ALPHA,                                                                 #
               linewidth=ZOOM_RULE_LW, dashes=ZOOM_RULE_DASH) -> None:                                                                 #
    """                                                                                                                                #
    Mark, with a dashed vertical rule at each edge, the frequency slice that an                                                        #
    inset magnifies.                                                                                                                   #
                                                                                                                                       #
    Rules rather than a rectangle: an inset carrying two y-scales magnifies a                                                          #
    slice of *frequency* and everything within it, so no box drawn in the units                                                        #
    of one axis can honestly bound what the inset shows.                                                                               #
                                                                                                                                       #
    Parameters:                                                                                                                        #
        - ax        [plt.Axes] : Axes to mark.                                                                                         #
        - span      [tuple]    : (f_lo, f_hi) in the units of the x-axis.                                                              #
        - color     [str]      : Rule colour.                                                                                          #
        - alpha     [float]    : Rule opacity.                                                                                         #
        - linewidth [float]    : Rule width in points.                                                                                 #
        - dashes               : Matplotlib dash spec.                                                                                 #
                                                                                                                                       #
    Returns:                                                                                                                           #
        - None                                                                                                                         #
    """                                                                                                                                #
    for f in span:                                                                                                                     #
        ax.axvline(f, color=color, linestyle=dashes, linewidth=linewidth,                                                              #
                   alpha=alpha, zorder=1.5)                                                                                            #
                                                                                                                                       #
                                                                                                                                       #
def link_inset(ax, axi, x, y_top, y_bot, color=ZOOM_RULE_COLOR, alpha=ZOOM_RULE_ALPHA,                                                 #
               linewidth=ZOOM_RULE_LW, dashes=ZOOM_RULE_DASH) -> None:                                                                 #
    """                                                                                                                                #
    Draw the two leaders joining a zoom slice to its inset, running from ``x``                                                         #
    on the parent axes to the inset's two LEFT corners.                                                                                #
                                                                                                                                       #
    Hand-drawn rather than left to ``indicate_inset_zoom``, which picks corners                                                        #
    by geometry: with the inset below and right of the slice it reaches for the                                                        #
    lower-right corner and sends a leader skimming along the x-axis.  Styled                                                           #
    like the rules, so slice and leaders read as one device, and drawn at                                                              #
    zorder 1.6 so a trace crossing a leader passes in front of it.                                                                     #
                                                                                                                                       #
    Parameters:                                                                                                                        #
        - ax    [plt.Axes] : Parent axes.                                                                                              #
        - axi   [plt.Axes] : The inset axes.                                                                                           #
        - x     [float]    : Frequency the leaders spring from - normally the                                                          #
                             slice edge nearest the inset.                                                                             #
        - y_top [float]    : Upper anchor, in parent data units.                                                                       #
        - y_bot [float]    : Lower anchor, in parent data units.                                                                       #
                                                                                                                                       #
    Returns:                                                                                                                           #
        - None                                                                                                                         #
    """                                                                                                                                #
    from matplotlib.patches import ConnectionPatch                                                                                     #
    for y, corner in ((y_top, (0.0, 1.0)), (y_bot, (0.0, 0.0))):                                                                       #
        ax.add_artist(ConnectionPatch(                                                                                                 #
            xyA=(x, y), coordsA=ax.transData,                                                                                          #
            xyB=corner, coordsB=axi.transAxes,                                                                                         #
            color=color, linewidth=linewidth, linestyle=dashes,                                                                        #
            alpha=alpha, zorder=1.6,                                                                                                   #
        ))                                                                                                                             #
# =================================================================================================================================== #
