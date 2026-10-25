import os
import shutil

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Sudoku Studio", page_icon="▦", layout="wide")

WARP_SIZE = 450
ALLOWED_TYPES = ["png", "jpg", "jpeg", "webp"]


# ----------------------------------------------------------------------------
# Tesseract
# ----------------------------------------------------------------------------
def configure_tesseract():
    """Find the Tesseract executable on Windows or Linux (incl. Streamlit Cloud)."""
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    executable = next((p for p in candidates if p and os.path.isfile(p)), None)
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
    elif shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")


def tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


configure_tesseract()


# ----------------------------------------------------------------------------
# Sudoku logic
# ----------------------------------------------------------------------------
def is_safe(row, col, num, board):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    sr, sc = row - row % 3, col - col % 3
    for i in range(3):
        for j in range(3):
            if board[sr + i][sc + j] == num:
                return False
    return True


def solve_sudoku(board):
    """Backtracking solver that always fills the most constrained empty cell first
    (much faster than scanning cells in order, so hard puzzles don't time out)."""
    best, best_options = None, None
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                options = [n for n in range(1, 10) if is_safe(r, c, n, board)]
                if not options:
                    return False
                if best is None or len(options) < len(best_options):
                    best, best_options = (r, c), options
                    if len(options) == 1:
                        break
        if best is not None and len(best_options) == 1:
            break
    if best is None:
        return True
    r, c = best
    for n in best_options:
        board[r][c] = n
        if solve_sudoku(board):
            return True
        board[r][c] = 0
    return False


def board_is_valid(board):
    for r in range(9):
        for c in range(9):
            value = board[r][c]
            if value == 0:
                continue
            board[r][c] = 0
            valid = is_safe(r, c, value, board)
            board[r][c] = value
            if not valid:
                return False, (r, c)
    return True, None


# ----------------------------------------------------------------------------
# Computer vision + OCR
# ----------------------------------------------------------------------------
def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32",
    )


def find_and_warp_board(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 1000 / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    annotated = img_bgr.copy()
    quad = None
    image_area = img_bgr.shape[0] * img_bgr.shape[1]
    for contour in contours[:10]:
        if cv2.contourArea(contour) < 0.15 * image_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            quad = order_points(approx)
            break

    if quad is None:
        resized = cv2.resize(gray, (WARP_SIZE, WARP_SIZE), interpolation=cv2.INTER_AREA)
        return resized, annotated, False

    cv2.polylines(annotated, [quad.astype(int)], True, (58, 26, 19), 9)    # ink underlay
    cv2.polylines(annotated, [quad.astype(int)], True, (63, 210, 255), 5)  # yellow marker
    dst = np.array(
        [[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(gray, matrix, (WARP_SIZE, WARP_SIZE))
    return warped, annotated, True


def extract_digit_image(cell_bin):
    height, width = cell_bin.shape
    margin = int(0.12 * min(height, width))
    cell = cell_bin[margin:height - margin, margin:width - margin]
    h, w = cell.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(cell, connectivity=8)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]
        touches_border = x <= 0 or y <= 0 or x + cw >= w or y + ch >= h
        if touches_border or area < 0.03 * h * w or ch < 0.3 * h:
            continue
        if not (0.2 * w < cx < 0.8 * w and 0.2 * h < cy < 0.8 * h):
            continue
        if area > best_area:
            best, best_area = i, area
    if best is None:
        return None

    x, y, w, h, _ = stats[best]
    digit = np.where(labels == best, 255, 0).astype(np.uint8)[y:y + h, x:x + w]
    side = max(h, w) + 12
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = digit
    canvas = cv2.resize(canvas, (80, 80), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas = 255 - canvas
    return cv2.copyMakeBorder(canvas, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)


def ocr_digit(digit_img):
    config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
    text = pytesseract.image_to_string(digit_img, config=config).strip()
    return int(text[0]) if text and text[0] in "123456789" else 0


def read_board(warped_gray):
    binary = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    step = WARP_SIZE // 9
    board = [[0] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            cell = binary[r * step:(r + 1) * step, c * step:(c + 1) * step]
            digit_img = extract_digit_image(cell)
            if digit_img is not None:
                board[r][c] = ocr_digit(digit_img)
    return board


def process_image(uploaded_file):
    pil = Image.open(uploaded_file)
    pil = ImageOps.exif_transpose(pil).convert("RGB")  # respects phone-photo rotation
    image_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    warped, annotated, found = find_and_warp_board(image_bgr)
    board = read_board(warped)
    return board, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), warped, found


# ----------------------------------------------------------------------------
# DESIGN: all visual styling lives in this one CSS block.
# Change the colours / fonts in the :root section at the top to re-theme the app.
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600;700&display=swap');

/* ===== 1. DESIGN TOKENS: edit these to re-theme everything ===== */
:root{
  --ink:#131A3A;          /* main dark colour: hero, text, grid lines   */
  --ink-2:#2A3358;
  --muted:#5E6786;        /* secondary text                              */
  --paper:#F4F5FA;        /* page background                             */
  --card:#FFFFFF;         /* card background                             */
  --line:#E1E4EF;         /* hairlines and borders                       */
  --mark:#FFD23F;         /* highlighter yellow: buttons + solved cells  */
  --mark-soft:#FFF1B8;
  --radius:20px;
  --display:'Bricolage Grotesque',system-ui,sans-serif;
  --body:'Instrument Sans',system-ui,sans-serif;
}

/* ===== 2. PAGE FRAME ===== */
.stApp, [data-testid="stAppViewContainer"]{background:var(--paper);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"], .block-container{max-width:1120px;padding:1.2rem 1.5rem 3rem;}
footer, #MainMenu, [data-testid="stDecoration"]{display:none !important;}
/* hide the empty box Streamlit makes around this <style> block */
[data-testid="stElementContainer"]:has(style), [data-testid="element-container"]:has(style){display:none;}

/* ===== 3. TOP BAR + HERO ===== */
.topbar{display:flex;align-items:center;gap:12px;margin:0 0 16px 4px;}
.logo{width:34px;height:34px;border-radius:10px;background:var(--ink);display:grid;grid-template-columns:repeat(3,1fr);gap:2px;padding:7px;}
.logo span{background:rgba(255,255,255,.28);border-radius:2px;}
.logo span.on{background:var(--mark);}
.brand-name{font:800 20px var(--display);letter-spacing:-.02em;color:var(--ink);}

.hero{display:grid;grid-template-columns:1.05fr .95fr;gap:48px;align-items:center;background:var(--ink);border-radius:28px;padding:56px;margin:0 0 22px;}
.hero-title{font:800 clamp(34px,4.6vw,58px)/1.04 var(--display);letter-spacing:-.03em;color:#fff;margin:0 0 18px;}
.hero-lede{font:400 17px/1.6 var(--body);color:#C9CEE6;max-width:46ch;margin:0 0 26px;}
.hero-points{display:grid;gap:11px;}
.hero-point{display:flex;gap:12px;align-items:flex-start;font:500 15px/1.4 var(--body);color:#fff;}
.hero-point:before{content:"";flex:none;width:11px;height:11px;margin-top:.3em;border-radius:3px;background:var(--mark);}
.hero-board .sdk{box-shadow:0 30px 70px rgba(0,0,0,.38);border-color:#fff;}

/* ===== 4. THE SUDOKU GRID (hero + solution) ===== */
.sdk{container-type:inline-size;display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1;width:100%;max-width:460px;margin:0 auto;
     background:#fff;border:3px solid var(--ink);border-radius:12px;overflow:hidden;}
.sdk .c{display:grid;place-items:center;border-right:1px solid var(--line);border-bottom:1px solid var(--line);
        font-family:var(--display);font-size:clamp(14px,4.4vw,24px);font-size:5.4cqw;font-weight:600;color:var(--ink);}
.sdk .c:nth-child(9n){border-right:0;}
.sdk .c:nth-child(n+73){border-bottom:0;}
.sdk .c.br{border-right:2px solid var(--ink);}
.sdk .c.bb{border-bottom:2px solid var(--ink);}
.sdk .c.g{font-weight:800;}
/* solved cells get swiped over with a highlighter, one after another */
.sdk .c.s{font-weight:600;background:linear-gradient(var(--mark),var(--mark)) no-repeat left center / 100% 80%;
          animation:mark .45s ease-out both;animation-delay:calc(var(--base,0ms) + var(--d,0ms));}
@keyframes mark{from{background-size:0% 80%;color:transparent;}to{background-size:100% 80%;color:var(--ink);}}
@media (prefers-reduced-motion:reduce){.sdk .c.s{animation:none;}}

/* ===== 5. PROGRESS STEPPER ===== */
.steps{display:flex;align-items:center;margin:4px 4px 6px;}
.step{display:flex;align-items:center;gap:10px;font:600 15px var(--body);color:var(--muted);white-space:nowrap;}
.step .n{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:#fff;border:2px solid var(--line);font:700 13px var(--display);}
.step.active{color:var(--ink);}
.step.active .n{background:var(--ink);border-color:var(--ink);color:#fff;}
.step.done{color:var(--ink);}
.step.done .n{background:var(--mark);border-color:var(--mark);color:var(--ink);}
.step-line{flex:1;height:2px;background:var(--line);margin:0 16px;min-width:18px;}
.step-line.done{background:var(--mark);}
@media (max-width:640px){.step .t{display:none;}.step.active .t{display:inline;}}

/* ===== 6. CARDS (each keyed st.container becomes a card) ===== */
.st-key-upload_card, .st-key-analysis_card, .st-key-board_card{
  background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:1.6rem 1.7rem;
  box-shadow:0 1px 0 rgba(19,26,58,.04),0 18px 44px rgba(19,26,58,.07);}
.card-head{display:flex;justify-content:space-between;align-items:center;gap:12px;}
.card-title{font:700 23px/1.2 var(--display);letter-spacing:-.015em;color:var(--ink);}
.card-sub{font:400 15px/1.5 var(--body);color:var(--muted);margin:5px 0 4px;}
.pill{font:600 13px var(--body);background:var(--mark-soft);color:#5A4300;border-radius:999px;padding:5px 12px;white-space:nowrap;}
.legend{display:flex;gap:18px;justify-content:center;flex-wrap:wrap;font:500 13px var(--body);color:var(--muted);margin-top:14px;}
.legend i{display:inline-block;width:14px;height:14px;border-radius:4px;vertical-align:-2px;margin-right:7px;background:#fff;border:2px solid var(--ink);}
.legend i.m{background:var(--mark);border-color:var(--mark);}

/* ===== 7. NATIVE STREAMLIT WIDGETS ===== */
/* buttons */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"]{
  border-radius:14px;min-height:3rem;padding:.6rem 1.3rem;font:700 15px var(--body);
  transition:transform .12s ease, box-shadow .12s ease, background .12s ease;}
[data-testid="stBaseButton-primary"]{background:var(--mark);color:var(--ink);border:2px solid var(--ink);box-shadow:0 4px 0 var(--ink);}
[data-testid="stBaseButton-primary"] p{color:var(--ink);font-weight:700;}
[data-testid="stBaseButton-primary"]:hover:not(:disabled){background:#FFDB5E;transform:translateY(-1px);box-shadow:0 5px 0 var(--ink);}
[data-testid="stBaseButton-primary"]:active:not(:disabled){transform:translateY(3px);box-shadow:0 1px 0 var(--ink);}
[data-testid="stBaseButton-primary"]:disabled{background:#EEF0F7;color:#9AA1BC;border-color:#D5D9E8;box-shadow:none;}
[data-testid="stBaseButton-secondary"]{background:#fff;color:var(--ink);border:2px solid var(--line);}
[data-testid="stBaseButton-secondary"]:hover{border-color:var(--ink);color:var(--ink);background:#fff;}
/* upload box */
[data-testid="stFileUploaderDropzone"]{background:var(--paper);border:2px dashed #B7BEDA;border-radius:18px;padding:2.4rem 1rem;transition:border-color .15s, background .15s;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--ink);background:#FAFAFD;}
/* images, grid editor, alerts */
[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--line);}
[data-testid="stImageCaption"]{font:500 13px var(--body);color:var(--muted);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:14px;}

/* ===== 8. TIPS + FOOTER ===== */
.tips{margin:34px 4px 8px;}
.tips-title{font:700 24px var(--display);letter-spacing:-.015em;color:var(--ink);margin-bottom:16px;}
.tips-grid{display:grid;grid-template-columns:repeat(3,1fr);}
.tip{padding:2px 28px;border-left:2px solid var(--line);}
.tip:first-child{padding-left:0;border-left:0;}
.tip b{display:block;font:700 16px var(--display);color:var(--ink);margin-bottom:6px;}
.tip span{display:block;font:400 14.5px/1.6 var(--body);color:var(--muted);}
.foot{margin:34px 4px 0;padding-top:18px;border-top:1px solid var(--line);font:500 13px var(--body);color:var(--muted);}

/* ===== 9. PHONES / TABLETS ===== */
@media (max-width:860px){
  .hero{grid-template-columns:1fr;gap:34px;padding:34px 26px;}
  .tips-grid{grid-template-columns:1fr;gap:18px;}
  .tip{padding:0;border-left:0;}
  .st-key-upload_card, .st-key-analysis_card, .st-key-board_card{padding:1.2rem 1.1rem;}
}
</style>
"""

SAMPLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


# ----------------------------------------------------------------------------
# UI helpers (HTML snippets). Keep these strings free of blank lines/indentation
# so Streamlit's markdown renderer treats them as raw HTML.
# ----------------------------------------------------------------------------
def board_html(original, solved=None, base_ms=0):
    """Render a 9x9 grid. Clues are bold; solver answers get the highlighter animation."""
    cells, k = [], 0
    for r in range(9):
        for c in range(9):
            cls, style, text = "c", "", ""
            if c in (2, 5):
                cls += " br"
            if r in (2, 5):
                cls += " bb"
            if original[r][c]:
                cls += " g"
                text = str(original[r][c])
            elif solved:
                cls += " s"
                text = str(solved[r][c])
                style = f' style="--d:{k * 30}ms"'
                k += 1
            cells.append(f'<div class="{cls}"{style}>{text}</div>')
    return (f'<div class="sdk" style="--base:{base_ms}ms" role="img" aria-label="Sudoku grid">'
            + "".join(cells) + "</div>")


@st.cache_data
def hero_markup():
    solved = [row[:] for row in SAMPLE]
    solve_sudoku(solved)
    logo = "".join(f'<span class="{"on" if i in (0, 4, 8) else ""}"></span>' for i in range(9))
    points = "".join(f'<div class="hero-point">{t}</div>' for t in (
        "Fix any misread digit before you solve",
        "Handles photos taken at an angle",
        "Solves even very hard puzzles in about a second",
    ))
    return (
        f'<div class="topbar"><div class="logo">{logo}</div><div class="brand-name">Sudoku Studio</div></div>'
        '<div class="hero"><div class="hero-copy">'
        '<div class="hero-title" role="heading" aria-level="1">Photograph a puzzle.<br>Get the solved grid.</div>'
        '<div class="hero-lede">Upload a photo or screenshot of any Sudoku. We find the grid and read the clues, '
        'you check them, and the solver fills in everything else.</div>'
        f'<div class="hero-points">{points}</div></div>'
        f'<div class="hero-board">{board_html(SAMPLE, solved, base_ms=700)}</div></div>'
    )


def stepper_html(step, all_done=False):
    labels = ["Upload photo", "Check digits", "Get solution"]
    parts = []
    for i, label in enumerate(labels, start=1):
        state = "done" if (i < step or all_done) else ("active" if i == step else "")
        mark = "✓" if state == "done" else str(i)
        parts.append(f'<div class="step {state}"><span class="n">{mark}</span><span class="t">{label}</span></div>')
        if i < 3:
            parts.append(f'<div class="step-line {"done" if (i < step or all_done) else ""}"></div>')
    return '<div class="steps">' + "".join(parts) + "</div>"


def card_head(title, sub, pill=None):
    pill_html = f'<div class="pill">{pill}</div>' if pill else ""
    return (f'<div class="card-head"><div class="card-title">{title}</div>{pill_html}</div>'
            f'<div class="card-sub">{sub}</div>')


TIPS_HTML = (
    '<div class="tips"><div class="tips-title">For the cleanest read</div><div class="tips-grid">'
    '<div class="tip"><b>Keep all four edges in view</b><span>Crop out the rest of the page so the grid fills most of the photo.</span></div>'
    '<div class="tip"><b>Light it evenly</b><span>Avoid shadows and glare across the grid. Faint or blurry digits are the most common cause of misreads.</span></div>'
    '<div class="tip"><b>Hold the camera straight on</b><span>Small angles are corrected for you. Heavy tilt or curved paper makes digits harder to read.</span></div>'
    '</div></div>'
    '<div class="foot">Sudoku Studio. Built with Streamlit, OpenCV and Tesseract OCR.</div>'
)


def board_to_df(board):
    return pd.DataFrame(board, columns=[str(i) for i in range(1, 10)], dtype="int64")


def reset_workspace():
    for key in ("board", "original", "annotated", "warped", "found", "solved"):
        st.session_state.pop(key, None)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(hero_markup(), unsafe_allow_html=True)

if "editor_version" not in st.session_state:
    st.session_state["editor_version"] = 0

has_board = st.session_state.get("board") is not None
is_solved = bool(st.session_state.get("solved"))
step = 3 if is_solved else (2 if has_board else 1)
st.markdown(stepper_html(step, all_done=is_solved), unsafe_allow_html=True)

# ---- Step 1: upload ---------------------------------------------------------
if not has_board:
    with st.container(key="upload_card"):
        st.markdown(card_head("Upload your puzzle",
                              "PNG, JPG or WEBP, up to 12 MB. A sharp, well-lit, straight-on shot works best."),
                    unsafe_allow_html=True)
        upload = st.file_uploader("Sudoku image", type=ALLOWED_TYPES, label_visibility="collapsed")
        if upload is not None:
            st.image(upload, caption="Ready to detect", width=260)
        if st.button("Detect puzzle", type="primary", disabled=upload is None):
            if not tesseract_available():
                st.error("Tesseract OCR could not be started. Make sure `packages.txt` "
                         "contains `tesseract-ocr` (or install Tesseract locally).")
            else:
                with st.spinner("Finding the grid and reading digits…"):
                    try:
                        board, annotated, warped, found = process_image(upload)
                    except Exception as exc:
                        st.error(f"Could not process this image: {exc}")
                    else:
                        st.session_state.update(
                            board=board, original=[row[:] for row in board],
                            annotated=annotated, warped=warped, found=found,
                        )
                        st.session_state.pop("solved", None)
                        st.rerun()

# ---- Steps 2 & 3: review + solve -------------------------------------------
else:
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(key="analysis_card"):
            st.markdown(card_head("What we found",
                                  "The yellow outline marks the grid we detected, shown straightened on the right."),
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.image(st.session_state["annotated"], caption="Your photo", width="stretch")
            c2.image(st.session_state["warped"], caption="Straightened grid", width="stretch")
            if not st.session_state.get("found", True):
                st.warning("We couldn't find the grid outline, so the whole image was used. "
                           "Cropping tightly around the puzzle usually fixes this.")

    with right:
        with st.container(key="board_card"):
            solved = st.session_state.get("solved")
            if solved:
                filled = sum(1 for r in range(9) for c in range(9)
                             if st.session_state["original"][r][c] == 0)
                st.markdown(card_head("Solved", "Every row, column and box now holds 1 to 9.",
                                      pill=f"{filled} cells filled"), unsafe_allow_html=True)
                st.markdown(board_html(st.session_state["original"], solved, base_ms=150)
                            + '<div class="legend"><span><i></i>Your clues</span>'
                              '<span><i class="m"></i>Solver answers</span></div>',
                            unsafe_allow_html=True)
                text = "\n".join(" ".join(str(v) for v in row) for row in solved)
                b1, b2 = st.columns(2)
                b1.download_button("Download solution", text, "sudoku_solution.txt", width="stretch")
                if b2.button("Solve another puzzle", type="primary", width="stretch"):
                    reset_workspace()
                    st.rerun()
            else:
                clues = sum(1 for row in st.session_state["board"] for v in row if v)
                st.markdown(card_head("Check the digits",
                                      "Double-click any cell to change it. Use 0 for an empty cell.",
                                      pill=f"{clues} clues found"), unsafe_allow_html=True)
                edited = st.data_editor(
                    board_to_df(st.session_state["board"]),
                    key=f"editor_{st.session_state['editor_version']}",
                    hide_index=True,
                    width="stretch",
                    column_config={
                        str(i): st.column_config.NumberColumn(
                            str(i), min_value=0, max_value=9, step=1, format="%d", width="small"
                        ) for i in range(1, 10)
                    },
                )
                b1, b2 = st.columns(2)
                solve_clicked = b1.button("Solve Sudoku", type="primary", width="stretch")
                if b2.button("Start over", width="stretch"):
                    reset_workspace()
                    st.rerun()

                if solve_clicked:
                    try:
                        grid = edited.fillna(0).astype(int).values.tolist()
                        if any(not 0 <= v <= 9 for row in grid for v in row):
                            raise ValueError("Each cell must be a number from 0 to 9.")
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["board"] = grid
                        st.session_state["original"] = [row[:] for row in grid]
                        ok, pos = board_is_valid(grid)
                        if not ok:
                            st.error(f"Row {pos[0] + 1}, column {pos[1] + 1} conflicts with another clue "
                                     "in its row, column or box. Correct it and solve again.")
                        else:
                            candidate = [row[:] for row in grid]
                            with st.spinner("Solving…"):
                                found_solution = solve_sudoku(candidate)
                            if found_solution:
                                st.session_state["solved"] = candidate
                                st.rerun()
                            else:
                                st.error("No solution exists for these clues. Check the digits against "
                                         "your photo for a misread number, then solve again.")

st.markdown(TIPS_HTML, unsafe_allow_html=True)
