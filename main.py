import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageTk

RED, BLUE, WHITE, BLACK = (0, 0, 255), (255, 0, 0), (255, 255, 255), (0, 0, 0)
IMAGE_TYPES = [("이미지 파일", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("모든 파일", "*.*")]

# 슬라이더: 키, 라벨, 기본값, 최소, 최대, 간격
SETTINGS = [
    ("sharp_min",    "윤곽 선명도",  13.0,  0,  30, 0.5),
    ("contrast_min", "배경 대비",    60,    0, 255, 1),
    ("r_max",        "최대 반지름",  12.5,  8,  17, 0.5),
    ("dark_threshold", "내부 어두운 영역 기준(%)", 8, 0, 40, 1),
    ("zoom",         "확대 배율",    0.9,   0.5, 3.0, 0.1),
]
DEFAULTS = {key: default for key, _, default, *_ in SETTINGS}

HOUGH_R_MIN, HOUGH_R_MAX = 4, 17     # 작은 원도 후보에 포함
MAX_EDGE_POS = 12                    # 윤곽 위치가 중심에서 이보다 멀면 제외

class Circle(NamedTuple):
    x: float
    y: float
    r: float
    empty: bool      # 중심부가 주변보다 어두우면 Empty


# ----------------------------------------------------------------------
# 1. 검출: 이미지당 1번 후보의 특징값을 계산(analyze) -> 슬라이더 값으로 즉시 걸러냄(select)
# ----------------------------------------------------------------------
def local_dist(shape, cx, cy, m):
    """(cx, cy) 주변 ±m 픽셀 영역의 슬라이스와, 그 영역 각 픽셀의 중심까지 거리."""
    h, w = shape
    ys = slice(int(max(cy - m, 0)), int(min(cy + m + 1, h)))
    xs = slice(int(max(cx - m, 0)), int(min(cx + m + 1, w)))
    yy, xx = np.ogrid[ys, xs]
    return (ys, xs), np.hypot(xx - cx, yy - cy)


def edge_features(gray_f, cx, cy, r):
    """윤곽 선명도(sharp), 배경 대비(contrast), 윤곽 위치(k, px)."""
    n = int(r) + 12
    (ys, xs), d = local_dist(gray_f.shape, cx, cy, r + 12)
    idx = d.astype(int).ravel()                          # 중심으로부터 1px 간격 고리 번호
    total = np.bincount(idx, gray_f[ys, xs].ravel(), minlength=n)[:n]
    count = np.bincount(idx, minlength=n)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = total / count                          # 반지름 방향 평균 밝기
    slope = np.diff(profile)
    k = int(np.nanargmin(slope[5:])) + 5                 # 가장 급격히 어두워지는 지점 = 윤곽
    inside = profile[:k]
    outside = profile[k + 3:k + 8]
    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]
    # 이미지 가장자리 등으로 바깥쪽 샘플이 부족하면 후보 필터에서 제외
    contrast = (float(inside.max() - outside.mean())
                if inside.size and outside.size else float("-inf"))
    return -slope[k], contrast, k


def inner_dark_score(blur, cx, cy, r):
    """원 내부에서 주변 고리보다 충분히 어두운 픽셀의 비율(%)을 계산한다.

    픽셀 하나의 최솟값만 보지 않고, 내부 영역 어디에 있든 주변보다
    25/255 이상 어두운 픽셀이 차지하는 비율을 센다. 별도의 점 모양 검사는 없다.
    """
    (ys, xs), d = local_dist(blur.shape, cx, cy, r)
    patch = blur[ys, xs]
    interior = patch[d < 0.6 * r]
    shell = patch[(d >= 0.65 * r) & (d < 0.95 * r)]
    if interior.size == 0 or shell.size == 0:
        return 0.0

    local_brightness = float(np.median(shell))
    dark_pixels = interior < (local_brightness - 25.0)
    return float(np.mean(dark_pixels) * 100.0)


def analyze(gray, blur):
    """허프 변환으로 후보 원을 찾고 후보별 특징값 [x, y, r, sharp, contrast, k, depth] 계산."""
    found = cv2.HoughCircles(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.HOUGH_GRADIENT,
                             dp=1, minDist=8, param1=60, param2=11,
                             minRadius=HOUGH_R_MIN, maxRadius=HOUGH_R_MAX)
    gray_f = gray.astype(np.float32)
    rows = []
    for x, y, r in (found[0] if found is not None else []):
        sharp, contrast, k = edge_features(gray_f, x, y, r)
        rows.append((x, y, r, sharp, contrast, k, inner_dark_score(blur, x, y, r)))
    return np.array(rows, dtype=float).reshape(-1, 7)


def select(cands, sharp_min, contrast_min, r_max, dark_threshold):
    ok = ((cands[:, 2] <= r_max) & (cands[:, 3] >= sharp_min)
          & (cands[:, 4] >= contrast_min) & (cands[:, 5] <= MAX_EDGE_POS))
    return [Circle(float(x), float(y), float(r), bool(score >= dark_threshold))
            for x, y, r, *_, score in cands[ok]]


# ----------------------------------------------------------------------
# 2. 그리기 / 파일 입출력
# ----------------------------------------------------------------------
def zoomed(gray, zoom):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img if zoom == 1 else cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)


def draw_circles(img, circles, zoom=1.0):
    for c in circles:
        cv2.circle(img, (round(c.x * zoom), round(c.y * zoom)), round(c.r * zoom),
                   BLUE if c.empty else RED, 1, cv2.LINE_AA)
    return img


def banner(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), BLACK, -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)


def read_gray(path):
    """한글 경로에서도 읽히도록 cv2.imread 대신 imdecode 사용."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


# ----------------------------------------------------------------------
# 3. GUI
# ----------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("Circle Counter")
        self.geometry("1200x780")
        self.circles, self.history = [], []

        # 이미지 표시 영역 (스크롤 가능)
        viewer = tk.Frame(self)
        viewer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(viewer, bg="#303030")
        hbar = tk.Scrollbar(viewer, orient="horizontal", command=self.canvas.xview)
        vbar = tk.Scrollbar(viewer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        viewer.grid_rowconfigure(0, weight=1)
        viewer.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", lambda e: self.on_click(e, right=False))
        self.canvas.bind("<Button-3>", lambda e: self.on_click(e, right=True))

        # 옵션 슬라이더
        options = tk.LabelFrame(self, text="검출 옵션", padx=8, pady=5)
        options.pack(fill="x", padx=8, pady=5)
        self.vars = {}
        for key, label, default, lo, hi, step in SETTINGS:
            self.vars[key] = tk.DoubleVar(value=default)
            scale = tk.Scale(options, label=label, variable=self.vars[key], from_=lo, to=hi,
                             resolution=step, orient="horizontal", length=165)
            scale.pack(side="left", fill="x", expand=True, padx=3)
            if key == "zoom":                       # 배율은 움직이는 즉시 화면 갱신
                scale.configure(command=lambda _v: self.render())

        self.count_label = tk.Label(self, anchor="w")
        self.count_label.pack(fill="x", padx=10)

        # 버튼
        buttons = tk.Frame(self, padx=8, pady=8)
        buttons.pack(fill="x")
        for text, command, width in [("재검출", self.redetect, 12), ("이미지 선택", self.open_image, 12), ("초기화", self.reset, 12)]:
            tk.Button(buttons, text=text, command=command, width=width).pack(side="left", padx=4)
        tk.Button(buttons, text="종료", command=self.destroy, width=12).pack(side="right", padx=4)

    # ----- 이미지 / 검출 -----
    def open_image(self):
        path = filedialog.askopenfilename(parent=self, title="이미지 선택", filetypes=IMAGE_TYPES)
        if not path:
            return False
        gray = read_gray(path)
        if gray is None:
            messagebox.showerror("오류", f"이미지를 읽을 수 없습니다:\n{path}")
            return False
        self.path, self.gray = path, gray
        self.blur = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
        self.cands = analyze(gray, self.blur)        # 무거운 계산은 이미지당 한 번만
        self.title(f"Circle Counter - {os.path.basename(path)}")
        self.redetect()
        return True

    def redetect(self):
        """현재 슬라이더 값으로 후보를 다시 거르고, 수동 편집 내역은 버린다."""
        params = {k: v.get() for k, v in self.vars.items() if k != "zoom"}
        self.circles = select(self.cands, **params)
        self.dark_threshold = params["dark_threshold"]
        self.default_r = float(np.median([c.r for c in self.circles])) if self.circles else 10.0
        self.history.clear()
        self.render()

    def reset(self):
        for key, var in self.vars.items():
            var.set(DEFAULTS[key])
        self.redetect()

    # ----- 편집 -----
    def find(self, x, y):
        """(x, y)를 포함하는 가장 가까운 원의 인덱스 (없으면 None)."""
        hits = [(d, i) for i, c in enumerate(self.circles)
                if (d := math.hypot(c.x - x, c.y - y)) <= max(c.r, 8)]
        return min(hits)[1] if hits else None

    def edit(self, x, y, right):
        i = self.find(x, y)
        if right and i is None:
            return False
        self.history.append(self.circles.copy())
        if right:                                    # 오른쪽 클릭: 점 있음/없음 전환
            self.circles[i] = self.circles[i]._replace(empty=not self.circles[i].empty)
        elif i is not None:                          # 왼쪽 클릭, 기존 원 위: 삭제
            del self.circles[i]
        else:                                        # 왼쪽 클릭, 빈 곳: 추가
            score = inner_dark_score(self.blur, x, y, self.default_r)
            self.circles.append(Circle(x, y, self.default_r, bool(score >= self.dark_threshold)))
        return True

    def on_click(self, event, right):
        zoom = self.vars["zoom"].get()
        h, w = self.gray.shape
        dw, dh = round(w * zoom), round(h * zoom)    # 화면에 표시되는 이미지 한 장의 크기
        pw = dw + 3                                  # 이미지 + 구분선(3px)
        px, py = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if not (0 <= py < dh and 0 <= px < 2 * pw - 3) or px % pw >= dw:
            return                                   # 이미지 밖이나 구분선은 무시
        # 원본/결과 어느 쪽을 눌러도 같은 이미지 좌표로 변환
        if self.edit((px % pw) / zoom, py / zoom, right):
            self.render()

    # ----- 표시 -----
    def summary(self):
        n_empty = sum(c.empty for c in self.circles)
        return f"Total: {len(self.circles)} | Full: {len(self.circles) - n_empty} | Empty: {n_empty}"

    def render(self):
        zoom = self.vars["zoom"].get()
        left = zoomed(self.gray, zoom)
        right = draw_circles(left.copy(), self.circles, zoom)
        banner(left, "Original")
        banner(right, self.summary())
        divider = np.full((left.shape[0], 3, 3), 255, np.uint8)
        pil = Image.fromarray(cv2.cvtColor(np.hstack([left, divider, right]), cv2.COLOR_BGR2RGB))

        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, pil.width, pil.height))
        self.count_label.config(text=self.summary())

def main():
    app = App()
    if app.open_image():
        app.deiconify()
        app.mainloop()
    else:
        app.destroy()


if __name__ == "__main__":
    main()