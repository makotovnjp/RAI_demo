"""
Blog thumbnail generator — RAI × UFACTORY Lite 6
Output: blog_thumbnail.png  (1200 × 630 px)
"""
import os, sys, struct
import numpy as np
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import urllib.request

matplotlib.rcParams["font.sans-serif"] = (
    ["Yu Gothic", "Meiryo", "MS Gothic"] +
    matplotlib.rcParams.get("font.sans-serif", [])
)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 定数 (generate_demo_video.py と同じパラメータ) ─────────────────────────
PI = np.pi
CACHE_DIR = "stl_cache"

_LIGHT = np.array([1.2, 0.8, 2.0], dtype=float)
LIGHT_DIR = _LIGHT / np.linalg.norm(_LIGHT)
_FILL  = np.array([-0.5, -0.4, 0.8], dtype=float)
FILL_DIR  = _FILL  / np.linalg.norm(_FILL)

LINK_BASE_COLORS_RGB = [
    [0.94, 0.94, 0.95],
    [0.80, 0.82, 0.87],
    [0.94, 0.94, 0.95],
    [0.80, 0.82, 0.87],
    [0.94, 0.94, 0.95],
    [0.80, 0.82, 0.87],
    [0.68, 0.74, 0.85],
]

URDF_JOINTS = [
    ("revolute", [0.0,    0.0,     0.2435], [0.0,    0.0,  0.0 ], [0,0,1]),
    ("revolute", [0.0,    0.0,     0.0   ], [PI/2,  -PI/2, PI  ], [0,0,1]),
    ("revolute", [0.2002, 0.0,     0.0   ], [-PI,    0.0,  PI/2], [0,0,1]),
    ("revolute", [0.087, -0.22761, 0.0   ], [PI/2,   0.0,  0.0 ], [0,0,1]),
    ("revolute", [0.0,    0.0,     0.0   ], [PI/2,   0.0,  0.0 ], [0,0,1]),
    ("revolute", [0.0,    0.0625,  0.0   ], [-PI/2,  0.0,  0.0 ], [0,0,1]),
    ("fixed",    [0.0,    0.0,     0.0   ], [0.0,    0.0,  0.0 ], [0,0,1]),
]

MESH_NAMES = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
_MESH_BASE = (
    "https://raw.githubusercontent.com/xArm-Developer"
    "/xarm_ros2/master/xarm_description/meshes/lite6/visual"
)
MESH_URLS  = {n: f"{_MESH_BASE}/{n}.stl" for n in MESH_NAMES}
DECIMATE   = 5   # 静止画は高品質

_STL_DTYPE = np.dtype([
    ('normal', '<f4', 3), ('v0', '<f4', 3),
    ('v1',     '<f4', 3), ('v2', '<f4', 3), ('attr', '<u2'),
])

# ── メッシュ読み込み ────────────────────────────────────────────────────────

def download_stl(name: str) -> bytes:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.stl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    print(f"  Downloading {name}.stl ...")
    with urllib.request.urlopen(MESH_URLS[name], timeout=20) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return data


def parse_stl(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    num     = struct.unpack_from("<I", data, 80)[0]
    records = np.frombuffer(data, dtype=_STL_DTYPE, offset=84, count=num)
    tris    = np.stack([records['v0'], records['v1'], records['v2']], axis=1).copy()
    normals = records['normal'].copy()
    return tris, normals


def compute_face_colors(tris_mm: np.ndarray, normals_local: np.ndarray,
                        base_rgb: list) -> np.ndarray:
    n = normals_local.astype(np.float64)
    norms = np.linalg.norm(n, axis=1)
    bad = norms < 1e-10
    if bad.any():
        v0, v1, v2 = tris_mm[bad,0], tris_mm[bad,1], tris_mm[bad,2]
        n[bad] = np.cross(v1-v0, v2-v0)
        norms[bad] = np.linalg.norm(n[bad], axis=1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    n /= norms[:, None]
    d_key  = np.abs(n @ LIGHT_DIR)
    d_fill = np.abs(n @ FILL_DIR)
    _view  = np.array([0.4, 0.1, 1.0]); _view /= np.linalg.norm(_view)
    _half  = LIGHT_DIR + _view;          _half /= np.linalg.norm(_half)
    spec   = np.clip(n @ _half, 0, 1) ** 32
    bright = np.clip(0.28 + 0.48*d_key + 0.12*d_fill + 0.20*spec, 0, 1)
    bc     = np.array(base_rgb, dtype=float)
    colors = np.clip(bright[:, None] * bc, 0, 1)
    alpha  = np.full((len(colors), 1), 0.97)
    return np.hstack([colors, alpha])


def load_meshes() -> Dict[str, tuple]:
    meshes = {}
    for idx, name in enumerate(MESH_NAMES):
        raw = download_stl(name)
        tris, normals = parse_stl(raw)
        td = tris[::DECIMATE]; nd = normals[::DECIMATE]
        tris_mm = td * 1000.0
        meshes[name] = (tris_mm, compute_face_colors(tris_mm, nd, LINK_BASE_COLORS_RGB[idx]))
        print(f"  {name}: {len(td)} tris")
    return meshes


# ── Forward Kinematics ─────────────────────────────────────────────────────

def rpy_to_R(rpy: list) -> np.ndarray:
    r,p,y = rpy
    Rx=np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry=np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz@Ry@Rx


def axis_angle_R(axis: list, angle: float) -> np.ndarray:
    axis=np.asarray(axis,float); nm=np.linalg.norm(axis)
    if nm<1e-10: return np.eye(3)
    axis/=nm
    K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def make_T(xyz: list, R: np.ndarray = None, rpy: list = None) -> np.ndarray:
    T=np.eye(4); T[:3,3]=xyz
    if R is not None: T[:3,:3]=R
    elif rpy is not None: T[:3,:3]=rpy_to_R(rpy)
    return T


def forward_kinematics(joint_angles: List[float]) -> List[np.ndarray]:
    transforms=[np.eye(4)]; T=np.eye(4); ai=0
    for jtype,xyz,rpy,axis in URDF_JOINTS:
        T=T@make_T(xyz,rpy=rpy)
        if jtype=="revolute" and ai<len(joint_angles):
            T=T@make_T([0,0,0],R=axis_angle_R(axis,joint_angles[ai])); ai+=1
        transforms.append(T.copy())
    return transforms


def transform_mesh(tris_mm: np.ndarray, T_m: np.ndarray) -> np.ndarray:
    N=len(tris_mm)
    flat=tris_mm.reshape(-1,3)/1000.0
    homog=np.c_[flat,np.ones(len(flat))]
    world=(T_m@homog.T).T[:,:3]*1000.0
    return world.reshape(N,3,3)


# ── サムネイル生成 ─────────────────────────────────────────────────────────

def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "blog_thumbnail.png")

    print("Loading meshes ...")
    meshes = load_meshes()

    # ── ポーズ: 物体を把持してドロップゾーンへ移送中 ──────────────────────
    pose = [0.55, -0.18, 1.12, 0.0, 0.38, 0.25]  # q1=0.55 で右側に旋回
    Ts   = forward_kinematics(pose)
    tcp  = Ts[-1][:3, 3] * 1000.0

    # ── Figure: 1200×630 ─────────────────────────────────────────────────
    BG = "#07101e"
    fig = plt.figure(figsize=(12, 6.3), facecolor=BG)

    # ─── Left: 3D robot (54% width) ──────────────────────────────────────
    ax = fig.add_axes([0.0, 0.0, 0.54, 1.0], projection="3d")
    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(BG)
        pane.set_edgecolor(BG)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim(-180, 420)
    ax.set_ylim(-320, 380)
    ax.set_zlim(0, 680)
    ax.view_init(elev=18, azim=-38)

    # 作業台面
    _surf_z = 302.0
    ax.plot_surface(
        np.array([[-80, 410], [-80, 410]]),
        np.array([[-290, -290], [350, 350]]),
        np.full((2,2), _surf_z),
        alpha=0.13, color="#2255aa", zorder=0)
    tx = np.array([-60, 390, 390, -60, -60])
    ty = np.array([-270, -270, 330, 330, -270])
    ax.plot(tx, ty, [_surf_z]*5, color="#3366bb", lw=0.9, alpha=0.55)

    # ベースの光輪
    theta = np.linspace(0, 2*PI, 80)
    for r, a in [(90, 0.18), (140, 0.09), (190, 0.05)]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), [2]*80,
                color="#4488ff", lw=0.7, alpha=a)

    # メッシュ描画
    for i, name in enumerate(MESH_NAMES):
        tris_mm, face_colors = meshes[name]
        tris_w = transform_mesh(tris_mm, Ts[i])
        coll = Poly3DCollection(tris_w, linewidth=0.10)
        coll.set_facecolor(face_colors)
        coll.set_edgecolor("#0e1828")
        ax.add_collection3d(coll)

    # ジョイントマーカー (オレンジ輝点)
    for T in Ts[1:7]:
        pt = T[:3, 3] * 1000
        ax.scatter(*pt, c="#f5a623", s=90, zorder=10, depthshade=False, alpha=0.95)

    # グリッパー (把持中: 半開き)
    T_eef = Ts[-1]
    origin = T_eef[:3, 3] * 1000.0
    z_ax   = T_eef[:3, 2]
    y_ax   = T_eef[:3, 1]
    for sign in (1, -1):
        base = origin + y_ax * 10 * sign
        tip  = base   + z_ax * 30
        ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
                color="#ff5533", lw=5, solid_capstyle="round", zorder=11)

    # 把持中オブジェクト (赤いブロック)
    ax.scatter(*tcp, c="#ff3344", s=550, marker="s", zorder=12,
               depthshade=False, edgecolors="#ffaaaa", linewidths=1.5)

    # TCP 軌跡 (ドット)
    ax.scatter(*tcp, c="#4488ff", s=30, alpha=0.4, depthshade=False)

    # ─── Right: text panel (46% width) ───────────────────────────────────
    ax_t = fig.add_axes([0.53, 0.0, 0.47, 1.0])
    ax_t.set_facecolor(BG)
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")

    # 縦区切り線
    ax_t.plot([0.04, 0.04], [0.06, 0.94], color="#1a3a6a", lw=1.8, alpha=0.7)

    # ── メインタイトル ─────────────────────────────────────────────────
    ax_t.text(0.10, 0.935, "RAI",
              ha="left", va="top", color="#e94560",
              fontsize=42, fontweight="bold", fontfamily="monospace")
    ax_t.text(0.10, 0.840, "x UFACTORY Lite 6",
              ha="left", va="top", color="#ffffff",
              fontsize=18, fontweight="bold", fontfamily="monospace")

    # サブタイトル
    ax_t.text(0.10, 0.760, "Natural Language Robot Control",
              ha="left", va="top", color="#88ccff",
              fontsize=11, fontfamily="monospace")
    ax_t.text(0.10, 0.700, "ROS2  x  LLM  x  Physical AI",
              ha="left", va="top", color="#5588bb",
              fontsize=9.5, fontfamily="monospace")

    ax_t.plot([0.10, 0.97], [0.668, 0.668], color="#1a3a6a", lw=0.9)

    # ── ミニターミナル ─────────────────────────────────────────────────
    ax_t.text(0.10, 0.654, "[User]",
              ha="left", va="top", color="#55aaff",
              fontsize=8.5, fontweight="bold", fontfamily="monospace")
    ax_t.text(0.10, 0.625,
              'Pick the red block and place\nit on the right side.',
              ha="left", va="top", color="#88ccff",
              fontsize=8.5, fontfamily="monospace", linespacing=1.5,
              bbox=dict(boxstyle="square,pad=0.28", facecolor="#0a1e3a",
                        edgecolor="#1a3a6a", linewidth=1))

    ax_t.text(0.10, 0.535, "[Agent Thinking]",
              ha="left", va="top", color="#ffd700",
              fontsize=8.5, fontweight="bold", fontfamily="monospace")
    ax_t.text(0.10, 0.506,
              'Plan: approach -> descend\n'
              '      -> grasp -> lift\n'
              '      -> transport -> place',
              ha="left", va="top", color="#ffffff",
              fontsize=8, fontfamily="monospace", linespacing=1.55,
              bbox=dict(boxstyle="square,pad=0.28", facecolor="#0c1a2e",
                        edgecolor="#2a4a1a", linewidth=1))

    ax_t.text(0.10, 0.395, "[Tool ->]  move_to_pose",
              ha="left", va="top", color="#ff9944",
              fontsize=8, fontweight="bold", fontfamily="monospace")
    ax_t.text(0.10, 0.366,
              '{"x":293,"y":0,"z":322}\n'
              '-> TCP: (293, 0, 322) mm  OK',
              ha="left", va="top", color="#aaffaa",
              fontsize=8, fontfamily="monospace", linespacing=1.55,
              bbox=dict(boxstyle="square,pad=0.28", facecolor="#0c1a0c",
                        edgecolor="#1a4a1a", linewidth=1))

    ax_t.plot([0.10, 0.97], [0.295, 0.295], color="#1a3a6a", lw=0.9)

    # ── フローダイアグラム ─────────────────────────────────────────────
    flow = [
        (0.14, "#55aaff",  "User"),
        (0.35, "#ffd700",  "LLM"),
        (0.57, "#ff9944",  "Tool"),
        (0.78, "#ff5533",  "Robot"),
    ]
    for x, color, label in flow:
        ax_t.text(x, 0.228, label, ha="center", va="center",
                  color=color, fontsize=9, fontweight="bold",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.38", facecolor="#0d1f3c",
                            edgecolor=color, linewidth=1.6))
    for xs, xe in [(0.22, 0.30), (0.43, 0.51), (0.65, 0.72)]:
        ax_t.annotate("", xy=(xe, 0.228), xytext=(xs, 0.228),
                       arrowprops=dict(arrowstyle="->", color="#3355aa", lw=1.3))

    # ── バッジ ─────────────────────────────────────────────────────────
    badges = [
        (0.16, "#102040", "#4488ff", "ROS2"),
        (0.42, "#1a0818", "#e94560", "Claude API"),
        (0.72, "#0a1a10", "#44bb88", "Physical AI"),
    ]
    for x, bg, fg, label in badges:
        ax_t.text(x, 0.125, label, ha="center", va="center",
                  color=fg, fontsize=9, fontweight="bold",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.42", facecolor=bg,
                            edgecolor=fg, linewidth=1.6))

    # GitHub リンク
    ax_t.text(0.52, 0.048,
              "github.com/makotovnjp/RAI_demo",
              ha="center", va="center",
              color="#445566", fontsize=7.5, fontfamily="monospace")

    # 保存
    plt.savefig(out_path, dpi=100, facecolor=BG, bbox_inches=None, pad_inches=0)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved: {out_path}  ({size_kb:.0f} KB, 1200x630)")
    plt.close(fig)


if __name__ == "__main__":
    main()
