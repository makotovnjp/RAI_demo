"""
RAI × UFACTORY Lite 6 デモアニメーション生成スクリプト (URDF + STL メッシュ版)
===============================================================================
xarm_ros2 リポジトリから UFACTORY Lite 6 の URDF と STL メッシュを取得し、
実際のロボット形状でピック＆プレース動作を 3D アニメーション化して MP4 に保存します。

必要パッケージ:
    pip install matplotlib numpy

動画エンコーダ (要インストール):
    ffmpeg  https://ffmpeg.org/download.html
    winget install ffmpeg

実行:
    python generate_demo_video.py
    → rai_lite6_demo.mp4 が生成されます (初回のみ STL キャッシュに数秒かかります)
"""

import os, sys, struct, time, json
from typing import Dict, List, Tuple, Optional
import numpy as np
import urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.font_manager import FontProperties

# ── 日本語フォント設定 ──────────────────────────────────────────────────────
matplotlib.rcParams["font.sans-serif"] = (
    ["Yu Gothic", "Meiryo", "MS Gothic"] + matplotlib.rcParams.get("font.sans-serif", [])
)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 定数 — UFACTORY Lite 6 URDF ジョイントパラメータ (xarm_ros2 から取得)
# ══════════════════════════════════════════════════════════════════════════════

PI = np.pi

# xarm_ros2/xarm_description/urdf/lite6/lite6.urdf.xacro から抽出した関節定義
# (type, xyz[m], rpy[rad], local_axis)  ─ すべての回転軸はローカルZ
URDF_JOINTS = [
    # joint1: ベース回転 (Z軸)
    ("revolute", [0.0,     0.0,      0.2435 ], [0.0,    0.0,   0.0  ], [0, 0, 1]),
    # joint2: 肩 (フレーム変換あり)
    ("revolute", [0.0,     0.0,      0.0    ], [PI/2,  -PI/2,  PI   ], [0, 0, 1]),
    # joint3: 肘
    ("revolute", [0.2002,  0.0,      0.0    ], [-PI,    0.0,   PI/2 ], [0, 0, 1]),
    # joint4: 前腕
    ("revolute", [0.087,  -0.22761,  0.0    ], [PI/2,   0.0,   0.0  ], [0, 0, 1]),
    # joint5: 手首1
    ("revolute", [0.0,     0.0,      0.0    ], [PI/2,   0.0,   0.0  ], [0, 0, 1]),
    # joint6: 手首2
    ("revolute", [0.0,     0.0625,   0.0    ], [-PI/2,  0.0,   0.0  ], [0, 0, 1]),
    # joint_eef: フランジ (固定)
    ("fixed",    [0.0,     0.0,      0.0    ], [0.0,    0.0,   0.0  ], [0, 0, 1]),
]

# STL ダウンロード URL (xarm_ros2 GitHub)
_MESH_BASE = (
    "https://raw.githubusercontent.com/xArm-Developer"
    "/xarm_ros2/master/xarm_description/meshes/lite6/visual"
)
MESH_URLS = {
    name: f"{_MESH_BASE}/{name}.stl"
    for name in ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
}
CACHE_DIR = "stl_cache"
DECIMATE = 8    # 1/8 のフェースを残す → 約 430~1900 三角形/リンク

# --- 照明 (Phong: キーライト + フィルライト) ---
_LIGHT = np.array([1.2, 0.8, 2.0], dtype=float)
LIGHT_DIR = _LIGHT / np.linalg.norm(_LIGHT)   # キーライト (斜め上)
_FILL  = np.array([-0.5, -0.4, 0.8], dtype=float)
FILL_DIR  = _FILL  / np.linalg.norm(_FILL)    # フィルライト (逆方向)

# Lite 6 実機に近い白/シルバー系カラー
LINK_BASE_COLORS_RGB = [
    [0.94, 0.94, 0.95],  # base   — オフホワイト
    [0.80, 0.82, 0.87],  # link1  — シルバー
    [0.94, 0.94, 0.95],  # link2  — オフホワイト
    [0.80, 0.82, 0.87],  # link3  — シルバー
    [0.94, 0.94, 0.95],  # link4  — オフホワイト
    [0.80, 0.82, 0.87],  # link5  — シルバー
    [0.68, 0.74, 0.85],  # link6  — ブルーアクセント (EEF)
]


# ══════════════════════════════════════════════════════════════════════════════
# STL ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

def download_stl(name: str) -> bytes:
    """STL を GitHub から取得 (ローカルキャッシュあり)"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.stl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    url = MESH_URLS[name]
    print(f"  ダウンロード中: {name}.stl")
    with urllib.request.urlopen(url, timeout=20) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return data


# numpy 構造体型 (50 bytes/record): normal(12) + v0(12) + v1(12) + v2(12) + attr(2)
_STL_DTYPE = np.dtype([
    ('normal', '<f4', 3),
    ('v0',     '<f4', 3),
    ('v1',     '<f4', 3),
    ('v2',     '<f4', 3),
    ('attr',   '<u2'),
])

def parse_stl(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """バイナリ STL → (N,3,3) 頂点配列, (N,3) 法線配列 [m単位]"""
    num = struct.unpack_from("<I", data, 80)[0]
    records = np.frombuffer(data, dtype=_STL_DTYPE, offset=84, count=num)
    tris    = np.stack([records['v0'], records['v1'], records['v2']], axis=1).copy()
    normals = records['normal'].copy()
    return tris, normals


def decimate(tris: np.ndarray, factor: int) -> np.ndarray:
    """単純間引き (factor 個に 1 個を残す)"""
    return tris[::factor]


def compute_face_colors(tris_mm: np.ndarray, normals_local: np.ndarray,
                        base_rgb: list) -> np.ndarray:
    """
    Phong シェーディング (STL 格納法線 + フォールバック計算)
    戻り値: (N, 4) RGBA 配列
    """
    n = normals_local.astype(np.float64)
    norms = np.linalg.norm(n, axis=1)
    bad = norms < 1e-10
    if bad.any():                                         # 縮退フェースは頂点計算
        v0, v1, v2 = tris_mm[bad, 0], tris_mm[bad, 1], tris_mm[bad, 2]
        n[bad] = np.cross(v1 - v0, v2 - v0)
        norms[bad] = np.linalg.norm(n[bad], axis=1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    n = n / norms[:, None]                                # 単位化

    # 両面 Lambert (キー + フィル)
    d_key  = np.abs(n @ LIGHT_DIR)
    d_fill = np.abs(n @ FILL_DIR)

    # Blinn-Phong スペキュラー (キーライトのみ)
    _view = np.array([0.4, 0.1, 1.0]); _view /= np.linalg.norm(_view)
    _half = LIGHT_DIR + _view;          _half /= np.linalg.norm(_half)
    spec  = np.clip(n @ _half, 0, 1) ** 32

    brightness = np.clip(0.28 + 0.48 * d_key + 0.12 * d_fill + 0.20 * spec, 0, 1)
    bc     = np.array(base_rgb, dtype=float)
    colors = np.clip(brightness[:, None] * bc, 0, 1)
    alpha  = np.full((len(colors), 1), 0.97)
    return np.hstack([colors, alpha])                     # (N, 4)


def load_meshes() -> Dict[str, tuple]:
    """全リンクメッシュをダウンロード・パース・デシメーション・Phong シェーディング計算"""
    meshes = {}
    names = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
    for idx, name in enumerate(names):
        raw = download_stl(name)
        tris, normals = parse_stl(raw)
        tris_d    = tris[::DECIMATE]
        normals_d = normals[::DECIMATE]
        tris_mm   = tris_d * 1000.0  # m → mm
        face_colors = compute_face_colors(tris_mm, normals_d, LINK_BASE_COLORS_RGB[idx])
        meshes[name] = (tris_mm, face_colors)
        print(f"  {name}.stl: {len(tris)} -> {len(tris_d)} tris")
    return meshes


# ══════════════════════════════════════════════════════════════════════════════
# Forward Kinematics (URDF ジョイントチェーン)
# ══════════════════════════════════════════════════════════════════════════════

def rpy_to_R(rpy) -> np.ndarray:
    """RPY → 3x3 回転行列 (R = Rz @ Ry @ Rx)"""
    r, p, y = rpy
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0         ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    return Rz @ Ry @ Rx


def axis_angle_R(axis, angle) -> np.ndarray:
    """Rodrigues の公式 → 3x3 回転行列"""
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3)
    axis = axis / norm
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def make_T(xyz, R=None, rpy=None) -> np.ndarray:
    """4x4 同次変換行列を生成"""
    T = np.eye(4)
    T[:3, 3] = xyz
    if R is not None:
        T[:3, :3] = R
    elif rpy is not None:
        T[:3, :3] = rpy_to_R(rpy)
    return T


def forward_kinematics(joint_angles: List[float]) -> List[np.ndarray]:
    """
    関節角度から各リンクフレームの world 変換行列リストを返す

    Returns:
        [T_base, T_link1, T_link2, ..., T_link6, T_eef]  (8個)
        各 T は 4x4 homogeneous matrix (単位: m)
    """
    transforms = [np.eye(4)]   # T_base = Identity
    T = np.eye(4)
    ai = 0  # joint_angles のインデックス

    for jtype, xyz_m, rpy, axis in URDF_JOINTS:
        # ジョイント原点への変換
        T_origin = make_T(xyz_m, rpy=rpy)
        T = T @ T_origin

        if jtype == "revolute" and ai < len(joint_angles):
            R_joint = axis_angle_R(axis, joint_angles[ai])
            T = T @ make_T([0, 0, 0], R=R_joint)
            ai += 1

        transforms.append(T.copy())

    return transforms   # 8 transforms: base, link1..link6, eef


def get_tcp_mm(joint_angles: List[float]) -> np.ndarray:
    """TCP 位置を mm 単位で返す"""
    Ts = forward_kinematics(joint_angles)
    return Ts[-1][:3, 3] * 1000.0


def transform_mesh(tris_mm: np.ndarray, T_m: np.ndarray) -> np.ndarray:
    """
    メッシュ三角形 (N,3,3) [mm] を world 変換行列 T [m] で変換

    T は m 単位の変換行列なので、mm → m → 変換 → mm の変換を行う
    """
    N = len(tris_mm)
    flat = tris_mm.reshape(-1, 3) / 1000.0          # mm → m
    homog = np.c_[flat, np.ones(len(flat))]          # (N*3, 4)
    world = (T_m @ homog.T).T[:, :3] * 1000.0       # m → mm
    return world.reshape(N, 3, 3)


# ══════════════════════════════════════════════════════════════════════════════
# アニメーション キーフレーム (関節角度 [rad])
# ══════════════════════════════════════════════════════════════════════════════
#
# Lite 6 関節可動範囲:
#   joint1: ±360°  joint2: ±150°  joint3: -3.5°〜300°
#   joint4: ±360°  joint5: ±124°  joint6: ±360°
#
# 各キーフレーム: (q1,q2,q3,q4,q5,q6, gripper_open, label)

# 12 キーフレーム: [0] Startup, [1] Planning, [2-9] 動作, [10] Home, [11] Complete
KEYFRAMES = [
    ( 0.0,  -0.8,  1.4,  0.0,  0.8,  0.0,  0.4,  ""),  # 0 Startup
    ( 0.0,  -0.8,  1.4,  0.0,  0.8,  0.0,  0.4,  ""),  # 1 Planning (same pos)
    ( 0.0,  -0.3,  1.2,  0.0,  0.5,  0.0,  1.0,  ""),  # 2 Approach
    ( 0.0,   0.1,  1.0,  0.0,  0.2,  0.0,  1.0,  ""),  # 3 Descend
    ( 0.0,   0.1,  1.0,  0.0,  0.2,  0.0,  0.1,  ""),  # 4 Grasp  ← pick_frame
    ( 0.0,  -0.3,  1.2,  0.0,  0.5,  0.0,  0.1,  ""),  # 5 Lift
    ( 0.9,  -0.3,  1.2,  0.0,  0.5,  0.0,  0.1,  ""),  # 6 Transport
    ( 0.9,   0.1,  1.0,  0.0,  0.2,  0.0,  0.1,  ""),  # 7 Lower
    ( 0.9,   0.1,  1.0,  0.0,  0.2,  0.0,  1.0,  ""),  # 8 Release ← drop_frame
    ( 0.9,  -0.3,  1.2,  0.0,  0.5,  0.0,  1.0,  ""),  # 9 Retract
    ( 0.0,  -0.8,  1.4,  0.0,  0.8,  0.0,  0.4,  ""),  # 10 Home
    ( 0.0,  -0.8,  1.4,  0.0,  0.8,  0.0,  0.4,  ""),  # 11 Complete (static)
]

FRAMES_PER_SEG = 40   # キーフレーム間の補間フレーム数 (40×11+1 = 441 frames ≈ 29s)
FPS = 15              # 出力フレームレート

# ── LLM/タスク ナレーティブ (セグメントごと) ──────────────────────────────────
# 各エントリ: task, think, tool, result
PHASE_DATA = [
    # seg 0: Startup
    dict(
        task="[INIT] System Online",
        think="Initializing RAI agent...\n"
              "  ROS2 node: connected\n"
              "  Camera feed: active\n"
              "  Robot arm: ready\n\n"
              "Waiting for task...",
        tool='get_camera_image({})',
        result='  -> 1280x720px captured\n'
               '  -> Scene analysis: OK',
    ),
    # seg 1: Planning
    dict(
        task="[PLAN] Generating Plan",
        think='Task received:\n'
              '  "Pick red block, move right"\n\n'
              'VLM: red block detected at\n'
              '  approx (293, 0) mm\n\n'
              'Execution plan:\n'
              '  1. move_to_pose (approach)\n'
              '  2. move_to_pose (descend)\n'
              '  3. control_gripper (close)\n'
              '  4. move_to_pose (lift)\n'
              '  5. move_to_pose (transport)\n'
              '  6. move_to_pose (lower)\n'
              '  7. control_gripper (open)',
        tool='move_home({})',
        result='  -> TCP: (51, 0, 577) mm\n'
               '  -> Status: OK',
    ),
    # seg 2: Approach
    dict(
        task="[EXEC] 1/7  Approach",
        think='Moving to approach position\n'
              'above the red block.\n\n'
              'Target approach height:\n'
              '  z = 472 mm\n'
              'Gripper: OPEN (1.0)',
        tool='move_to_pose({\n'
             '  "x": 227, "y": 0,\n'
             '  "z": 472, "speed": 150\n'
             '})\n'
             'control_gripper({"position": 1.0})',
        result='  -> TCP: (227, 0, 472) mm\n'
               '  -> Gripper: open\n'
               '  -> OK',
    ),
    # seg 3: Descend
    dict(
        task="[EXEC] 2/7  Descend",
        think='Descending slowly to\n'
              'grasp height.\n\n'
              'Grasp z: 322 mm\n'
              'Speed: reduced to 80\n'
              'Gripper stays OPEN',
        tool='move_to_pose({\n'
             '  "x": 293, "y": 0,\n'
             '  "z": 322, "speed": 80\n'
             '})',
        result='  -> TCP: (293, 0, 322) mm\n'
               '  -> At grasp position\n'
               '  -> OK',
    ),
    # seg 4: Grasp
    dict(
        task="[EXEC] 3/7  Grasp",
        think='Object in gripper range!\n\n'
              'Closing gripper to\n'
              'secure the block.\n\n'
              'Target position: 0.1\n'
              '(10% = fully closed)',
        tool='control_gripper({\n'
             '  "position": 0.1\n'
             '})',
        result='  -> Gripper: CLOSED\n'
               '  -> Object secured\n'
               '  -> OK',
    ),
    # seg 5: Lift
    dict(
        task="[EXEC] 4/7  Lift",
        think='Object grasped.\n'
              'Lifting to safe\n'
              'transport height.\n\n'
              'Target z: 472 mm\n'
              'Carrying: red block',
        tool='move_to_pose({\n'
             '  "x": 227, "y": 0,\n'
             '  "z": 472, "speed": 120\n'
             '})',
        result='  -> TCP: (227, 0, 472) mm\n'
               '  -> Carrying object\n'
               '  -> OK',
    ),
    # seg 6: Transport
    dict(
        task="[EXEC] 5/7  Transport",
        think='Moving to drop zone.\n\n'
              'Rotating joint 1:\n'
              '  +51.6 deg\n\n'
              'Drop target:\n'
              '  (182, 229) mm',
        tool='move_to_pose({\n'
             '  "x": 141, "y": 177,\n'
             '  "z": 472, "speed": 150\n'
             '})',
        result='  -> TCP: (141,177,472)mm\n'
               '  -> At drop zone\n'
               '  -> OK',
    ),
    # seg 7: Lower
    dict(
        task="[EXEC] 6/7  Lower",
        think='Arrived at drop zone.\n'
              'Lowering to place\n'
              'position.\n\n'
              'Place z: 322 mm\n'
              '(same height as pick)',
        tool='move_to_pose({\n'
             '  "x": 182, "y": 229,\n'
             '  "z": 322, "speed": 80\n'
             '})',
        result='  -> TCP: (182,229,322)mm\n'
               '  -> At place position\n'
               '  -> OK',
    ),
    # seg 8: Release
    dict(
        task="[EXEC] 7/7  Release",
        think='Object at destination.\n\n'
              'Opening gripper to\n'
              'release the block.\n\n'
              'Target position: 1.0\n'
              '(100% = fully open)',
        tool='control_gripper({\n'
             '  "position": 1.0\n'
             '})',
        result='  -> Gripper: OPEN\n'
               '  -> Object placed\n'
               '  -> OK',
    ),
    # seg 9: Retract
    dict(
        task="[DONE] Retracting",
        think='All steps complete!\n\n'
              'Retracting arm to\n'
              'safe height before\n'
              'returning to home.',
        tool='move_to_pose({\n'
             '  "x": 141, "y": 177,\n'
             '  "z": 472, "speed": 120\n'
             '})',
        result='  -> Arm retracted\n'
               '  -> OK',
    ),
    # seg 10: Return Home
    dict(
        task="[DONE] Return Home",
        think='Returning to home.\n\n'
              'Task summary:\n'
              '  Object: red block\n'
              '  Picked: (293,0,302)mm\n'
              '  Placed: (182,229,302)mm\n'
              '  Tools used: 8 calls\n'
              '  Status: SUCCESS',
        tool='move_home({})',
        result='  -> TCP: (51, 0, 577) mm\n'
               '  -> Home position\n'
               '  -> Task COMPLETE',
    ),
]


def smooth_step(t: float) -> float:
    return t * t * (3 - 2 * t)


def interpolate_keyframes(kf, n_seg):
    frames = []
    for i in range(len(kf) - 1):
        q0, g0, lbl1 = np.array(kf[i][:6]), kf[i][6], kf[i][7]
        q1, g1, lbl2 = np.array(kf[i+1][:6]), kf[i+1][6], kf[i+1][7]
        for f in range(n_seg):
            t = smooth_step(f / n_seg)
            q = q0 + (q1 - q0) * t
            g = g0 + (g1 - g0) * t
            frames.append((q, g, lbl2))
    frames.append((np.array(kf[-1][:6]), kf[-1][6], kf[-1][7]))
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# グリッパー描画
# ══════════════════════════════════════════════════════════════════════════════

def compute_gripper_lines(T_eef_m: np.ndarray, open_ratio: float):
    """
    EEF フレームから 2 本の指先の始点・終点を mm で返す
    """
    MAX_SPREAD = 35.0  # mm 半開き
    FINGER_LEN = 30.0  # mm 指の長さ

    # EEF の位置と Z 軸 (接近方向) を取得
    origin = T_eef_m[:3, 3] * 1000.0
    z_axis = T_eef_m[:3, 2]  # 接近方向
    y_axis = T_eef_m[:3, 1]  # 指の開き方向

    spread = MAX_SPREAD * open_ratio
    # 2 本の指 (y_axis に沿ってオフセット)
    base_l = origin + y_axis * spread
    base_r = origin - y_axis * spread
    tip_l  = base_l + z_axis * FINGER_LEN
    tip_r  = base_r + z_axis * FINGER_LEN

    return base_l, tip_l, base_r, tip_r


# ══════════════════════════════════════════════════════════════════════════════
# アニメーション生成
# ══════════════════════════════════════════════════════════════════════════════

def create_animation(meshes: Dict[str, np.ndarray]):
    frames = interpolate_keyframes(KEYFRAMES, FRAMES_PER_SEG)
    total = len(frames)
    link_names = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]

    # ── Figure セットアップ ────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
    fig.suptitle("RAI × UFACTORY Lite 6  ピック＆プレース デモ",
                 color="white", fontsize=14, fontweight="bold", y=0.97)

    # 左: 3D ビュー
    ax = fig.add_axes([0.0, 0.0, 0.62, 0.95], projection="3d")
    ax.set_facecolor("#0d1b2a")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor("#0d1b2a")
        pane.set_edgecolor("#2a4060")
    ax.tick_params(colors="#88aacc", labelsize=7)
    ax.set_xlabel("X (mm)", color="#88aacc", fontsize=8)
    ax.set_ylabel("Y (mm)", color="#88aacc", fontsize=8)
    ax.set_zlabel("Z (mm)", color="#88aacc", fontsize=8)
    ax.set_xlim(-150, 500)
    ax.set_ylim(-350, 450)
    ax.set_zlim(0, 700)
    ax.view_init(elev=22, azim=-40)

    # FK から実際の把持/解放 TCP 位置を計算 (z は投影しない)
    _Ts_grasp = forward_kinematics(list(KEYFRAMES[4][:6]))   # KF[4] = Grasp
    pick_pos  = _Ts_grasp[-1][:3, 3] * 1000.0

    _Ts_drop = forward_kinematics(list(KEYFRAMES[8][:6]))    # KF[8] = Release
    drop_pos  = _Ts_drop[-1][:3, 3] * 1000.0

    BLOCK_HALF = 20.0
    work_z = pick_pos[2] - BLOCK_HALF

    pick_frame = FRAMES_PER_SEG * 4   # KF[4] = Grasp
    drop_frame = FRAMES_PER_SEG * 8   # KF[8] = Release

    # 作業台 (work_z の高さに描画)
    _tx = np.array([ 50, 480, 480,  50,  50])
    _ty = np.array([-350, -350, 400, 400, -350])
    ax.plot(_tx, _ty, [work_z]*5, color="#4488bb", lw=1.2, alpha=0.8)
    ax.plot_surface(
        np.array([[ 50, 480], [ 50, 480]]),
        np.array([[-350, -350], [400, 400]]),
        np.full((2, 2), work_z),
        alpha=0.18, color="#3366aa")
    # 床面 (薄く)
    ax.plot_surface(
        np.array([[-200, 500], [-200, 500]]),
        np.array([[-400, -400], [450, 450]]),
        np.zeros((2, 2)),
        alpha=0.05, color="#223355")

    obj_scatter = ax.scatter([pick_pos[0]], [pick_pos[1]], [pick_pos[2]],
                              c="#ff4444", s=500, marker="s", zorder=8,
                              depthshade=False, edgecolors="#ffcccc", linewidths=1.5)
    # ドロップゾーンマーカー (テーブル面に投影して表示)
    ax.scatter([drop_pos[0]], [drop_pos[1]], [pick_pos[2]],
               c="#44ff88", s=300, marker="^",
               alpha=0.55, zorder=4, depthshade=False,
               edgecolors="#aaffcc", linewidths=1)

    # ── メッシュ Poly3DCollection の初期生成 ───────────────────────────
    link_collections = []
    q0, _, _ = frames[0]
    Ts0 = forward_kinematics(q0)

    for i, name in enumerate(link_names):
        tris_mm_local, face_colors = meshes[name]
        tris_world = transform_mesh(tris_mm_local, Ts0[i])
        coll = Poly3DCollection(tris_world, linewidth=0.15)
        coll.set_facecolor(face_colors)       # per-face Phong shading
        coll.set_edgecolor("#111122")         # 細い暗エッジでCAD感
        ax.add_collection3d(coll)
        link_collections.append(coll)

    # ジョイントマーカー
    joint_pos0 = [T[:3, 3] * 1000 for T in Ts0[1:7]]
    jx = [p[0] for p in joint_pos0]
    jy = [p[1] for p in joint_pos0]
    jz = [p[2] for p in joint_pos0]
    joint_scatter = ax.scatter(jx, jy, jz, c="#f0a500", s=60, zorder=10,
                               depthshade=False)

    # グリッパー
    grip_L, = ax.plot([], [], [], color="#ff5533", linewidth=5, zorder=11, solid_capstyle="round")
    grip_R, = ax.plot([], [], [], color="#ff5533", linewidth=5, zorder=11, solid_capstyle="round")

    # TCP 軌跡
    tcp_trail, = ax.plot([], [], [], color="#4488cc", lw=0.8, alpha=0.5,
                          linestyle="--", zorder=3)
    trail_x, trail_y, trail_z = [], [], []

    # ── 右側情報パネル (4ゾーン) ─────────────────────────────────────
    ax_info = fig.add_axes([0.61, 0.01, 0.38, 0.96])
    ax_info.set_facecolor("#060e1a")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    # ── ヘッダー ────────────────────────────────────────────────────
    ax_info.text(0.5, 0.977, "RAI Agent",
                 ha="center", va="top", color="#ffffff",
                 fontsize=11, fontweight="bold", fontfamily="monospace")
    ax_info.plot([0.02, 0.98], [0.955, 0.955], color="#1a3a6a", lw=1.5)

    # ── Zone 1: User command (静的) ─────────────────────────────────
    ax_info.text(0.04, 0.948, "[User]",
                 ha="left", va="top", color="#55aaff",
                 fontsize=8, fontweight="bold", fontfamily="monospace")
    ax_info.text(0.04, 0.924,
                 'Pick the red block and\nplace it on the right.',
                 ha="left", va="top", color="#88ccff",
                 fontsize=8.5, fontfamily="monospace", linespacing=1.5,
                 bbox=dict(boxstyle="square,pad=0.3", facecolor="#0a1e3a",
                           edgecolor="#1a3a6a", linewidth=1))
    ax_info.plot([0.02, 0.98], [0.860, 0.860], color="#1a3a6a", lw=0.8)

    # ── Zone 2: Agent thinking ───────────────────────────────────────
    ax_info.text(0.04, 0.852, "[Agent Thinking]",
                 ha="left", va="top", color="#ffd700",
                 fontsize=8, fontweight="bold", fontfamily="monospace")
    think_text = ax_info.text(0.04, 0.828, "",
                               ha="left", va="top", color="#ffffff",
                               fontsize=7.8, fontfamily="monospace",
                               linespacing=1.65,
                               bbox=dict(boxstyle="square,pad=0.35",
                                         facecolor="#0c1a2e",
                                         edgecolor="#2a4a1a", linewidth=1,
                                         alpha=1.0))
    ax_info.plot([0.02, 0.98], [0.560, 0.560], color="#1a3a6a", lw=0.8)

    # ── Zone 3: Tool execution ───────────────────────────────────────
    ax_info.text(0.04, 0.552, "[Tool Execution]",
                 ha="left", va="top", color="#ff9944",
                 fontsize=8, fontweight="bold", fontfamily="monospace")
    tool_text = ax_info.text(0.04, 0.528, "",
                              ha="left", va="top", color="#aaffaa",
                              fontsize=7.8, fontfamily="monospace",
                              linespacing=1.65,
                              bbox=dict(boxstyle="square,pad=0.35",
                                        facecolor="#0c1a0c",
                                        edgecolor="#1a4a1a", linewidth=1,
                                        alpha=1.0))
    ax_info.plot([0.02, 0.98], [0.330, 0.330], color="#1a3a6a", lw=0.8)

    # ── Zone 4: Task status ──────────────────────────────────────────
    task_label = ax_info.text(0.5, 0.308, "",
                               ha="center", va="top",
                               color="#ffe566", fontsize=10,
                               fontweight="bold", fontfamily="monospace",
                               bbox=dict(boxstyle="round,pad=0.45",
                                         facecolor="#0d1f3c",
                                         edgecolor="#ff6b35", linewidth=2.5))

    # 関節角度 (ボックス付き白文字)
    joint_text = ax_info.text(0.04, 0.230, "",
                               ha="left", va="top",
                               color="#ffffff", fontsize=7.8,
                               fontfamily="monospace", linespacing=1.65,
                               bbox=dict(boxstyle="square,pad=0.35",
                                         facecolor="#0a1a30",
                                         edgecolor="#2255aa", linewidth=1,
                                         alpha=1.0))

    # 進捗バー
    ax_info.text(0.04, 0.088, "Progress:",
                 ha="left", va="center",
                 color="#aaccee", fontsize=8, fontfamily="monospace")
    ax_info.plot([0.04, 0.96], [0.065, 0.065], color="#1a2d50", lw=12,
                 solid_capstyle="butt")
    prog_bar, = ax_info.plot([0.04, 0.04], [0.065, 0.065], color="#ff6b35",
                              lw=12, solid_capstyle="butt")

    frame_ctr = ax_info.text(0.96, 0.022, "",
                              ha="right", va="bottom",
                              color="#7799bb", fontsize=7.5,
                              fontfamily="monospace")

    # ── アップデート関数 ────────────────────────────────────────────
    def update(fi: int):
        nonlocal trail_x, trail_y, trail_z

        q, grip, _ = frames[fi]
        Ts = forward_kinematics(q)

        # メッシュを更新 (ジオメトリのみ、フェース色はローカル空間で固定)
        for i, name in enumerate(link_names):
            tris_mm_local = meshes[name][0]
            tris_w = transform_mesh(tris_mm_local, Ts[i])
            link_collections[i].set_verts(tris_w)

        # ジョイントマーカー
        jpts = [T[:3, 3] * 1000 for T in Ts[1:7]]
        joint_scatter._offsets3d = (
            [p[0] for p in jpts],
            [p[1] for p in jpts],
            [p[2] for p in jpts],
        )

        # グリッパー
        T_eef = Ts[-1]
        bl, tl, br, tr = compute_gripper_lines(T_eef, grip)
        grip_L.set_data([bl[0], tl[0]], [bl[1], tl[1]])
        grip_L.set_3d_properties([bl[2], tl[2]])
        grip_R.set_data([br[0], tr[0]], [br[1], tr[1]])
        grip_R.set_3d_properties([br[2], tr[2]])

        # TCP 位置
        tcp = Ts[-1][:3, 3] * 1000

        # 把持オブジェクト追跡
        if fi < pick_frame:
            obj_pos = pick_pos                          # テーブル上で静止
        elif fi < drop_frame:
            obj_pos = tcp.copy()                        # TCP に追従 (grasped)
        else:
            # 解放後: drop_pos の x,y、pick_pos の z (同じ作業面高さ)
            obj_pos = np.array([drop_pos[0], drop_pos[1], pick_pos[2]])
        obj_scatter._offsets3d = ([obj_pos[0]], [obj_pos[1]], [obj_pos[2]])

        # TCP 軌跡
        trail_x.append(tcp[0])
        trail_y.append(tcp[1])
        trail_z.append(tcp[2])
        if len(trail_x) > 120:
            trail_x = trail_x[-120:]
            trail_y = trail_y[-120:]
            trail_z = trail_z[-120:]
        tcp_trail.set_data(trail_x, trail_y)
        tcp_trail.set_3d_properties(trail_z)

        # ── LLM ナレーティブ更新 ──────────────────────────────────────
        seg = min(fi // FRAMES_PER_SEG, len(PHASE_DATA) - 1)
        t_in_seg = (fi % FRAMES_PER_SEG) / max(FRAMES_PER_SEG - 1, 1)
        phase = PHASE_DATA[seg]

        # Agent thinking: セグメント前半で行を逐次表示
        think_lines = phase["think"].split("\n")
        n_show = max(1, int(len(think_lines) * min(1.0, t_in_seg * 2.2)))
        think_text.set_text("\n".join(think_lines[:n_show]))

        # Tool execution: 前半=コール表示, 後半=結果追加
        if t_in_seg < 0.50:
            tool_text.set_text(phase["tool"])
        else:
            tool_text.set_text(phase["tool"] + "\n" + phase["result"])

        task_label.set_text(phase["task"])

        # 関節角度 (ASCII only)
        joint_text.set_text(
            "Joint Angles:\n" +
            "\n".join(f"  J{i+1}: {np.degrees(a):+7.1f} deg"
                      for i, a in enumerate(q))
        )

        # プログレスバー
        p = (fi + 1) / total
        prog_bar.set_xdata([0.04, 0.04 + 0.92 * p])
        prog_bar.set_ydata([0.065, 0.065])

        frame_ctr.set_text(f"{fi/FPS:.1f}s / {total/FPS:.1f}s")

        return (link_collections + [joint_scatter, grip_L, grip_R,
                obj_scatter, tcp_trail, think_text, tool_text,
                task_label, joint_text, prog_bar, frame_ctr])

    ani = animation.FuncAnimation(
        fig, update, frames=total,
        interval=1000 // FPS, blit=False,
    )
    return fig, ani, total


# ══════════════════════════════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════════════════════════════

def save_mp4(ani, path, fps, total):
    print(f"\n[INFO] MP4 エンコード中: {path}")
    print(f"       フレーム数: {total} / {fps}fps ({total/fps:.1f}秒)")
    print(f"       ※ STL メッシュ描画のため数分かかる場合があります...")
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"title": "RAI x UFACTORY Lite6 Demo"},
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", "-preset", "fast"],
    )
    ani.save(path, writer=writer, dpi=110)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[OK] 保存完了: {path} ({size_mb:.1f} MB)")


def save_gif(ani, path, fps, total):
    print(f"[INFO] GIF エンコード中: {path} ({fps}fps, {total}frames)")
    writer = animation.PillowWriter(fps=fps)
    ani.save(path, writer=writer, dpi=80)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[OK] 保存完了: {path} ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# エントリポイント
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    mp4_path = os.path.join(out_dir, "rai_lite6_demo.mp4")
    gif_path = os.path.join(out_dir, "rai_lite6_demo.gif")

    print("=" * 60)
    print("  RAI × UFACTORY Lite 6 デモ動画生成 (URDF + STL メッシュ版)")
    print("=" * 60)

    # STL メッシュ読み込み
    print("\n[1/3] STL メッシュを取得・デシメーション中...")
    t0 = time.time()
    meshes = load_meshes()
    print(f"      完了 ({time.time()-t0:.1f}秒)")

    # アニメーション生成
    print("\n[2/3] アニメーションを構築中...")
    fig, ani, total = create_animation(meshes)
    print(f"      フレーム数: {total}")

    # 保存
    print("\n[3/3] 動画を保存中...")
    if animation.FFMpegWriter.isAvailable():
        save_mp4(ani, mp4_path, FPS, total)
    else:
        print("[WARN] ffmpeg が見つかりません → GIF で保存")
        print("       MP4 には: winget install ffmpeg")
        try:
            save_gif(ani, gif_path, fps=10, total=total)
        except ImportError:
            print("[ERROR] pip install pillow が必要です")
            sys.exit(1)

    plt.close(fig)
    print("\n完了しました。")


if __name__ == "__main__":
    main()
