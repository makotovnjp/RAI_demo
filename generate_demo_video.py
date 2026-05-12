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

# 各リンクメッシュの色 (link_base, link1 … link6)
LINK_COLORS = [
    [0.88, 0.88, 0.88, 0.95],  # base   — ライトグレー
    [0.75, 0.75, 0.75, 0.95],  # link1  — シルバー
    [0.88, 0.88, 0.88, 0.95],  # link2  — ホワイト
    [0.75, 0.75, 0.75, 0.95],  # link3  — シルバー
    [0.88, 0.88, 0.88, 0.95],  # link4  — ホワイト
    [0.75, 0.75, 0.75, 0.95],  # link5  — シルバー
    [0.88, 0.88, 0.88, 0.95],  # link6  — ホワイト
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
DECIMATE = 35   # 1/35 のフェースを残す → 約 200~400 三角形/リンク


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


def parse_stl(data: bytes) -> np.ndarray:
    """
    バイナリ STL → (N, 3, 3) 三角形配列 [m単位]
    """
    num = struct.unpack_from("<I", data, 80)[0]
    tris = np.zeros((num, 3, 3), dtype=np.float32)
    offset = 84
    for i in range(num):
        offset += 12  # ノーマル skip
        tris[i, 0] = struct.unpack_from("<fff", data, offset); offset += 12
        tris[i, 1] = struct.unpack_from("<fff", data, offset); offset += 12
        tris[i, 2] = struct.unpack_from("<fff", data, offset); offset += 12
        offset += 2
    return tris


def decimate(tris: np.ndarray, factor: int) -> np.ndarray:
    """単純間引き (factor 個に 1 個を残す)"""
    return tris[::factor]


def load_meshes() -> Dict[str, np.ndarray]:
    """全リンクメッシュをダウンロード・パース・デシメーション"""
    meshes = {}
    names = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
    for name in names:
        raw = download_stl(name)
        tris = parse_stl(raw)
        tris_d = decimate(tris, DECIMATE)
        meshes[name] = tris_d * 1000.0  # m → mm
        print(f"  {name}.stl: {len(tris)} → {len(tris_d)} 三角形")
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

KEYFRAMES = [
    # ホームポジション
    ( 0.0,  -0.8,  1.4,  0.0,  0.8,  0.0,  0.4,  "ホームポジション"),
    # アプローチ (物体の上方)
    ( 0.0,  -0.3,  1.2,  0.0,  0.5,  0.0,  1.0,  "アプローチ\nグリッパー開"),
    # 把持高さへ降下
    ( 0.0,   0.1,  1.0,  0.0,  0.2,  0.0,  1.0,  "降下\n把持準備"),
    # 把持
    ( 0.0,   0.1,  1.0,  0.0,  0.2,  0.0,  0.1,  "把持完了\nグリッパー閉"),
    # 持ち上げ
    ( 0.0,  -0.3,  1.2,  0.0,  0.5,  0.0,  0.1,  "持ち上げ"),
    # ドロップゾーンへ移動 (yaw 方向に旋回)
    ( 0.9,  -0.3,  1.2,  0.0,  0.5,  0.0,  0.1,  "ドロップゾーンへ移動"),
    # ドロップ降下
    ( 0.9,   0.0,  1.1,  0.0,  0.3,  0.0,  0.1,  "配置位置へ降下"),
    # 解放
    ( 0.9,   0.0,  1.1,  0.0,  0.3,  0.0,  1.0,  "物体を解放\nグリッパー開"),
    # 引き上げ
    ( 0.9,  -0.3,  1.2,  0.0,  0.5,  0.0,  1.0,  "引き上げ"),
    # ホームへ帰還
    ( 0.0,  -0.8,  1.4,  0.0,  0.8,  0.0,  0.4,  "ホームへ帰還"),
]

FRAMES_PER_SEG = 20   # キーフレーム間の補間フレーム数
FPS = 15              # 出力フレームレート


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
    ax.set_facecolor("#16213e")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#334466")
    ax.tick_params(colors="#aaaacc", labelsize=7)
    ax.set_xlabel("X (mm)", color="#aaaacc", fontsize=8)
    ax.set_ylabel("Y (mm)", color="#aaaacc", fontsize=8)
    ax.set_zlabel("Z (mm)", color="#aaaacc", fontsize=8)
    ax.set_xlim(-250, 500)
    ax.set_ylim(-400, 400)
    ax.set_zlim(0, 650)
    ax.view_init(elev=18, azim=-45)

    # 作業台
    tx = np.array([100, 450, 450, 100, 100])
    ty = np.array([-300, -300, 300, 300, -300])
    ax.plot(tx, ty, [0]*5, color="#445588", lw=0.8, alpha=0.5)
    ax.plot_surface(
        np.array([[100, 450], [100, 450]]),
        np.array([[-300, -300], [300, 300]]),
        np.zeros((2, 2)),
        alpha=0.08, color="#6688bb")

    # 把持対象オブジェクト (赤いブロック)
    pick_pos  = np.array([280.0, 0.0,  0.0])   # 初期位置
    drop_pos  = np.array([180.0, 270.0, 0.0])  # 配置位置
    pick_frame = FRAMES_PER_SEG * 3            # 把持フレーム
    drop_frame = FRAMES_PER_SEG * 7            # 解放フレーム

    obj_scatter = ax.scatter([pick_pos[0]], [pick_pos[1]], [pick_pos[2]],
                              c="#ff4444", s=400, marker="s", zorder=8,
                              depthshade=False, edgecolors="#ffaaaa", linewidths=1)
    ax.scatter(*drop_pos, c="#44ff88", s=200, marker="^",
               alpha=0.4, zorder=4, depthshade=False)

    # ── メッシュ Poly3DCollection の初期生成 ───────────────────────────
    link_collections = []
    q0, _, _ = frames[0]
    Ts0 = forward_kinematics(q0)

    for i, name in enumerate(link_names):
        tris_world = transform_mesh(meshes[name], Ts0[i])
        coll = Poly3DCollection(
            tris_world,
            alpha=LINK_COLORS[i][3],
            linewidth=0,
        )
        coll.set_facecolor(LINK_COLORS[i][:3])
        coll.set_edgecolor("none")
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
    grip_L, = ax.plot([], [], [], color="#e94560", linewidth=3, zorder=11)
    grip_R, = ax.plot([], [], [], color="#e94560", linewidth=3, zorder=11)

    # TCP 軌跡
    tcp_trail, = ax.plot([], [], [], color="#4488cc", lw=0.8, alpha=0.5,
                          linestyle="--", zorder=3)
    trail_x, trail_y, trail_z = [], [], []

    # ── 右側情報パネル ───────────────────────────────────────────────
    ax_info = fig.add_axes([0.63, 0.05, 0.36, 0.88])
    ax_info.set_facecolor("#0f3460")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    ax_info.text(0.5, 0.97, "RAI エージェント ログ",
                 ha="center", va="top", color="white",
                 fontsize=11, fontweight="bold")
    ax_info.plot([0.02, 0.98], [0.93, 0.93], color="#445588", lw=0.8)

    log_text  = ax_info.text(0.05, 0.89, "", ha="left", va="top",
                              color="#a0e7e5", fontsize=8.5, linespacing=1.7,
                              fontfamily="monospace")
    task_label = ax_info.text(0.5, 0.28, "", ha="center", va="center",
                               color="white", fontsize=11, fontweight="bold",
                               bbox=dict(boxstyle="round,pad=0.5",
                                         facecolor="#1a1a2e",
                                         edgecolor="#e94560", linewidth=2))
    ax_info.text(0.05, 0.12, "進捗:", ha="left", va="center",
                 color="#aaaacc", fontsize=8)
    ax_info.plot([0.05, 0.95], [0.095, 0.095], color="#334466", lw=14,
                 solid_capstyle="butt")
    prog_bar, = ax_info.plot([0.05, 0.05], [0.095, 0.095], color="#e94560",
                              lw=14, solid_capstyle="butt")

    # 関節角度テキスト
    joint_text = ax_info.text(0.05, 0.57, "", ha="left", va="top",
                               color="#d0d0d0", fontsize=7.5,
                               fontfamily="monospace", linespacing=1.6)

    frame_ctr = ax_info.text(0.95, 0.04, "", ha="right", va="bottom",
                               color="#666688", fontsize=7)

    LOG_LINES: List[str] = []

    # ── アップデート関数 ────────────────────────────────────────────
    def update(fi: int):
        nonlocal trail_x, trail_y, trail_z, LOG_LINES

        q, grip, label = frames[fi]
        Ts = forward_kinematics(q)

        # メッシュを更新
        for i, name in enumerate(link_names):
            tris_w = transform_mesh(meshes[name], Ts[i])
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
            obj_pos = pick_pos
        elif fi < drop_frame:
            obj_pos = tcp + Ts[-1][:3, 2] * 20  # TCP 先端に追従
        else:
            obj_pos = drop_pos
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

        # ログ更新
        seg = fi // FRAMES_PER_SEG
        prog = fi / FRAMES_PER_SEG - seg
        if seg < len(KEYFRAMES) - 1:
            log = (
                f"[Tool→] move_to_pose\n"
                f"  q=[{', '.join(f'{np.degrees(a):.0f}' for a in q)}]°\n"
                f"  TCP=({tcp[0]:.0f},{tcp[1]:.0f},{tcp[2]:.0f})mm\n"
                f"[←Tool] 移動中 {prog*100:.0f}%\n"
                f"[Tool→] control_gripper\n"
                f"  position={grip:.2f}\n"
            )
            if not LOG_LINES or LOG_LINES[-1] != log:
                LOG_LINES.append(log)
                if len(LOG_LINES) > 3:
                    LOG_LINES = LOG_LINES[-3:]

        log_text.set_text("\n".join(LOG_LINES))
        task_label.set_text(label)

        # 関節角度表示
        joint_text.set_text(
            "関節角度:\n" +
            "\n".join(f"  J{i+1}: {np.degrees(a):+7.1f}°" for i, a in enumerate(q))
        )

        # プログレスバー
        p = (fi + 1) / total
        prog_bar.set_xdata([0.05, 0.05 + 0.90 * p])
        prog_bar.set_ydata([0.095, 0.095])

        frame_ctr.set_text(f"{fi/FPS:.1f}s / {total/FPS:.1f}s")

        return (link_collections + [joint_scatter, grip_L, grip_R,
                obj_scatter, tcp_trail, log_text, task_label,
                joint_text, prog_bar, frame_ctr])

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
