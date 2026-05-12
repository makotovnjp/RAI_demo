"""
RAI × UFACTORY Lite 6 デモアニメーション生成スクリプト
========================================================
ロボットアームのピック＆プレース動作を3Dアニメーション化してMP4に保存します。

必要パッケージ:
    pip install matplotlib numpy

動画エンコーダ（どちらか一方が必要）:
    - ffmpeg: https://ffmpeg.org/download.html  ← 推奨
    - pillow: pip install pillow               ← GIF出力のみ

実行:
    python generate_demo_video.py
    → rai_lite6_demo.mp4 が生成されます
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # ヘッドレス描画（画面表示なし）
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 日本語フォント設定（matplotlib 3.1.x 対応）
# sans-serif の先頭に Yu Gothic を追加することで日本語グリフを有効化
matplotlib.rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "MS Gothic"] \
    + matplotlib.rcParams.get("font.sans-serif", [])
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

JP_PROP = FontProperties(family="Yu Gothic")
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import sys
import os


# ============================================================
# ロボットアームの簡易逆運動学
# ============================================================

SHOULDER_HEIGHT = 243.0   # mm — ベースから肩関節までの高さ
UPPER_ARM_LEN   = 200.0   # mm — 上腕リンク長
FOREARM_LEN     = 314.0   # mm — 前腕リンク長（手首含む）
HAND_LEN        =  61.5   # mm — ハンドリンク長


def compute_arm_segments(tcp_x, tcp_y, tcp_z):
    """
    TCPの目標位置からアームの各関節位置を計算する（簡易IK）

    Returns:
        list of (x, y, z) tuples: [base_bottom, base_top, elbow, wrist, tcp]
    """
    # ウェストの回転角（yaw）
    waist = np.arctan2(tcp_y, tcp_x)

    # 肩関節の位置（ベース上部）
    shoulder = np.array([0.0, 0.0, SHOULDER_HEIGHT])

    # 水平距離（ベースからTCPまでのXY平面距離）
    r = np.sqrt(tcp_x**2 + tcp_y**2)

    # 肩関節を原点とした相対座標
    dx = r
    dz = tcp_z - SHOULDER_HEIGHT

    # 2リンクIK（上腕 + 前腕）
    d = np.sqrt(dx**2 + dz**2)
    d = np.clip(d, abs(UPPER_ARM_LEN - FOREARM_LEN) + 1.0,
                UPPER_ARM_LEN + FOREARM_LEN - 1.0)

    cos_elbow = (d**2 - UPPER_ARM_LEN**2 - FOREARM_LEN**2) / (2 * UPPER_ARM_LEN * FOREARM_LEN)
    cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
    elbow_angle = np.arccos(cos_elbow)  # 肘上配置（elbow-up）

    phi = np.arctan2(dz, dx)
    psi = np.arctan2(FOREARM_LEN * np.sin(elbow_angle),
                     UPPER_ARM_LEN + FOREARM_LEN * np.cos(elbow_angle))
    shoulder_angle = phi - psi

    # 肘関節の位置
    cos_w = np.cos(waist)
    sin_w = np.sin(waist)
    ex = cos_w * UPPER_ARM_LEN * np.cos(shoulder_angle)
    ey = sin_w * UPPER_ARM_LEN * np.cos(shoulder_angle)
    ez = SHOULDER_HEIGHT + UPPER_ARM_LEN * np.sin(shoulder_angle)
    elbow = np.array([ex, ey, ez])

    # TCPの位置
    tcp = np.array([tcp_x, tcp_y, tcp_z])

    # 手首関節の位置（TCPから手首分だけ肘方向へ戻る）
    elbow_to_tcp = tcp - elbow
    dist_et = np.linalg.norm(elbow_to_tcp)
    if dist_et > 0.1:
        unit = elbow_to_tcp / dist_et
        wrist = tcp - unit * HAND_LEN
    else:
        wrist = tcp - np.array([0.0, 0.0, HAND_LEN])

    return [
        np.array([0.0, 0.0, 0.0]),    # ベース底面
        shoulder,                       # ベース上部（肩）
        elbow,                          # 肘
        wrist,                          # 手首
        tcp,                            # TCP
    ]


def compute_gripper(tcp, waist_angle, open_ratio):
    """
    グリッパーの2本の指の位置を計算

    Args:
        tcp: TCP位置 [x, y, z]
        waist_angle: ウェスト角（rad）
        open_ratio: 0.0=閉 / 1.0=開

    Returns:
        (finger1_tip, finger2_tip): 各指先の位置
    """
    max_open = 35.0  # mm — 最大開き幅の半分
    spread = max_open * open_ratio

    # グリッパーの横方向（ウェストに垂直な方向）
    perp = np.array([-np.sin(waist_angle), np.cos(waist_angle), 0.0])
    down = np.array([0.0, 0.0, -1.0])

    # 指先位置（TCPから少し下へ）
    base_of_finger = tcp + down * 20.0
    f1 = base_of_finger + perp * spread
    f2 = base_of_finger - perp * spread
    return f1, f2


# ============================================================
# アニメーション キーフレーム定義
# ============================================================

# 各キーフレーム: (TCP_x, TCP_y, TCP_z, gripper_open_ratio, label)
KEYFRAMES = [
    # ホームポジション
    (300.0,   0.0, 350.0, 0.3, "ホームポジション"),

    # ピック前のアプローチ（物体の上方へ移動）
    (280.0,   0.0, 250.0, 1.0, "アプローチ開始\nグリッパー開"),

    # 物体高さまで降下
    (280.0,   0.0,  90.0, 1.0, "物体位置へ降下"),

    # 把持（グリッパー閉）
    (280.0,   0.0,  90.0, 0.1, "把持完了\nグリッパー閉"),

    # 持ち上げ
    (280.0,   0.0, 250.0, 0.1, "持ち上げ"),

    # ドロップゾーン上方へ移動
    (200.0, 180.0, 250.0, 0.1, "ドロップゾーンへ移動"),

    # ドロップ位置へ降下
    (200.0, 180.0, 120.0, 0.1, "配置位置へ降下"),

    # 解放（グリッパー開）
    (200.0, 180.0, 120.0, 1.0, "物体を解放\nグリッパー開"),

    # 引き上げ
    (200.0, 180.0, 250.0, 1.0, "引き上げ"),

    # ホームへ帰還
    (300.0,   0.0, 350.0, 0.3, "ホームへ帰還\n完了"),
]

FRAMES_PER_SEGMENT = 30   # キーフレーム間の補間フレーム数
FPS = 30


def smooth_step(t):
    """滑らかなS字補間 (ease in-out)"""
    return t * t * (3 - 2 * t)


def interpolate_keyframes(keyframes, frames_per_seg):
    """キーフレーム間を補間してフレームデータを生成"""
    frames = []
    for i in range(len(keyframes) - 1):
        x0, y0, z0, g0, _ = keyframes[i]
        x1, y1, z1, g1, label = keyframes[i + 1]

        for f in range(frames_per_seg):
            t = smooth_step(f / frames_per_seg)
            frames.append((
                x0 + (x1 - x0) * t,
                y0 + (y1 - y0) * t,
                z0 + (z1 - z0) * t,
                g0 + (g1 - g0) * t,
                label,
            ))

    # 最後のキーフレームを追加
    frames.append(keyframes[-1])
    return frames


# ============================================================
# 物体（ブロック）の位置追跡
# ============================================================

PICK_POSITION  = np.array([280.0,   0.0,  90.0])  # 物体の初期位置
DROP_POSITION  = np.array([200.0, 180.0, 120.0])  # 物体の配置位置
PICKUP_FRAME   = FRAMES_PER_SEGMENT * 3            # 把持フレーム
RELEASE_FRAME  = FRAMES_PER_SEGMENT * 7            # 解放フレーム


def get_object_position(frame_idx, tcp_pos):
    """現在フレームでの物体位置を返す"""
    if frame_idx < PICKUP_FRAME:
        return PICK_POSITION
    elif frame_idx < RELEASE_FRAME:
        return tcp_pos.copy()  # 把持中はTCPと同じ位置
    else:
        return DROP_POSITION


# ============================================================
# 描画・アニメーション
# ============================================================

def create_animation():
    frames_data = interpolate_keyframes(KEYFRAMES, FRAMES_PER_SEGMENT)
    total_frames = len(frames_data)

    # Figureセットアップ
    fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
    fig.suptitle("RAI × UFACTORY Lite 6  ピック＆プレース デモ",
                 color="white", fontsize=15, fontweight="bold",
                 fontfamily="sans-serif", y=0.97)

    # 左: 3Dビュー
    ax3d = fig.add_axes([0.0, 0.0, 0.62, 0.95], projection="3d")
    ax3d.set_facecolor("#16213e")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#334466")
    ax3d.yaxis.pane.set_edgecolor("#334466")
    ax3d.zaxis.pane.set_edgecolor("#334466")
    ax3d.tick_params(colors="#aaaacc", labelsize=7)
    ax3d.set_xlabel("X (mm)", color="#aaaacc", fontsize=8)
    ax3d.set_ylabel("Y (mm)", color="#aaaacc", fontsize=8)
    ax3d.set_zlabel("Z (mm)", color="#aaaacc", fontsize=8)
    ax3d.set_xlim(-50, 500)
    ax3d.set_ylim(-350, 350)
    ax3d.set_zlim(0, 550)
    ax3d.view_init(elev=20, azim=-50)

    # 作業台（テーブル）
    table_x = np.array([100, 450, 450, 100, 100])
    table_y = np.array([-300, -300, 300, 300, -300])
    ax3d.plot(table_x, table_y, np.zeros(5), color="#445588", linewidth=0.8, alpha=0.5)
    ax3d.plot_surface(
        np.array([[100, 450], [100, 450]]),
        np.array([[-300, -300], [300, 300]]),
        np.zeros((2, 2)),
        alpha=0.08, color="#6688bb"
    )

    # 右: 情報パネル（データ座標を 0〜1 で使用）
    ax_info = fig.add_axes([0.63, 0.05, 0.36, 0.88])
    ax_info.set_facecolor("#0f3460")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    # 静的テキスト
    ax_info.text(0.5, 0.97, "RAI エージェント ログ",
                 ha="center", va="top", color="white",
                 fontsize=11, fontweight="bold")
    ax_info.plot([0.02, 0.98], [0.93, 0.93], color="#445588", linewidth=0.8)

    # ログテキスト（更新用）
    log_text = ax_info.text(0.05, 0.89, "",
                            ha="left", va="top", color="#a0e7e5",
                            fontsize=8.5, fontfamily="monospace",
                            linespacing=1.7)

    # タスクラベル
    task_label = ax_info.text(0.5, 0.25, "",
                              ha="center", va="center", color="white",
                              fontsize=11, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.5",
                                        facecolor="#1a1a2e", edgecolor="#e94560", linewidth=2))

    # プログレスバー（ax_info.plot で描画）
    ax_info.text(0.05, 0.12, "進捗:", ha="left", va="center",
                 color="#aaaacc", fontsize=8)
    ax_info.plot([0.05, 0.95], [0.095, 0.095], color="#334466", linewidth=14,
                 solid_capstyle="butt")
    progress_bar, = ax_info.plot([0.05, 0.05], [0.095, 0.095], color="#e94560",
                                  linewidth=14, solid_capstyle="butt")

    # フレームカウンタ
    frame_counter = ax_info.text(0.95, 0.04, "",
                                  ha="right", va="bottom", color="#666688",
                                  fontsize=7)

    # ============================================================
    # アームのプロットオブジェクト（更新用）
    # ============================================================
    arm_line,      = ax3d.plot([], [], [], color="#4ec9b0", linewidth=4,
                               solid_capstyle="round", zorder=5)
    joint_scatter  = ax3d.scatter([], [], [], c="#f0a500", s=80, zorder=6, depthshade=False)
    grip_l,        = ax3d.plot([], [], [], color="#e94560", linewidth=3, zorder=7)
    grip_r,        = ax3d.plot([], [], [], color="#e94560", linewidth=3, zorder=7)
    obj_scatter    = ax3d.scatter([], [], [], c="#ff6b6b", s=300, marker="s",
                                   zorder=8, depthshade=False, edgecolors="#ffaaaa", linewidths=1)
    drop_marker    = ax3d.scatter(*DROP_POSITION, c="#44ff88", s=200, marker="^",
                                   alpha=0.5, zorder=4, depthshade=False)
    tcp_trail_line, = ax3d.plot([], [], [], color="#334488", linewidth=1,
                                alpha=0.5, linestyle="--", zorder=3)

    # TCPの軌跡バッファ
    trail_x, trail_y, trail_z = [], [], []
    LOG_LINES = []

    # ============================================================
    # 更新関数
    # ============================================================
    def update(frame_idx):
        nonlocal trail_x, trail_y, trail_z, LOG_LINES

        tcp_x, tcp_y, tcp_z, grip_open, label = frames_data[frame_idx]

        # アーム関節位置の計算
        segments = compute_arm_segments(tcp_x, tcp_y, tcp_z)
        xs = [s[0] for s in segments]
        ys = [s[1] for s in segments]
        zs = [s[2] for s in segments]

        # アームライン更新
        arm_line.set_data(xs, ys)
        arm_line.set_3d_properties(zs)

        # 関節マーカー更新
        joint_scatter._offsets3d = (xs[1:-1], ys[1:-1], zs[1:-1])

        # グリッパー更新
        tcp = np.array([tcp_x, tcp_y, tcp_z])
        waist = np.arctan2(tcp_y, tcp_x)
        f1, f2 = compute_gripper(tcp, waist, grip_open)
        wrist = segments[-2]
        grip_l.set_data([wrist[0], f1[0]], [wrist[1], f1[1]])
        grip_l.set_3d_properties([wrist[2], f1[2]])
        grip_r.set_data([wrist[0], f2[0]], [wrist[1], f2[1]])
        grip_r.set_3d_properties([wrist[2], f2[2]])

        # 物体位置更新
        obj_pos = get_object_position(frame_idx, tcp)
        obj_scatter._offsets3d = ([obj_pos[0]], [obj_pos[1]], [obj_pos[2]])

        # TCP軌跡更新
        trail_x.append(tcp_x)
        trail_y.append(tcp_y)
        trail_z.append(tcp_z)
        if len(trail_x) > 150:
            trail_x = trail_x[-150:]
            trail_y = trail_y[-150:]
            trail_z = trail_z[-150:]
        tcp_trail_line.set_data(trail_x, trail_y)
        tcp_trail_line.set_3d_properties(trail_z)

        # ログ更新
        seg_idx = frame_idx // FRAMES_PER_SEGMENT
        seg_progress = (frame_idx % FRAMES_PER_SEGMENT) / FRAMES_PER_SEGMENT

        if seg_idx < len(KEYFRAMES) - 1:
            new_kf = KEYFRAMES[seg_idx + 1]
            log_entry = (
                f"[Tool→] move_to_pose\n"
                f"  x={new_kf[0]:.0f}, y={new_kf[1]:.0f}, z={new_kf[2]:.0f}\n"
                f"[←Tool] 移動中... {seg_progress*100:.0f}%\n"
                f"[Tool→] control_gripper\n"
                f"  position={new_kf[3]:.1f}\n"
                f"[←Tool] グリッパー制御完了\n"
            )
            if not LOG_LINES or LOG_LINES[-1] != log_entry:
                LOG_LINES.append(log_entry)
                if len(LOG_LINES) > 4:
                    LOG_LINES = LOG_LINES[-4:]

        log_text.set_text("\n".join(LOG_LINES[-4:]) if LOG_LINES else "")

        # タスクラベル更新
        task_label.set_text(label)

        # プログレスバー更新
        progress = (frame_idx + 1) / total_frames
        progress_bar.set_xdata([0.05, 0.05 + 0.90 * progress])
        progress_bar.set_ydata([0.095, 0.095])

        # フレームカウンタ
        t_sec = frame_idx / FPS
        frame_counter.set_text(f"{t_sec:.1f}s / {total_frames/FPS:.1f}s")

        return (arm_line, joint_scatter, grip_l, grip_r,
                obj_scatter, tcp_trail_line, log_text,
                task_label, progress_bar, frame_counter)

    # ============================================================
    # アニメーション生成・保存
    # ============================================================
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 // FPS,
        blit=False,
    )
    return fig, ani, total_frames


def save_mp4(ani, output_path, fps, total_frames):
    """ffmpegでMP4として保存"""
    print(f"[INFO] MP4 エンコード中: {output_path}")
    print(f"       フレーム数: {total_frames}  / {fps}fps  "
          f"({total_frames/fps:.1f}秒)")

    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"title": "RAI x UFACTORY Lite6 Demo", "artist": "RAI Agent"},
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-preset", "fast"],
    )
    ani.save(output_path, writer=writer, dpi=120)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] 保存完了: {output_path} ({size_mb:.1f} MB)")


def save_gif(ani, output_path, fps, total_frames):
    """Pillow で GIF として保存（ffmpeg が使えない場合の代替）"""
    print(f"[INFO] GIF エンコード中: {output_path}")
    print(f"       フレーム数: {total_frames}  / {fps}fps  "
          f"({total_frames/fps:.1f}秒)")

    writer = animation.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer, dpi=80)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] 保存完了: {output_path} ({size_mb:.1f} MB)")


# ============================================================
# メイン
# ============================================================

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    mp4_path = os.path.join(output_dir, "rai_lite6_demo.mp4")
    gif_path = os.path.join(output_dir, "rai_lite6_demo.gif")

    print("=" * 56)
    print("  RAI × UFACTORY Lite 6 デモアニメーション生成")
    print("=" * 56)

    fig, ani, total_frames = create_animation()

    # ffmpeg が使えるか確認してMP4またはGIFで保存
    ffmpeg_ok = animation.FFMpegWriter.isAvailable()

    if ffmpeg_ok:
        save_mp4(ani, mp4_path, FPS, total_frames)
    else:
        print("[WARN] ffmpeg が見つかりません → GIF形式で保存します")
        print("       MP4で保存するには ffmpeg をインストールしてください:")
        print("         Windows: https://ffmpeg.org/download.html")
        print("         または: winget install ffmpeg")
        try:
            save_gif(ani, gif_path, fps=15, total_frames=total_frames)
        except ImportError:
            print("[ERROR] Pillow も見つかりません。pip install pillow を実行してください")
            sys.exit(1)

    plt.close(fig)
    print("\n完了しました。")


if __name__ == "__main__":
    main()
