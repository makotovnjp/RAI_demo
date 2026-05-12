"""
RAI + UFACTORY Lite 6 自律操作エージェント
========================================
ROS2 Kilted Kaiju + RAI Framework pattern + Claude claude-sonnet-4-6

必要パッケージ:
  pip install anthropic numpy opencv-python-headless

ROS2パッケージ:
  - xarm_ros2 (xArm-Developer/xarm_ros2)
  - realsense2_camera (オプション: Intel RealSenseカメラ使用時)

起動手順:
  1. ros2 launch xarm_api lite6_driver.launch.py robot_ip:=<IP>
  2. export ANTHROPIC_API_KEY=your-key
  3. python3 rai_lite6_agent.py
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import numpy as np
import anthropic
import base64
import json
import time
import cv2
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from xarm_msgs.srv import MoveCartesian, SetInt16
except ImportError:
    # xarm_msgs が未インストール環境向けのスタブ
    class MoveCartesian:
        class Request:
            def __init__(self):
                self.pose = [0.0] * 6
                self.speed = 100.0
                self.acc = 500.0
                self.mvtime = 0.0
                self.wait = True

    class SetInt16:
        class Request:
            def __init__(self, data=0):
                self.data = data


# ============================================================
# データモデル
# ============================================================

@dataclass
class Pose:
    x: float = 300.0   # mm
    y: float = 0.0     # mm
    z: float = 300.0   # mm
    roll: float = 3.14159   # rad (π = ツール下向き)
    pitch: float = 0.0      # rad
    yaw: float = 0.0        # rad


HOME_POSE = Pose(x=300.0, y=0.0, z=350.0, roll=3.14159, pitch=0.0, yaw=0.0)

WORKSPACE_LIMITS = {
    "x": (100.0, 450.0),
    "y": (-300.0, 300.0),
    "z": (50.0, 500.0),
}


# ============================================================
# ROS2 Lite6 コネクタ
# ============================================================

class Lite6Connector(Node):
    """
    UFACTORY Lite 6 の ROS2 インターフェース

    対応サービス:
      /ufactory/motion_enable   (SetInt16)  - モーション有効化
      /ufactory/set_mode        (SetInt16)  - 制御モード設定
      /ufactory/set_state       (SetInt16)  - ロボット状態設定
      /ufactory/set_position    (MoveCartesian) - デカルト移動
      /ufactory/set_gripper_position (SetInt16) - グリッパー制御
    """

    SERVICE_TIMEOUT = 5.0

    def __init__(self, use_sim: bool = False):
        super().__init__("rai_lite6_connector")
        self.use_sim = use_sim

        # サービスクライアント
        self._motion_enable = self.create_client(SetInt16, "/ufactory/motion_enable")
        self._set_mode = self.create_client(SetInt16, "/ufactory/set_mode")
        self._set_state = self.create_client(SetInt16, "/ufactory/set_state")
        self._set_position = self.create_client(MoveCartesian, "/ufactory/set_position")
        self._set_gripper = self.create_client(SetInt16, "/ufactory/set_gripper_position")

        # 状態管理
        self._joint_state: Optional[JointState] = None
        self._latest_image: Optional[np.ndarray] = None
        self._gripper_position: float = 0.0
        self._bridge = CvBridge()

        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(Image, "/camera/image_raw", self._image_cb, 10)

        self.get_logger().info(f"Lite6Connector initialized (sim={use_sim})")

    def _joint_cb(self, msg: JointState):
        self._joint_state = msg

    def _image_cb(self, msg: Image):
        self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    def initialize(self) -> bool:
        """ロボットの初期化シーケンス（Enable → Mode0 → State0）"""
        if self.use_sim:
            self.get_logger().info("[SIM] Robot initialized (simulation mode)")
            return True

        steps = [
            (self._motion_enable, SetInt16.Request(data=8), "モーション有効化"),
            (self._set_mode, SetInt16.Request(data=0), "モード0(Position)設定"),
            (self._set_state, SetInt16.Request(data=0), "State0(Ready)設定"),
        ]

        for client, req, label in steps:
            result = self._call_service(client, req)
            if result is None:
                self.get_logger().error(f"初期化失敗: {label}")
                return False
            time.sleep(0.5)
            self.get_logger().info(f"OK: {label}")

        return True

    def _validate_pose(self, pose: Pose) -> tuple[bool, str]:
        """作業空間の範囲チェック"""
        for axis, (lo, hi) in WORKSPACE_LIMITS.items():
            val = getattr(pose, axis)
            if not (lo <= val <= hi):
                return False, f"{axis}={val:.1f}mm は作業範囲外({lo}〜{hi}mm)"
        return True, "OK"

    def move_to_pose(self, pose: Pose, speed: float = 100.0, wait: bool = True) -> tuple[bool, str]:
        """
        デカルト空間での目標位置への移動

        Args:
            pose: 目標TCP姿勢
            speed: 移動速度 [mm/s]
            wait: 移動完了待ち

        Returns:
            (成功フラグ, メッセージ)
        """
        valid, msg = self._validate_pose(pose)
        if not valid:
            return False, f"安全チェック失敗: {msg}"

        if self.use_sim:
            time.sleep(0.3)
            return True, f"[SIM] 移動完了: ({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f})"

        req = MoveCartesian.Request()
        req.pose = [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]
        req.speed = float(speed)
        req.acc = 500.0
        req.mvtime = 0.0
        req.wait = wait

        result = self._call_service(self._set_position, req)
        if result is None:
            return False, "サービス呼び出し失敗"

        success = result.ret == 0
        return success, f"移動{'成功' if success else '失敗'} ret={result.ret}"

    def control_gripper(self, position: float) -> tuple[bool, str]:
        """
        グリッパー制御

        Args:
            position: 0.0(完全閉) 〜 1.0(完全開)

        Returns:
            (成功フラグ, メッセージ)
        """
        position = float(np.clip(position, 0.0, 1.0))
        pos_int = int(position * 850)

        if self.use_sim:
            self._gripper_position = position
            state = "開" if position > 0.5 else "閉"
            return True, f"[SIM] グリッパー{state}: {position:.2f}"

        req = SetInt16.Request(data=pos_int)
        result = self._call_service(self._set_gripper, req)
        if result is None:
            return False, "グリッパーサービス呼び出し失敗"

        self._gripper_position = position
        return True, f"グリッパー位置={pos_int} (={position:.2f})"

    def get_joint_angles(self) -> list[float]:
        """現在の関節角度を返す [rad x 6]"""
        if self._joint_state is not None:
            return list(self._joint_state.position)
        return [0.0] * 6

    def get_camera_image(self) -> Optional[np.ndarray]:
        """最新のカメラ画像を取得"""
        return self._latest_image.copy() if self._latest_image is not None else None

    def _call_service(self, client, request):
        if not client.wait_for_service(timeout_sec=self.SERVICE_TIMEOUT):
            self.get_logger().error(f"Service not available: {client.srv_name}")
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.SERVICE_TIMEOUT)
        return future.result() if future.done() else None


# ============================================================
# RAI ツール定義
# ============================================================

class Lite6Tools:
    """Anthropic Claude tool use形式のロボット操作ツール群"""

    def __init__(self, connector: Lite6Connector):
        self.robot = connector
        self._log: list[dict] = []

    @property
    def execution_log(self) -> list[dict]:
        return self._log

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "move_to_pose",
                "description": (
                    "ロボットアームのTCP（ツール中心点）を指定デカルト座標に移動する。"
                    "作業空間: X=100〜450mm, Y=-300〜300mm, Z=50〜500mm。"
                    "Rollは通常π(3.14159)で下向き把持。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X座標 (mm)"},
                        "y": {"type": "number", "description": "Y座標 (mm)"},
                        "z": {"type": "number", "description": "Z座標 (mm)"},
                        "roll":  {"type": "number", "description": "Roll角 (rad), default=π"},
                        "pitch": {"type": "number", "description": "Pitch角 (rad), default=0"},
                        "yaw":   {"type": "number", "description": "Yaw角 (rad), default=0"},
                        "speed": {"type": "number", "description": "速度 mm/s (default=100)"},
                    },
                    "required": ["x", "y", "z"],
                },
            },
            {
                "name": "control_gripper",
                "description": (
                    "グリッパーを制御する。"
                    "position=0.0で完全閉じ、1.0で完全開き。"
                    "物体をつかむ前に1.0に開き、つかんだら0.1〜0.3に閉じる。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "number", "description": "0.0(閉)〜1.0(開)"}
                    },
                    "required": ["position"],
                },
            },
            {
                "name": "get_camera_image",
                "description": "ハンドカメラから現在の画像を取得して作業台の状況を確認する",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_joint_angles",
                "description": "現在の6軸関節角度（rad）を取得する",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "move_home",
                "description": "ロボットをホームポジション（X=300, Y=0, Z=350）に戻す",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "wait_seconds",
                "description": "指定秒数だけ待機する（把持安定化などに使用）",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "number", "description": "待機秒数 (0.1〜10.0)"}
                    },
                    "required": ["seconds"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """ツールを実行し、結果文字列を返す"""
        start = time.time()
        result = self._dispatch(tool_name, tool_input)
        elapsed = time.time() - start

        self._log.append({
            "tool": tool_name,
            "input": tool_input,
            "result": result,
            "elapsed_s": round(elapsed, 3),
        })
        return result

    def _dispatch(self, tool_name: str, inp: dict) -> str:
        if tool_name == "move_to_pose":
            pose = Pose(
                x=float(inp.get("x", 300.0)),
                y=float(inp.get("y", 0.0)),
                z=float(inp.get("z", 200.0)),
                roll=float(inp.get("roll", 3.14159)),
                pitch=float(inp.get("pitch", 0.0)),
                yaw=float(inp.get("yaw", 0.0)),
            )
            speed = float(inp.get("speed", 100.0))
            ok, msg = self.robot.move_to_pose(pose, speed=speed)
            return msg

        elif tool_name == "control_gripper":
            pos = float(inp.get("position", 0.5))
            ok, msg = self.robot.control_gripper(pos)
            return msg

        elif tool_name == "get_camera_image":
            img = self.robot.get_camera_image()
            if img is None:
                return "カメラ画像なし（未接続またはトピック未受信）"
            h, w = img.shape[:2]
            brightness = float(np.mean(img))
            # 画像をファイル保存（タイムスタンプ付き）
            fname = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, img)
            return f"画像取得: {w}x{h}px, 輝度={brightness:.1f}, 保存先={fname}"

        elif tool_name == "get_joint_angles":
            angles = self.robot.get_joint_angles()
            deg = [f"J{i+1}={np.degrees(a):.1f}°" for i, a in enumerate(angles)]
            return "関節角度: " + ", ".join(deg)

        elif tool_name == "move_home":
            ok, msg = self.robot.move_to_pose(HOME_POSE, speed=50.0)
            return f"ホームへ移動: {msg}"

        elif tool_name == "wait_seconds":
            secs = float(np.clip(inp.get("seconds", 1.0), 0.1, 10.0))
            time.sleep(secs)
            return f"{secs:.1f}秒待機完了"

        else:
            return f"未知のツール: {tool_name}"


# ============================================================
# RAI エージェント
# ============================================================

SYSTEM_PROMPT = """あなたはUFACTORY Lite 6ロボットアームを操作するAIエージェントです。

利用できるツールを組み合わせて、ユーザーの指示を自律的に実行してください。

## ロボット仕様
- 6軸ロボットアーム、可搬0.5kg、リーチ440mm
- 作業空間: X=100〜450mm, Y=-300〜300mm, Z=50〜500mm（ロボットベース原点基準）
- グリッパー: 0.0(完全閉) 〜 1.0(完全開)、把持時は0.1〜0.3推奨
- 通常姿勢: Roll=π(下向き把持)、Pitch=0、Yaw=0

## 安全ルール（厳守）
1. Z座標は常に50mm以上を維持する（テーブル衝突防止）
2. 物体をつかむ前にグリッパーを開く(1.0)
3. 移動速度: 通常≤100mm/s、精密作業≤50mm/s
4. 不明な状況はget_camera_imageで確認してから行動する

## 作業手順の基本
1. カメラで状況確認
2. ホームポジションから安全な高さで作業位置上方へ移動
3. ゆっくり降下してアプローチ
4. グリッパー操作
5. 持ち上げてから移送
6. 降下して解放
7. ホームへ帰還

各ツール呼び出しの結果を確認しながら進め、エラーが発生したら安全側の動作をとってください。
タスク完了後は実行結果を日本語で要約して報告してください。"""


class RAILite6Agent:
    """
    RAIパターンのReActエージェント
    - LLMがツールを自律的に選択・実行
    - Anthropic Claude claude-sonnet-4-6をバックエンドに使用
    """

    def __init__(self, connector: Lite6Connector, api_key: str, model: str = "claude-sonnet-4-6"):
        self.tools_handler = Lite6Tools(connector)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.conversation_history: list[dict] = []

    def run_task(self, task: str, max_iterations: int = 25, verbose: bool = True) -> str:
        """
        自然言語タスクをReActループで実行

        Args:
            task: 自然言語タスク記述
            max_iterations: 最大ツール呼び出し回数
            verbose: ログ出力フラグ

        Returns:
            タスク実行結果サマリー
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[RAI] タスク: {task}")
            print(f"{'='*60}")

        self.conversation_history.append({"role": "user", "content": task})

        for i in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools_handler.get_tool_definitions(),
                messages=self.conversation_history,
            )

            text_parts = []
            tool_uses = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    if verbose and block.text.strip():
                        print(f"[Agent] {block.text.strip()}")
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # 終了条件
            if response.stop_reason == "end_turn" or not tool_uses:
                self.conversation_history.append(
                    {"role": "assistant", "content": response.content}
                )
                return "\n".join(text_parts)

            # ツール実行
            self.conversation_history.append(
                {"role": "assistant", "content": response.content}
            )

            tool_results = []
            for tu in tool_uses:
                if verbose:
                    print(f"\n[Tool→] {tu.name}({json.dumps(tu.input, ensure_ascii=False)})")

                result = self.tools_handler.execute_tool(tu.name, tu.input)

                if verbose:
                    print(f"[←Tool] {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })

            self.conversation_history.append({"role": "user", "content": tool_results})

        return f"最大イテレーション({max_iterations})に達しました"

    def print_execution_log(self):
        """ツール実行ログを表示"""
        log = self.tools_handler.execution_log
        print(f"\n{'='*60}")
        print(f"実行ログ ({len(log)} ツール呼び出し)")
        print(f"{'='*60}")
        for entry in log:
            print(f"  {entry['tool']:25s} → {entry['result'][:60]}... ({entry['elapsed_s']}s)")

    def interactive_session(self):
        """インタラクティブREPLセッション"""
        print("\n" + "="*60)
        print("  RAI × UFACTORY Lite 6  インタラクティブセッション")
        print("  終了: 'exit' または Ctrl+C")
        print("="*60)

        while True:
            try:
                cmd = input("\n[You] ").strip()
                if cmd.lower() in ("exit", "quit", "終了"):
                    print("セッション終了")
                    break
                if cmd == "log":
                    self.print_execution_log()
                    continue
                if not cmd:
                    continue

                result = self.run_task(cmd)
                print(f"\n[完了]\n{result}")

            except KeyboardInterrupt:
                print("\n中断しました")
                break


# ============================================================
# メインエントリポイント
# ============================================================

def main():
    use_sim = os.environ.get("RAI_SIM", "0") == "1"

    rclpy.init()
    connector = Lite6Connector(use_sim=use_sim)

    if not connector.initialize():
        print("ロボット初期化失敗。環境変数 RAI_SIM=1 でシミュレーションモード使用可能")
        rclpy.shutdown()
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY 環境変数を設定してください")
        rclpy.shutdown()
        return

    agent = RAILite6Agent(connector, api_key=api_key)

    if os.environ.get("RAI_DEMO", "0") == "1":
        # デモシナリオ自動実行
        demo_tasks = [
            "ホームポジションに移動して、グリッパーを一度開いてから閉じてください",
            "X=300, Y=0, Z=150mmに移動してグリッパーを開き、物体をつかんでZ=250mmに持ち上げてください",
            "X=200, Y=150, Z=200mmに移動して物体を置いてホームに戻ってください",
        ]
        for i, task in enumerate(demo_tasks, 1):
            print(f"\n--- デモタスク {i}/{len(demo_tasks)} ---")
            agent.run_task(task)
            time.sleep(1.5)

        agent.print_execution_log()
    else:
        # インタラクティブモード
        agent.interactive_session()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
