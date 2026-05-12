"""
RAI VLM統合 ピック＆プレース デモ
===================================
カメラ画像をVLM(Claude claude-sonnet-4-6)で解析し、
認識した物体を自律的にピックアップして配置するデモ

使用方法:
  python3 pick_and_place_demo.py
"""

import anthropic
import base64
import cv2
import numpy as np
import json
import time
import os
from typing import Optional

# 同ディレクトリのメインエージェントをインポート
import sys
sys.path.insert(0, os.path.dirname(__file__))
from rai_lite6_agent import RAILite6Agent, Lite6Connector, Pose


# ============================================================
# VLM 物体検出
# ============================================================

def image_to_base64_jpeg(image: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.standard_b64encode(buf).decode("utf-8")


def detect_objects_with_vlm(
    client: anthropic.Anthropic,
    image: np.ndarray,
    model: str = "claude-sonnet-4-6"
) -> list[dict]:
    """
    VLMで画像から物体を検出する

    Returns:
        [{"name": str, "color": str, "image_position": str, "confidence": float}]
    """
    img_b64 = image_to_base64_jpeg(image)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                },
                {
                    "type": "text",
                    "text": (
                        "画像上の把持可能な物体をすべて検出してください。"
                        "以下のJSONのみを返してください（説明文は不要）:\n"
                        '{"objects": ['
                        '{"name": "物体名", "color": "色", '
                        '"image_position": "左上|左中|左下|中上|中央|中下|右上|右中|右下"}'
                        "]}"
                    ),
                },
            ],
        }],
    )

    try:
        text = response.content[0].text.strip()
        # JSONブロックが含まれる場合に対応
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        return data.get("objects", [])
    except (json.JSONDecodeError, IndexError, KeyError):
        return []


# ============================================================
# 画像位置 → ロボット座標の変換
# ============================================================

# 画像位置ラベルから推定ロボット座標へのマッピング（mm）
IMAGE_POS_TO_ROBOT = {
    "左上":  Pose(x=350, y=180, z=150),
    "左中":  Pose(x=300, y=200, z=150),
    "左下":  Pose(x=230, y=180, z=150),
    "中上":  Pose(x=370, y=0,   z=150),
    "中央":  Pose(x=300, y=0,   z=150),
    "中下":  Pose(x=230, y=0,   z=150),
    "右上":  Pose(x=350, y=-180, z=150),
    "右中":  Pose(x=300, y=-200, z=150),
    "右下":  Pose(x=230, y=-180, z=150),
}

# ドロップゾーン（物体を置く場所）
DROP_ZONES = {
    "zone_A": Pose(x=200, y=220, z=150),   # 左側ゾーン
    "zone_B": Pose(x=200, y=0,   z=150),   # 中央ゾーン
    "zone_C": Pose(x=200, y=-220, z=150),  # 右側ゾーン
}


# ============================================================
# ピック＆プレース プランナー
# ============================================================

class PickAndPlacePlanner:
    """
    検出物体に対してピック＆プレース計画を生成するプランナー

    計画:
      1. ホームへ
      2. アプローチ高 (Z=250) へ移動
      3. ゆっくり降下 (Z=pick_z)
      4. グリッパー閉じ
      5. 持ち上げ (Z=250)
      6. ドロップゾーン上方へ
      7. 降下 (Z=drop_z)
      8. グリッパー開き
      9. ホームへ帰還
    """

    APPROACH_Z = 250.0  # mm - アプローチ高
    PICK_Z = 80.0       # mm - 把持高
    DROP_Z = 120.0      # mm - 解放高
    PICK_SPEED = 50.0   # mm/s
    MOVE_SPEED = 120.0  # mm/s

    def generate_plan(self, obj: dict, drop_zone_key: str = "zone_A") -> list[tuple[str, dict]]:
        """
        1物体分のピック＆プレース計画を生成

        Returns:
            [(tool_name, tool_input), ...]
        """
        pos_label = obj.get("image_position", "中央")
        pick_pose = IMAGE_POS_TO_ROBOT.get(pos_label, IMAGE_POS_TO_ROBOT["中央"])
        drop_pose = DROP_ZONES.get(drop_zone_key, DROP_ZONES["zone_B"])

        plan = [
            # ホームへ
            ("move_home", {}),

            # アプローチ高へ
            ("move_to_pose", {
                "x": pick_pose.x, "y": pick_pose.y, "z": self.APPROACH_Z,
                "roll": 3.14159, "speed": self.MOVE_SPEED,
            }),

            # グリッパーを開く
            ("control_gripper", {"position": 1.0}),

            # 把持高へ降下
            ("move_to_pose", {
                "x": pick_pose.x, "y": pick_pose.y, "z": self.PICK_Z,
                "roll": 3.14159, "speed": self.PICK_SPEED,
            }),

            # 把持
            ("wait_seconds", {"seconds": 0.5}),
            ("control_gripper", {"position": 0.2}),
            ("wait_seconds", {"seconds": 0.5}),

            # 持ち上げ
            ("move_to_pose", {
                "x": pick_pose.x, "y": pick_pose.y, "z": self.APPROACH_Z,
                "roll": 3.14159, "speed": self.PICK_SPEED,
            }),

            # ドロップゾーン上方へ
            ("move_to_pose", {
                "x": drop_pose.x, "y": drop_pose.y, "z": self.APPROACH_Z,
                "roll": 3.14159, "speed": self.MOVE_SPEED,
            }),

            # 降下
            ("move_to_pose", {
                "x": drop_pose.x, "y": drop_pose.y, "z": self.DROP_Z,
                "roll": 3.14159, "speed": self.PICK_SPEED,
            }),

            # 解放
            ("control_gripper", {"position": 1.0}),
            ("wait_seconds", {"seconds": 0.3}),

            # ホームへ帰還
            ("move_home", {}),
        ]
        return plan


# ============================================================
# VLM統合ピック＆プレース デモ
# ============================================================

class VLMPickAndPlaceDemo:
    """VLM（Claude claude-sonnet-4-6）と連携したピック＆プレースデモ"""

    DROP_ZONE_KEYS = ["zone_A", "zone_B", "zone_C"]

    def __init__(self, connector: Lite6Connector, api_key: str):
        self.connector = connector
        self.client = anthropic.Anthropic(api_key=api_key)
        self.agent = RAILite6Agent(connector, api_key=api_key)
        self.planner = PickAndPlacePlanner()

    def run(self, max_objects: int = 5):
        """メインデモループ"""
        print("\n" + "="*60)
        print("  VLM統合 ピック＆プレース デモ")
        print("="*60)

        # カメラ画像取得
        print("\n[1/4] カメラ画像を取得中...")
        img = self.connector.get_camera_image()
        if img is None:
            print("  カメラ画像なし → RAIエージェントに委任")
            task = (
                "カメラで作業台の状況を確認し、見えているすべての物体を"
                "1つずつピックアップして別の場所に移動してください。"
                "各動作を実行しながら状況を報告してください。"
            )
            result = self.agent.run_task(task, max_iterations=50)
            print(f"\n[完了]\n{result}")
            return

        # VLMで物体検出
        print("[2/4] VLMで物体を検出中...")
        objects = detect_objects_with_vlm(self.client, img)
        if not objects:
            print("  物体が検出されませんでした")
            return

        print(f"  検出物体 ({len(objects)}個):")
        for obj in objects:
            print(f"    - {obj.get('name', '不明')} [{obj.get('color', '?')}] @ {obj.get('image_position', '?')}")

        # ピック＆プレース実行
        print(f"\n[3/4] ピック＆プレース実行 (最大{max_objects}個)...")
        target_objects = objects[:max_objects]

        for i, obj in enumerate(target_objects):
            zone_key = self.DROP_ZONE_KEYS[i % len(self.DROP_ZONE_KEYS)]
            print(f"\n  [{i+1}/{len(target_objects)}] {obj.get('name')} → {zone_key}")

            plan = self.planner.generate_plan(obj, zone_key)
            self._execute_plan(plan)
            time.sleep(0.5)

        # 完了報告
        print("\n[4/4] デモ完了")
        print(f"  処理物体数: {len(target_objects)}")
        self.agent.print_execution_log()

    def _execute_plan(self, plan: list[tuple[str, dict]]) -> bool:
        """計画を順番に実行する"""
        for tool_name, tool_input in plan:
            result = self.agent.tools_handler.execute_tool(tool_name, tool_input)
            print(f"    {tool_name}: {result}")
            if "失敗" in result or "エラー" in result:
                print(f"    ! エラー検出 → ホームへ退避")
                self.agent.tools_handler.execute_tool("move_home", {})
                return False
        return True


# ============================================================
# エントリポイント
# ============================================================

def main():
    import rclpy

    use_sim = os.environ.get("RAI_SIM", "0") == "1"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY 環境変数を設定してください")
        return

    rclpy.init()
    connector = Lite6Connector(use_sim=use_sim)
    connector.initialize()

    demo = VLMPickAndPlaceDemo(connector, api_key=api_key)
    demo.run(max_objects=3)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
