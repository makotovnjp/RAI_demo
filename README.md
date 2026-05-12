# ROS2 × RAI × UFACTORY Lite 6 — 成果物一覧

## ファイル構成

```
ros2_rai_ufactory/
├── README.md                         ← このファイル
│
├── wix_blog_note_style.html          ← ★ Wix貼り付け用ブログ（note.comスタイル）
├── wix_blog_post.html                ← Wix貼り付け用ブログ（リッチスタイル版）
├── blog_ros2_rai_ufactory.md         ← ブログ全文（Markdown版）
│
├── video_script.md                   ← 動画録画スクリプト・YouTube説明文
│
└── rai_lite6_agent/
    ├── rai_lite6_agent.py            ← ★ メインエージェント（実行エントリポイント）
    ├── pick_and_place_demo.py        ← VLM統合ピック＆プレースデモ
    └── requirements.txt              ← pip依存パッケージ
```

## Wixへの貼り付け手順

1. `wix_blog_note_style.html` をテキストエディタで開く
2. 全文コピー（Ctrl+A → Ctrl+C）
3. Wix Editor → ブログ記事編集画面を開く
4. テキストブロックの「+」→「HTMLを埋め込む」を選択
5. 貼り付け（Ctrl+V）→ 適用

## サンプルコード実行手順

```bash
# 環境構築
pip install -r rai_lite6_agent/requirements.txt

# APIキー設定
export ANTHROPIC_API_KEY=your-key-here

# シミュレーションモード（実機なしで動作確認）
RAI_SIM=1 python3 rai_lite6_agent/rai_lite6_agent.py

# 実機モード（Lite6のIPを指定してドライバ起動後）
ros2 launch xarm_api lite6_driver.launch.py robot_ip:=192.168.1.xxx
python3 rai_lite6_agent/rai_lite6_agent.py

# デモシナリオ自動実行
RAI_DEMO=1 RAI_SIM=1 python3 rai_lite6_agent/rai_lite6_agent.py

# VLM統合ピック＆プレース
RAI_SIM=1 python3 rai_lite6_agent/pick_and_place_demo.py
```

## 動画撮影について

`video_script.md` に以下が含まれています：
- シーン構成（5シーン・約5分）
- 各シーンのナレーション・テロップ案
- 録画設定チェックリスト
- YouTube動画説明文（コピペ用）
- 編集メモ

## 参考リンク

| リソース | URL |
|---------|-----|
| RAI GitHub | https://github.com/RobotecAI/rai |
| RAI 論文 | https://arxiv.org/abs/2505.07532 |
| RAI ドキュメント | https://robotecai.github.io/rai/ |
| RAI Manipulation Demo | https://github.com/RobotecAI/rai-manipulation-demo |
| xarm_ros2 | https://github.com/xArm-Developer/xarm_ros2 |
| xArm Python SDK | https://pypi.org/project/xarm-python-sdk/ |
