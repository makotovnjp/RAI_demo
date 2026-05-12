# RAI × UFACTORY Lite 6 Demo

RAI Framework + ROS2 + UFACTORY Lite 6 による自然言語ロボット自律操作デモ

## ファイル構成

```
RAI_demo/
├── README.md
├── rai_lite6_agent/
│   ├── rai_lite6_agent.py        ← ★ メインエージェント
│   ├── pick_and_place_demo.py    ← VLM統合ピック＆プレース
│   └── requirements.txt
├── generate_demo_video.py        ← ★ 3Dアニメ → MP4生成
├── rai_lite6_demo.mp4            ← 生成済みデモ動画
├── wix_blog_note_style.html      ← Wixブログ記事（note.comスタイル）
├── wix_blog_post.html            ← Wixブログ記事（リッチ版）
├── blog_ros2_rai_ufactory.md     ← ブログ全文（Markdown）
└── video_script.md               ← 動画撮影スクリプト
```

## クイックスタート

```bash
git clone https://github.com/makotovnjp/RAI_demo.git
cd RAI_demo

pip install -r rai_lite6_agent/requirements.txt
export ANTHROPIC_API_KEY=your-key-here

# デモ動画を生成
python generate_demo_video.py

# エージェント起動（シミュレーションモード）
RAI_SIM=1 python rai_lite6_agent/rai_lite6_agent.py
```

## 参考リンク

| リソース | URL |
|---------|-----|
| RAI Framework | https://github.com/RobotecAI/rai |
| RAI 論文 (arXiv) | https://arxiv.org/abs/2505.07532 |
| RAI ドキュメント | https://robotecai.github.io/rai/ |
| xarm_ros2 | https://github.com/xArm-Developer/xarm_ros2 |
| xArm Python SDK | https://pypi.org/project/xarm-python-sdk/ |
