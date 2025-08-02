## FEATURE:

ブラウザ上で動作する文字起こし Web アプリを **Gradio**（Python `gr.Blocks`）で実装します。
ユーザはドラッグ＆ドロップまたはファイル選択で 1 時間程度の大容量音声ファイル（`.mp3`、`.wav`、`.m4a` など）をアップロードし、OpenAI の音声認識 API を利用して文字起こしを取得できます。
モデル ID や言語などのパラメータは UI から設定可能です。単一ユーザを想定しているため認証や共有機能は実装しません。

### Core Functional Requirements

1. **ファイルアップロード UI**

   * ドラッグ＆ドロップ対応の `gr.File` コンポーネントを使用。
   * アップロード進行状況と残り推定時間をリアルタイム表示。
   * 拡張子とファイルサイズを検証し、500 MB 超の場合は警告。

2. **文字起こし処理**

   * OpenAI `/v1/audio/transcriptions` を呼び出す。
   * モデル（`whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe`）をドロップダウンで選択。
   * 言語コード（IETF BCP‑47、例:`en`, `ja`）を指定。「auto」（自動判定）をデフォルトに設定。
   * 温度やタイムスタンプ出力などの高度なパラメータは「詳細設定」アコーディオンにまとめる。
   * **25 MB の API サイズ制限** に対応するため、音声ファイルをユーザが選択した **N 分 (1–10)** ごとに分割（`util.split_audio`) し、隣接セクションと **2 秒オーバーラップ** させてチャンク (`chunk_01.mp3`, `chunk_02.mp3`, …) を生成する。

     * **処理シーケンス**:
       1. **スプリット** — pydub で全チャンクをローカル生成。
       2. **チャンクごとに文字起こし** — `transcribe(transcribe.py)` が `yield` で部分結果を返し、`gr.Progress` を更新。
       3. **結果表示パネルをリアルタイム更新** — 各チャンク完了時に Markdown へ追記してユーザが途中経過を確認可能（`gr.Textbox.update(value += new_text)` など）。
       4. **全チャンク終了後に翻訳フェーズへ移行**。
   * **翻訳オプション**: トグルを有効化すると、**全チャンク終了後に文字起こし全文を LLM に一括入力して 1 回で翻訳**（チャンク単位では行わない）。

     * **全文翻訳を採用する意図**

       * チャンク単位で翻訳すると文脈が分断され、訳語のブレ・固有名詞の揺れが発生しやすい。全文をまとめて翻訳することで **一貫性と自然さ** を確保できる。
       * タイムスタンプを保持したままテキスト部のみを翻訳することで、**セクション間の整合性** を保ちながら字幕用途にも転用できる。
       * API 呼び出し回数を 1 回に抑えられるため、**コストとレイテンシを最小化** できる。

       **全文翻訳戦略（推奨） — JSON 構造化方式**

       * テキストを翻訳前に JSON 配列へ変換: `[ {"ts": "00:00:00-00:01:00", "text": "..."}, ... ]` の形でタイムスタンプと本文を分離。
       * LLM へは **`text` フィールドのみ翻訳対象** と指示し、`ts` は変更しないよう明示する。
       * 翻訳後は同じ JSON 構造で受け取り、`text` 部分だけを置換してから元のテキスト形式 `# 00:00:00 --> 00:01:00` に再組み立てする。
       * この方式は **(1) タイムスタンプと本文が明確に分離されるためフォーマット崩れリスクが最小**、**(2) LLM の structured output 機能（JSON 出力）を利用することで出力形式が保証される**(3) 将来に別形式（SRT, VTT）への変換もシンプル** という理由で推奨する。
       * トークン上限を超える場合は JSON を複数チャンクに分割し、順番を保持して LLM へ順次送信後に再結合する。
     * 翻訳完了後に **原文 / 翻訳文** タブへ一括反映し、`transcript.zip` を生成。  **原文 / 翻訳文** タブへ一括反映し、`transcript.zip` を生成。 **原文 / 翻訳文** タブへ一括反映し、`transcript.zip` を生成。

3. **設定パネル**

   * **config.yaml** から `audio_models`・`language_models`・`system_message` を読み込み、UI の選択肢と初期値を生成。
   * **OpenAI API キー**入力欄（必須）を設置し、`gr.BrowserState` で `localStorage` に保存。
   * **音声モデル**: `audio_models` のドロップダウンでユーザが選択。
   * **言語モデル**: `language_models` のドロップダウンでユーザが選択。
   * **System Message**: `gr.Textbox` に表示し、ユーザが編集して上書き可能。編集内容はチャット機能に即時反映し、`localStorage` に保存。
   * 「保存」ボタン: 現在の設定を `localStorage` に書き込み。
   * 「デフォルトにリセット」ボタン: `config.yaml` の既定値に戻す。

4. **結果表示**

   * 文字起こし結果は **MarkdownL** に変換し、`gr.Markdown` で表示。

     * タイムスタンプ `# 00:00:00 --> 00:01:00` を `<span class="timestamp">...</span>` に変換。
     * CSS 例: `.timestamp { font-size: 0.85rem; color: #888; }` を `Blocks(css=...)` で適用し、タイムスタンプを本文より小さく薄いフォントで表示。
   * コピー用ボタンを付与し、クリックで全文をクリップボードへ。
   * **ダウンロード**

     * 翻訳オプションが **オフ** の場合: `transcript.txt` を直接ダウンロード。
     * 翻訳オプションが **オン** の場合: `transcript.zip` を生成してダウンロード。ZIP には以下 2 ファイルを含める。

       * `transcript.txt` — 原文
       * `transcript.<lang>.txt` — 翻訳文（`<lang>` はターゲット言語コード） 各ファイルの内部フォーマットは共通で以下の例に準拠。は共通で以下の例に準拠。

     ```
     # 00:00:00 --> 00:01:00
     最初の1分のトランスクリプト

     # 00:02:00 --> 00:03:00
     次の1分のトランスクリプト
     ```
   * 再生時間・単語数・WPM などの統計情報を表示。
   * 翻訳を有効にした場合は **原文** と **翻訳文** をタブまたはスプリットビューで切替表示し、それぞれ `.txt` ダウンロード対応。

5. **履歴ビュー**

   * 過去ジョブ（ファイル名、日時、長さ、言語、ステータス）をテーブルやカードで一覧表示。
   * 選択すると文字起こし内容とダウンロードリンクを再読み込み。
   * サーバー側 `data/{YYYY-MM-DD}/{job_id}/` に元音声と文字起こしを保存。
   * 過去ジョブと関連データの削除も行えるようにする。

6. **チャット機能**

   * `gr.Chatbot` で OpenAI LLM と対話。
   * 表示中の文字起こしテキストを「コンテキスト」としてメッセージに注入。

     * 送信フロー:

       1. **system**: `config.yaml` の `system_message`。
       2. **user**: コンテキスト全文（または選択部分）。
       3. **user**: 質問本文。
   * `assistant` 応答を受け取り履歴に追加し、次回リクエストに連続して渡す。
   * 「クリア」ボタンで履歴をリセットし、system メッセージのみ再設定。
   * チャットパネルは折りたたみ可能で画面スペースを節約。

7. **UI構成**

   * ルートは `gr.Blocks` にカスタム CSS を適用し、ライト系スタイルとする。
   * **メインレイアウト**: 単一カラム (`gr.Column`) を基本とし、上から下へ以下の順序で配置。
     1. `gr.File` アップロードボックスおよび各コントロール（モデル選択・言語設定・N 分分割・翻訳トグルなど）を `gr.Row` 内にまとめて省スペース化。
     2. `gr.Progress` 表示（アップロード & 処理進捗）。
     3. **処理ログパネル** (`gr.Textbox` or `gr.Markdown`) — **最小限の縦幅**（例: `lines=4` または CSS `max-height: 120px; overflow-y: auto;`）で固定し、自動スクロール。必要なら `gr.Accordion` で折りたたみ可能にして結果表示パネルの領域を確保。
     4. **結果表示パネル** (`gr.HTML` / `gr.Markdown`) — 画面幅いっぱいに広げ、可読性を最大化。
     5. ダウンロードボタン (`gr.Button`)。
     6. **チャットパネル** (`gr.Chatbot`) — ページ最下部に配置し、折りたたみボタンで表示/非表示を切替。\*\* (`gr.Chatbot`) — ページ最下部に配置し、折りたたみボタンで表示/非表示を切替。
   * **設定パネル**: 画面右上にギアアイコン (gr.Button with icon) を配置し、クリックで gr.Modal を表示。モーダル内に API キー入力欄、音声モデル・言語モデルドロップダウン、System Message 編集テキストボックス、保存／リセットボタンをレイアウト（gr.Column 使用）。設定内容は gr.BrowserState で localStorage に保持。
   * **履歴ビュー**: `履歴を表示` ボタンを押すと `gr.Modal` が開き、`gr.Dataframe` または `gr.Dataset` でジョブ履歴を表示。モーダル内で削除・再読み込みが可能。
   * **通知 / アラート**
   * `gr.Alert` は **致命的エラー** のみ固定表示（閉じるまで残る）。
   * 正常終了や軽微な情報は `gr.Notification`（カスタム CSS でトースト風）を右下に数秒間ポップアップし、自動でフェードアウト。
   * これにより常時固定要素を減らし、結果表示パネルの可視領域を確保。

### CONFIG FILE (`config.yaml`)

```yaml
audio_models:
  - whisper-1
  - gpt-4o-mini-transcribe
  - gpt-4o-transcribe

language_models:
  - gpt-4o-mini
  - gpt-4o
  - gpt-4o-speed

system_message: |
  あなたはプロフェッショナルで親切な文字起こしアシスタントです。
  ユーザの要求に簡潔かつ正確に答えてください。
```

> **備考**: `language_models` はチャット機能・翻訳機能で使用する LLM のリスト。必要に応じて追加・変更可能。

### File Structure / Modules

```
project_root/src/
├── app.py           # Gradio UI / routing only
├── transcribe.py     # Whisper / OpenAI 呼び出しと結果統合のみ担当
├── util.py           # pydub での音声分割（N 分チャンク + 2 秒オーバーラップ）
├── llm.py            # OpenAI Chat / 翻訳・要約など汎用 LLM 呼び出し
├── config.yaml       # モデルリスト & system message
└── ...
```

* 全てのソースコードは src 下に配置。
* **app.py** は Gradio コンポーネントとハンドラのみを記述し、実装詳細を `transcribe.py` と `llm.py` に委譲する。
* 共通の例外クラスやユーティリティは `utils.py` などにまとめると保守性が向上。
* 単体テストは `tests/test_transcribe.py`, `tests/test_llm.py` を用意。
* `transcribe.py` と `llm.py` は CLI で各機能を試用するために __main__ ブロックでコマンドラインインターフェースを実装。argparse を使用すること。

## EXAMPLES:

* `examples/transcribe.py` – 文字起こしサンプル
* `examples/llm.py` - chat, structured output サンプル

---

## DOCUMENTATION:

* Gradio Blocks
  [https://www.gradio.app/guides/blocks-and-event-listeners](https://www.gradio.app/guides/blocks-and-event-listeners)

* Gradio Components
  [https://www.gradio.app/docs/gradio/introduction](https://www.gradio.app/docs/gradio/introduction)

* Gradio「State in Blocks」– Browser State
  [https://www.gradio.app/guides/state-in-blocks#browser-state](https://www.gradio.app/guides/state-in-blocks#browser-state)

* OpenAI Speech‑to‑Text ガイド
  [https://platform.openai.com/docs/guides/speech-to-text](https://platform.openai.com/docs/guides/speech-to-text)

* OpenAI Structured Output
  [https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat](https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat)

---

## OTHER CONSIDERATIONS:

* **大容量ファイル対策**

  * `tempfile.SpooledTemporaryFile` を用いたストリーム書き込みでメモリ使用量を抑制。
  * 25 MB 制限対策として **N 分チャンク + 2 秒オーバーラップ** 分割を採用 (pydubで実装)

* **レート制限 & リトライ**

  * HTTP 429/5xx で指数バックオフ再試行。
  * `gr.Alert` でユーザにわかりやすくエラー表示。

* **セキュリティ・プライバシー**

  * API キーはクライアントの `localStorage` のみ保存し、サーバーには送信しない。
  * 音声・文字起こしはローカル保存のみ。履歴ビューで削除できるゴミ箱アイコンを用意。

* **ローカライズ**

  * UI ラベルは英語のみ (UI のローカライズ不要)
