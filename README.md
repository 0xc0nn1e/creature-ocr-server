# 生き物調査用紙 OCR サーバ

「みんなと生きもの調査隊 夏の虫調査」の手書き調査用紙を読み取る HTTP API。
デスクトップ版 `creature-ocr` の相方で、**AI 呼び出しに必要なものを全部こちら
側に集める**ために存在する。

## これは何か

デスクトップ版はこれまで、各オペレータの PC に `.env`、Google プロジェクト
設定、サービスアカウント鍵、ベンダ SDK を置く必要があった。保守が難しいうえ、
クラウドの資格情報を配って回ることになる。

このサーバはその中身を引き受ける。**資格情報・モデル ID・プロンプト・エンジン
SDK・セルグリッド・値チェックは全部こちら**。アプリ側に残るのは切り抜き（個人
情報帯の除去）、キャッシュ、xlsx、GUI、ベンチマークだけで、**オペレータの PC に
設定項目は一つも残らない**。

    ┌─ creature-ocr（デスクトップ）────────┐      ┌─ creature-ocr-server ─────┐
    │ PDF → 切り抜き（個人情報帯を除去）    │      │ プロンプト / スキーマ      │
    │ キャッシュ                            │ ───▶ │ セルグリッド / 値チェック  │
    │ xlsx 出力 / GUI / ベンチマーク        │ ◀─── │ 資格情報 / モデル / 再試行 │
    └───────────────────────────────────────┘      └────────────────────────────┘
                          PNG 1 ページ  →  8 行の JSON

切り抜きは**必ずアプリ側**で行う。個人情報帯を落とした画像しかここには来ない。
このサーバはそれを検証できないし、検証するふりもしない。(6.2)

## 必要環境

- Python 3.11 以上（コンテナは 3.12）
- Docker と Docker Compose（コンテナで動かす場合）
- Vertex AI が有効な Google Cloud プロジェクトと、`roles/aiplatform.user` を
  持つ ID

## セットアップ（ローカル）

    pip install -e ".[gemini,test]"
    cp .env.example .env      # 値を埋める
    python -m creature_ocr_server

既定では `127.0.0.1:8000` で待ち受ける。`HOST` と `PORT` で変えられる。

ベンダ SDK は**すべて extra** で、基本インストールには入らない。`pip install -e .`
だけでもパッケージは入り、テストも全部通る。エンジン名と extra 名は同じ語なので、
`engine=gemini` と `pip install -e ".[gemini]"` は対になる。(4.2, 6.5)

エンジンは三つ。`documentai` を使うなら `.[documentai]`、`nemotron` は**何も
要らない** ― HTTP を一回投げるだけなので、基本インストールのまま動く。
`.[nemotron]` が要るのは `NEMOTRON_MAX_BYTES` を設定したときだけで、その一つの
設定のためだけに pymupdf が入る。

## Docker で動かす

**認証はホスト側で済ませる。** `gcloud auth application-default login` は
ブラウザを開き、ホストの gcloud 設定に書き込む。コンテナの中でやる意味は
無い ― 開くブラウザが無いし、書いたものはコンテナと一緒に消える。

    gcloud auth application-default login
    gcloud auth application-default set-quota-project <プロジェクト>

    cp .env.example .env      # 値を埋める。GCLOUD_CONFIG_DIR も
    docker compose up --build

`GCLOUD_CONFIG_DIR` にはホストの gcloud 設定ディレクトリを、**円記号ではなく
スラッシュ**で書く。

| OS | 場所 |
|---|---|
| Windows | `C:/Users/<自分>/AppData/Roaming/gcloud` |
| macOS, Linux | `/home/<自分>/.config/gcloud` |

compose はそこを `/gcloud` に**読み取り専用**でマウントし、`CLOUDSDK_CONFIG`
を同じ場所へ向ける。コンテナはホストの資格情報を読むだけで、自分では何も
書かない。`python -m creature_ocr_server` で直接動かす場合は空のままでよい。
SDK が自分で同じファイルを見つける。(6.3)

### サービスアカウントを使う場合

無人運用や共有環境では、ADC は運用を「そのマシンで最後にログインした人」に
縛ってしまう。その場合は鍵を `secrets/` に置き、`.env` の
`GOOGLE_APPLICATION_CREDENTIALS` で `/run/secrets/<鍵>.json` を指す。
`GCLOUD_CONFIG_DIR` は空のままでよい。

鍵には `roles/aiplatform.user` が要る。無いと全ページが
`403 PERMISSION_DENIED` になり、**権限不足ではなく壊れた配備のように見える**。

### 公開範囲

`compose.yaml` は既定で `127.0.0.1:8000` にだけ公開する。**このサーバはクラウド
資格情報を持ち、リクエストごとに課金される**ので、外に開くのは意図した決定で
あるべきで、既定で引き継ぐものではない。開く前に `SERVER_API_KEY` を設定する
こと。

### イメージに資格情報が入っていないことの確認

鍵も `.env` もイメージには入らない。実際に確かめられる。(6.2, 6.3)

    docker run --rm --entrypoint sh creature-ocr-server:0.1.0 -c \
      "find / -xdev -name '.env*' -print; ls -a /app"

`.env*` が一つも出ず `/app` が空なら正しい。パッケージは site-packages にあり、
`/app` にソースの二重コピーは残さない。

## API

すべて `/v1` 以下。`SERVER_API_KEY` を設定した場合、`/v1/health` を除く全ての
エンドポイントが鍵を要求する。ヘッダは `X-API-Key` でも
`Authorization: Bearer` でもよく、**鍵は同じ一つ**。クライアントの HTTP
ライブラリが書きやすい方を使えばよい。クエリ文字列に入れてはいけない ―
通過する全てのアクセスログに残る。

リクエストとレスポンスには `X-Request-ID` が付く。クライアントが送ればそれを
使い、送らなければサーバが振る。**502 の本文にはベンダのエラー文は入らない**
ので、実際の原因を探すときはこの ID でサーバログを引く。(6.2, 6.3)

### `GET /v1/health`

生存確認。エンジンには一切触れないので、何も設定していないサーバでも 200 を
返す。ヘルスチェックが 4 分かかるページの後ろに並んでプロセスを再起動する、
という事故を防ぐため。認証不要。

    {"status": "ok", "version": "0.1.0"}

### `GET /v1/engines`

**クライアントが設定を一つも持たなくて済むのは、このエンドポイントのおかげ。**
何が使えるか、どのモデルを頼めるか、そして**読み取り結果をどのキーでキャッシュ
すべきか**を返す。設定を読むだけで、エンジンを組み立てず、資格情報も読まず、
ソケットも開かない。

    [
      {
        "name": "documentai",
        "ready": false,
        "detail": "DOCUMENTAI_LOCATION is not set: see .env.example",
        "default_model": "",
        "settings": "// hints=ja checks=79ebbe0c",
        "cache_name": "documentai",
        "models": []
      },
      {
        "name": "gemini",
        "ready": true,
        "detail": "",
        "default_model": "gemini-3.7-flash",
        "settings": "",
        "cache_name": "",
        "models": [
          {
            "name": "gemini-3.7-flash",
            "settings": "gemini-3.7-flash temperature=0.0 top_p=0.1 grid=d103bef3 prompt=7b48ed75 checks=79ebbe0c",
            "cache_name": "gemini-3.7-flash"
          }
        ]
      },
      {
        "name": "nemotron",
        "ready": false,
        "detail": "NEMOTRON_ENDPOINT is not set: see .env.example",
        "settings": "nemotron-ocr-v2 v2_multilingual merge=word unsure_below=0.5 endpoint= checks=79ebbe0c",
        "cache_name": "nemotron-v2_multilingual",
        "default_model": "",
        "models": []
      }
    ]

`settings` が要点。クライアントはページを頼む**前に**キャッシュを見るので、
頼む前にキーを組み立てられなければならない。だがプロンプトもグリッドも値
チェックもこのサーバのものなので、クライアントには自力で計算できない。だから
ここで公開する。設定が足りないエンジンは `ready: false` になり、`detail` に
コンストラクタが投げるはずだった文がそのまま入る。(3.2, 4.2)

`documentai` と `nemotron` の `models` が空なのは、**読み手がモデル ID ではない**
から。前者はプロセッサ、後者は `NEMOTRON_ENDPOINT` が指す先そのものが読み手で、
どちらもリクエストで選べるものではない。`model` を送れば 400 になる。

だからこの二つは `settings` と `cache_name` を**エンジンの階層で**返す。モデルの
無いエンジンには `models[]` という置き場所が無く、そこに何も無いままだと
キャッシュキーが**頼む前に読める場所のどこにも存在しない**ことになる。頼む前に
キャッシュを見られるようにするのがこのエンドポイントの唯一の仕事なので、それ
では用を成さない。(3.2)

逆にモデルを取るエンジンでは、この二つは空のまま。鍵はモデルのものであり
`models[]` が既に持っているからで、空のモデル ID から組み立てた文字列は
**存在しない読み手を指す**。何も言わないほうがまだ良い。

`ready: false` のエンジンの鍵は、まだ設定されていない読み手を書き表している。
上の例の `//` がそれで、設定が入れば実際のプロジェクトとプロセッサに変わる。
鍵として使ってよいかは `ready` が答える。

### `GET /v1/sheet`

用紙の定義そのものと、その指紋。列、印刷された選択肢、文字集合、正規化表、
対になる列。分裂したパイプラインの両側が同じ紙について話していることを、
二つの写しではなく一つの出典から確かめられるようにするため。

指紋は**気づくため**のもので、門ではない。不一致でクライアントが動作を止める
設計にすると、こちら側の設定変更一つで現場の全端末が止まる。(6.5)

### `POST /v1/ocr`

`multipart/form-data`:

| 項目 | 必須 | 内容 |
|---|---|---|
| `image` | はい | 切り抜き済みの 1 ページ。PNG のみ |
| `engine` | いいえ | 読み取り方式（engine）。省略時は `OCR_ENGINE`、それも空なら `gemini` |
| `model` | いいえ | モデル ID。`GET /v1/engines` が挙げたものだけ |

`temperature` と `top_p` は**リクエスト項目ではない**。5.2-4 の決定性は
この二つの数値についての約束であり、送れるようにした瞬間に、しかも静かに、
その日読んだページの分だけ失われる。

    curl -sS http://127.0.0.1:8000/v1/ocr \
      -H "X-API-Key: $SERVER_API_KEY" \
      -F "image=@output/生き物/page01.png" \
      -F "engine=gemini"

応答:

    {
      "request_id": "live-smoke-2",
      "engine": "gemini",
      "model": "",
      "settings": "gemini-3.7-flash temperature=0.0 top_p=0.1 grid=d103bef3 prompt=7b48ed75 checks=79ebbe0c",
      "cache_name": "gemini-3.7-flash",
      "sheet_fingerprint": "79ebbe0c",
      "rows": [
        {"no": "1", "bug_name": "ショウリョウバッタ", "symbol": "き",
         "where": "しめった地面", "what_doing": "④", "found_month": "7",
         "found_day": "8", "location_town": "芝公園", "location_chome": "4",
         "location_name": "芝公園", "map_symbol": "G", "notice": ""}
      ],
      "report": {
        "values": 33, "rejected": 0, "disagreements": 0, "gaps": 0, "unsure": 2,
        "score": 0.9393939393939394, "percent": "94%",
        "cells": [{"row": 1, "field": "bug_name", "problem": "unsure"}]
      },
      "findings": ["row 1: bug_name: the engine could not read '...' cleanly"],
      "usage": {"calls": 1, "seconds": 15.7, "prompt_tokens": 3177,
                "output_tokens": 1336, "thought_tokens": 426}
    }

固定されている点:

- `rows` は**空行を含めて必ず 8 行**。5.1 は行の欠落を許さない。
- `rows[].no` は**文字列**。デスクトップ版でも文字列で、そのまま xlsx に入る。
  親切なつもりで整数にすると、誰も触っていないクライアントの挙動が静かに変わる。
- `report.cells[].row` は**整数**で、だからこそ `no` という名前にしていない。
  同じ名前で型が違う項目は、一度だけ、静かに事故を起こす。
- `report.score` はページが値を一つも返さなかったとき `null`。1.0 ではない。
  白紙だったのか、書かれた紙を読んで何も返さなかったのかは区別できず、**一番
  開いて確認すべきページが一番良く見える**ことだけは避けなければならない。
- `usage` のキーはデスクトップ版 `Usage` の項目と完全に同じなので、
  `Usage(**usage)` でそのまま既存のコードに戻せる。
- `findings` に**ページ名は付かない**。このサーバは画像を渡されただけで、それが
  どのファイルだったかを知らない。`output/ocr_error.txt` を書く側が前に足す。
  両側で付けても、どちらも付けなくても、1000 枚のバッチのログは区別のつかない
  行の羅列になる。この規則はどちらか片方だけ直してはいけない。(7.2)

### エラー

| コード | 意味 |
|---|---|
| 400 | 未知の engine、許可リストにない model、model を取らない engine への model 指定 |
| 401 | `X-API-Key` も `Authorization: Bearer` も無いか違う |
| 413 | ページが `SERVER_MAX_IMAGE_BYTES` を超えている |
| 415 | PNG ではない（申告ではなく先頭バイトで判定する） |
| 429 | 同時実行の空きが `SERVER_QUEUE_TIMEOUT_SECONDS` 以内に出なかった。`Retry-After` 付き |
| 502 | 再試行を使い切ってもエンジンが読めなかった |
| 503 | エンジンが未設定、または extra が未インストール |
| 504 | 1 ページの予算 `SERVER_REQUEST_DEADLINE_SECONDS` を使い切った |

502 の本文は「読めなかった」ことと `request_id` しか言わない。Vertex のエラー
文にはプロジェクト ID が、ときにサービスアカウントのアドレスが入るためで、
資格情報を `settings` とログから締め出す規則は応答本文にも及ぶ。**実際のエラー
文はサーバログにしかない。** (6.2, 6.3)

## クライアントからの接続

アプリ側がこの API に切り替えるとき、アプリ側が持つのは次の 4 つだけ。
プロジェクト ID もリージョンもモデル ID もサービスアカウントもベンダ SDK も
要らない。それが、この API が存在する理由そのもの。

| アプリ側が持つもの | 例 |
|---|---|
| サーバの URL | `http://127.0.0.1:8000` |
| API 鍵 | `SERVER_API_KEY` と同じ文字列 |
| どのエンジンで読むか（任意） | `gemini` |
| どのモデルで読むか（任意） | `gemini-3.7-flash` |

### 1. 起動時に一度

    GET /v1/engines      使えるエンジン、頼めるモデル、キャッシュキー
    GET /v1/sheet        用紙の定義と指紋

どちらも課金されず、エンジンも組み立てず、資格情報も読まない。起動のたびに
呼んでよい。`ready: false` のエンジンは `detail` に理由が入っているので、
選択肢から外すか、そのまま利用者に見せる。

モデルの一覧もここから取る。**アプリ側にモデル ID を書かない。** Vertex は
Google の都合でモデル ID を廃止するので、書いた瞬間に賞味期限が始まる。

### 2. ページごとに

    POST /v1/ocr         multipart/form-data

    curl -sS http://127.0.0.1:8000/v1/ocr \
      -H "X-API-Key: $SERVER_API_KEY" \
      -H "X-Request-ID: page01" \
      -F "image=@output/生き物/page01.png;type=image/png" \
      -F "engine=gemini" \
      -F "model=gemini-3.7-flash"

Python なら:

    import requests

    BASE = "http://127.0.0.1:8000"
    KEY = "..."                       # the same string as SERVER_API_KEY

    # Connect fast, read slow, and they are two different numbers. The read
    # timeout has to outlast the server's own budget: the queue wait, the page
    # budget, and one more call on top. See "タイムアウトの予算" below.
    TIMEOUT = (10, 420)

    session = requests.Session()
    session.headers["X-API-Key"] = KEY

    engines = session.get(f"{BASE}/v1/engines", timeout=(10, 30)).json()

    def read_page(png: bytes, name: str, engine: str = "gemini", model: str = ""):
        reply = session.post(
            f"{BASE}/v1/ocr",
            headers={"X-Request-ID": name},
            files={"image": (f"{name}.png", png, "image/png")},
            data={"engine": engine, "model": model},
            timeout=TIMEOUT,
        )
        # Placeholder. What to do per status code is the table below: 429 is
        # a wait, 502 and 504 are a page to record and move past, and 503 is a
        # reason to stop the batch.
        reply.raise_for_status()
        return reply.json()

`image` の中身は PNG でなければならない。判定は申告した Content-Type ではなく
先頭バイトで行う。`engine` と `model` は省略でき、空文字列は省略と同じに扱う。

### 3. 再試行はサーバが済ませている

1 回の `POST /v1/ocr` の中で、サーバはエンジンを最大 4 回呼び、1 + 2 + 4 秒
待つ（`OCR_RETRIES`、6.4）。**アプリ側がその上から無条件で投げ直すと、1 ページ
の課金がその倍数になる。** 応答コードごとの扱い:

| コード | アプリ側の扱い |
|---|---|
| 200 | 保存する |
| 400 / 413 / 415 | 送り方の誤り。投げ直さない |
| 401 | 鍵の誤り。バッチを止める |
| 429 | `Retry-After` 秒待って**同じページを投げ直す**。サーバが混んでいるだけで、ページには何の問題もない |
| 502 | エンジンが 4 回とも読めなかった。投げ直しても同じになりやすいので、そのページを「読めなかった」と記録して次へ進む |
| 503 | サーバ側の設定不足。自然には直らないので、バッチを止めて `detail` を利用者に見せる |
| 504 | 1 ページの予算切れ。502 と同じ扱い |

**502 と 504 の本文には `request_id` しか入らない。** ベンダのエラー文は
サーバのログにしかないので、失敗を記録するときは `X-Request-ID` を必ず一緒に
残す。それが唯一の突き合わせ手段になる。

### 4. キャッシュの鍵

`cache_name` がディレクトリ名、`settings` がその中の鍵。**どのエンジンでも
`GET /v1/engines` で先に取れる**ので、ページを頼む**前に**キャッシュを見て、
当たれば 1 回も課金せずに済む。gemini はモデルごとに `models[]` の中、
`documentai` と `nemotron` はモデルを取らないのでエンジンの階層に入っている。
どちらを読むかは `models` が空かどうかで決まる。

`sheet_fingerprint` が `GET /v1/sheet` の指紋と食い違ったら、用紙の定義が
変わったということ。**気づくためのもので、止めるためのものではない。**

### 5. 応答をアプリ側の型に戻す

- `usage` のキーはデスクトップ版 `Usage` と完全に同じなので `Usage(**usage)`。
- `rows` は空行込みで必ず 8 行、`rows[].no` は文字列、`report.cells[].row` は
  整数。
- `report.score` は値が一つも返らなかったとき `null`。1.0 ではない。
- `findings` にページ名は付いていない。`output/ocr_error.txt` を書く側が前に
  足す。両側で足しても、どちらも足さなくても、1000 枚のログは区別のつかない
  行の羅列になる。

固定されている点の詳細は `POST /v1/ocr` の項。

## 設定

すべて環境変数。既定値のない必須項目は Google の 3 つだけで、残りは全部
既定値で動く。詳細と理由は `.env.example` に書いてある。

| 変数 | 既定 | 内容 |
|---|---|---|
| `GCLOUD_CONFIG_DIR` | `./secrets` | ホストの gcloud 設定ディレクトリ。compose が `/gcloud` に読み取り専用でマウントする |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | 空 | ユーザ ADC のクォータプロジェクト。`set-quota-project` の代わり |
| `GOOGLE_CLOUD_PROJECT` | なし | モデルを提供するプロジェクト |
| `GEMINI_LOCATION` | なし | Vertex のリージョン。`global` も有効 |
| `GEMINI_MODEL` | なし | 既定のモデル ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | 空 | 空なら ADC、パスなら鍵ファイル、`{` 始まりなら鍵そのもの |
| `GEMINI_MODELS` | `config.py` の一覧 | 許可するモデル、カンマ区切り |
| `OCR_ENGINE` | `gemini` | 既定のエンジン。`gemini` / `documentai` / `nemotron` |
| `DOCUMENTAI_LOCATION` | なし | Document AI のリージョン。`us` か `eu` のみ |
| `DOCUMENTAI_PROCESSOR_ID` | なし | Document OCR プロセッサの ID |
| `NEMOTRON_ENDPOINT` | なし | ページを POST する URL そのもの。何も足さない |
| `NVIDIA_API_KEY` | なし | nemotron の bearer トークン |
| `NEMOTRON_VARIANT` | `v2_multilingual` | どのビルドを指しているか。キャッシュ名になるだけ |
| `NEMOTRON_MAX_BYTES` | 空 | 超えたら JPEG で再エンコード。画質を捨てるので既定は空 |
| `SERVER_API_KEY` | 空 | 設定すると `X-API-Key` か `Authorization: Bearer` を要求する |
| `SERVER_MAX_IMAGE_BYTES` | 20 MiB | 受け付けるページの上限 |
| `SERVER_MAX_CONCURRENCY` | 4 | 同時に走らせるエンジン呼び出し |
| `SERVER_REQUEST_DEADLINE_SECONDS` | 300 | 1 ページの予算 |
| `SERVER_QUEUE_TIMEOUT_SECONDS` | 30 | 空き待ちの上限 |
| `LOG_LEVEL` | `info` | このパッケージのログ水準 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | `python -m` で起動したときだけ |

`GEMINI_MODEL` に既定値を置いていないのは意図的。Vertex のモデル ID は
バージョン付きで、Google の都合で廃止される。コードに書いた既定はいずれ全件
失敗するか、もっと悪いことに、**誰も選んでいないモデルで静かに読み続ける**。

`GEMINI_MODELS` は献立表であると同時に**メモリの上限**でもある。エンジンは
モデルごとに一度だけ組み立てて保持するので、自由入力を許すとリクエストごとに
SDK クライアントが増え続ける。

### タイムアウトの予算

デスクトップ版での 1 ページの実測は 18〜68 秒。6.4 が求める 3 回の再試行と
1 + 2 + 4 秒の待機を足すと、**まともに動いていても 1 リクエストが約 280 秒
かかりうる**。`SERVER_REQUEST_DEADLINE_SECONDS` の既定 300 はそこから来ている。

このサーバ経由の実測は下の「実測」の項の通りもっと速く、`gemini-3.7-flash` で
8〜12 秒、いちばん遅い `gemini-3.8-flash` でも 13〜27 秒。締めるなら
`SERVER_REQUEST_DEADLINE_SECONDS=180`、`SERVER_QUEUE_TIMEOUT_SECONDS=60` が
実測に対して十分な余裕を残す。既定のままでも困らない。

**予算は走っている呼び出しを途中で切らない。** 期限は次の試行を始める前に
見るので、最後の試行が期限の直前に始まれば、その 1 回分だけ超過する。gemini
エンジンは呼び出し単位のタイムアウトを持たないので、超過の上限を決めるのは
ベンダ SDK の側になる。

したがって:

- **クライアントの読み取りタイムアウトは
  `SERVER_QUEUE_TIMEOUT_SECONDS + SERVER_REQUEST_DEADLINE_SECONDS` に 1 回分の
  呼び出しを足した値より大きくする。** 既定値のままなら 30 + 300 + 90 で
  420 秒。接続タイムアウトと読み取りタイムアウトは別の数値である。
- **前段にリバースプロキシを置くなら、最初に壊れるのはその読み取りタイムアウト。**
  nginx の既定は 60 秒で、健全なだけの遅いページを三分の一で切る。
- 予算切れは 504 で、502 ではない。自分で決めた予算の話であって、エンジンが
  壊れている話ではないから。

### 同時実行

要件 6.1 が言う「エンジンのクォータに合わせて 2〜8」の設定は、いまここにある。
既定 4。制約になるのはこのマシンではなく Vertex のクォータなので、上げても
こちらでは何も起きず、あちらでは全部起きうる。

ワーカープロセスは 1 つ。エンジンはプロセス内に保持されるので、2 つ目の
ワーカーは 2 組目の SDK クライアントと 2 回目の資格情報更新を意味する。
待っているだけの仕事に対してそれは無駄。増やすならコンテナを増やす。

## テスト

    python -m unittest discover -s tests

ネットワークも資格情報もベンダ SDK も要らない。SDK は `_load_sdk()` という
一点の縫い目でしか名指されておらず、テストはそこを丸ごと差し替える。

大半はデスクトップ版 `tests/` からそのまま持ってきたもので、それ自体が検査に
なっている。**ここで書き換えないと通らなかったケースは、二つの実装の差**で
あり、その差こそ `tests/test_parity.py` が防ぐためにある。

### `tests/test_parity.py`

両リポジトリの `clean`、`fold`、`names_a_choice`、`row_gaps`、
`paired_field_problems`、`cell_problems`、`page_confidence` を同じ入力で走らせ、
一文字でも違えば落ちる。さらに同じボックス列から作った 1 ページ分の行と品質
カウントも突き合わせる。API 呼び出しは 0 回。

デスクトップ版のチェックアウトが無い環境では skip する。**skip は合格では
ない。** どちらかの `ocr.py` か `config.py` を触ったら、両方が揃っているマシン
で必ず走らせること。場所は `CREATURE_OCR_SRC` で指定できる。既定では二つの
チェックアウトが隣り合っていることを期待する。

## 実測（2026-09-07、1 ページ × 5 回）

コンテナに実際の切り抜き済みページ（3367 × 1442、412 KiB、8 行 33 値）を
1 枚ずつ投げた結果。同じページ、同じプロンプト、変えたのはモデルだけ:

| モデル | 応答 | 出力トークン | うち思考 | 確信度 |
|---|---|---|---|---|
| `gemini-3.8-flash` | 200、26.8 秒 | 3,536 | 2,606 | 91% |
| `gemini-3.8-flash` | 200、15.4 秒 | 2,272 | 1,350 | 94% |
| `gemini-3.8-flash` | 200、13.1 秒 | 1,928 | 1,006 | 94% |
| `gemini-3.7-flash` | 200、11.6 秒 | 1,786 | 864 | 94% |
| `gemini-3.7-flash` | 200、8.1 秒 | 1,219 | 309 | 94% |

入力は 5 回とも 3,177 トークン、呼び出しは 1 回、再試行 0、値 33、拒否 0、
欠落 0、不一致 0。所要時間は思考トークン数にそのまま比例している。

**`gemini-3.8-flash` は `gemini-3.7-flash` の約 1.9 倍遅く、読み取りは良く
ならない。** 33 の値は 5 回とも同一で、3.8 の 1 回だけモデル自身が「読みにくい」
と申告したセルが 1 つ増えて 91% になった。既定を 3.7 にしている理由がこれ。

同じページについてデスクトップ版が保存していた読み取りと突き合わせたところ、
**88 セル中 88 セルが一致**。`grid=` と `prompt=` の指紋も一致しており、
プロンプトとセルグリッドが移送で変わっていないことを示している。

キャッシュ済みの実ページ 20 枚（3 モデル + nemotron）を両実装に通した比較でも、
**全ページ全セルが一致**した。

## 未実装

- **サーバ側キャッシュ。** アプリ側が既にページごとに持っている。
- **バッチ endpoint、非同期ジョブ、PDF アップロード。** 切り抜きと個人情報帯の
  除去は、これからもアプリ側に残る。
- **アプリ側の `RemoteEngine`。** アプリをこの API 経由に切り替える作業は、
  この API を使う側の変更であり、別の作業になる。必要なことは
  「クライアントからの接続」の項にまとめてある。
