"""The survey sheet, as data, and the names of the settings that steer a run.

Two kinds of thing live here and nothing else. The first is the sheet itself:
which columns it has, which of them are answered off a printed list, which
characters each can hold, and where every cell sits on the page. The second is
the names of the environment variables an operator fills in; the values are
read where they are used, never here.

This is the desktop pipeline's config.py with everything about files taken out.
The server never renders a PDF, never crops, never writes an xlsx and never
reads a .env from beside an executable, so the DPI, the data band, the cell
inset, the output filenames and load_env do not appear. What is left is the
half that decides what a value means, and that half is copied rather than
paraphrased: tests/test_parity.py compares it against the desktop repository's
so the two cannot drift apart unnoticed. (3.1, 6.2, 6.5)
"""

import string
import unicodedata
from collections.abc import Iterable

# 3.1: the OCR fields, in xlsx column order. The key is what the pipeline uses
# everywhere; the second element is the xlsx column header. A change of paper
# format should need only new entries here and new coordinates below. (3.1, 6.5)
OCR_FIELDS = (
    ("bug_name", "発見した虫の名前"),
    ("symbol", "記号"),
    ("where", "どこで"),
    ("what_doing", "何してた"),
    ("found_month", "見つけた月"),
    ("found_day", "見つけた日"),
    ("location_town", "見つけた場所_町"),
    ("location_chome", "見つけた場所_丁目"),
    ("location_name", "場所の名前"),
    ("map_symbol", "マップ記号"),
    ("notice", "気が付いたこと"),
)

# Three of the eleven columns are not free writing at all: the child copies one
# answer off a printed list. どこで's 記号 and 何してた are listed on page 5 of
# the survey handbook, マップ記号 on the 探検地環境マップ across pages 6 and 7.
# The lists live here rather than in the prompt so 6.2's "the rules come from
# configuration" holds for the vocabulary as well as for the character classes,
# and so one table does all three jobs: it tells the model what the column can
# hold, it is what a returned value is checked against, and its meanings are
# what the paired free-text column is read against below.
#
# Only the key ever reaches the xlsx. The sheet carries the letter, so the xlsx
# carries the letter; the meaning is there to read the paper with. (3.1, 6.2)
SYMBOL_CHOICES = {
    "あ": "道や駐車場",
    "い": "家やビルなどの建物",
    "う": "高い木",
    "え": "低い木",
    "お": "葉っぱ",
    "か": "花",
    "き": "地面で",
    "く": "池",
    "け": "その他",
}

WHAT_DOING_CHOICES = {
    "①": "飛んで通りすぎた",
    "②": "ぐるぐる飛び回っていた",
    "③": "ゆっくり飛んでいた",
    "④": "止まっていた",
    "⑤": "みつをすっていた",
    "⑥": "食べていた",
    "⑦": "たまごをうんでいた",
    "⑧": "歩きまわっていた",
    "⑨": "死んでいた",
    "⑩": "その他",
}

# A to N, not A to H: the handbook's own bubble says Ⓐ 〜 Ⓝ の記号で書いてね.
MAP_SYMBOL_CHOICES = {
    "A": "住宅地",
    "B": "学校",
    "C": "かだん",
    "D": "並木",
    "E": "駐車場",
    "F": "お寺や神社",
    "G": "大きな公園",
    "H": "駅",
    "I": "ビル",
    "J": "道路",
    "K": "町中の公園",
    "L": "家の中",
    "M": "川や運河",
    "N": "高層ビルのまわりの緑地",
}

# Which columns are answered off a list, and which list. A value that is not on
# its column's list is not a reading of that column, so ocr.clean empties it the
# same way it empties a character outside the set. This is also the list the
# model is given, so it is only the printed spelling of each answer. (3.1, 5.2-2)
OCR_FIELD_CHOICES = {
    "symbol": SYMBOL_CHOICES,
    "what_doing": WHAT_DOING_CHOICES,
    "map_symbol": MAP_SYMBOL_CHOICES,
}

# Other spellings that name the same printed answer, and are accepted as it.
# 何してた is a numbered list, so an engine that reads ④ and drops the ring has
# read the answer, not a different one - measured, not guessed: 4 of the 13
# 何してた cells in output/ came back as a bare 4.
#
# Field by field rather than in OCR_CHARACTER_FOLDS, which is field-blind and
# would have to hold this the other way round, as ④ meaning 4. That is exactly
# the fold that table refuses: 何してた sits one column from 見つけた月, where a
# ④ is a column bleed and never a month.
#
# Only what counts as an answer is decided here. ocr.clean still returns the
# value the engine gave, so nothing in the xlsx was written by this table.
OCR_FIELD_SPELLINGS = {
    "what_doing": {
        str(number): circled
        for number, circled in enumerate(WHAT_DOING_CHOICES, start=1)
    },
}

# 記号 and どこで are the same answer twice, and so are マップ記号 and
# 場所の名前: the letter is the printed category and the column beside it is the
# child's own wording for it. The two cannot be checked against each other by
# equality - the sheet's own printed example answers き, 地面で, with
# しめった地面 - so what is listed here is the words a wording of that category
# is expected to contain.
#
# A wording carrying none of any category's words says nothing about the letter
# and is never complained about; only one that carries another category's words
# and not its own is. And it is only ever reported: which of the two the child
# got wrong is not knowable from the page, so rewriting either would be exactly
# the invented value 5.2-2 forbids. (3.1, 5.2-2, 6.2)
SYMBOL_WORDS = {
    "あ": ("道", "駐車"),
    "い": ("家", "ビル", "建物", "アパート", "マンション"),
    "う": ("木",),
    "え": ("木",),
    "お": ("葉",),
    "か": ("花",),
    "き": ("地面", "じめん", "土"),
    "く": ("池",),
    "け": (),  # その他 is answered with anything at all.
}

MAP_SYMBOL_WORDS = {
    "A": ("住宅", "団地"),
    "B": ("学校", "小学", "中学", "高校"),
    "C": ("かだん", "花だん", "花壇"),
    "D": ("並木", "なみき"),
    "E": ("駐車", "パーキング"),
    "F": ("寺", "神社", "宮"),
    "G": ("公園", "庭園"),
    "H": ("駅",),
    "I": ("ビル",),
    "J": ("道路", "通り"),
    "K": ("公園",),
    "L": ("家", "自宅", "室内"),
    "M": ("川", "運河"),
    "N": ("緑地", "ビル"),
}

# (letter column, wording column, the words each letter is expected to bring
# with it). Read by ocr.paired_field_problems.
OCR_PAIRED_FIELDS = (
    ("symbol", "where", SYMBOL_WORDS),
    ("map_symbol", "location_name", MAP_SYMBOL_WORDS),
)

# 発見した虫の名前 is free writing - any species can turn up - but the same
# survey ran last year and published its tally, and these 28 names, most common
# first, are all of it. They go to the model as a reading aid and nothing else:
# no value is ever checked against this list, because a species nobody saw last
# year is a perfectly good answer.
#
# It is a hint that has to be given carefully. 7.2's measurement already caught
# Gemini "correcting" a child's ハラビロオマキリ to ハラビロカマキリ with no
# list in front of it, so the prompt states explicitly that the list settles
# handwriting and never replaces it. Re-benchmark after touching this. (3.1, 7.2)
BUG_NAME_HINTS = (
    "シオカラトンボ",
    "アオスジアゲハ",
    "カナブン",
    "ヤマトシジミ",
    "ナミアゲハ",
    "キアゲハ",
    "オオシオカラトンボ",
    "アオドウガネ",
    "ツマグロヒョウモン",
    "ショウリョウバッタ",
    "タマムシ",
    "リュウキュウツヤハナムグリ",
    "クロアゲハ",
    "ギンヤンマ",
    "コフキコガネ",
    "アカボシゴマダラ",
    "オンブバッタ",
    "ハラビロカマキリ",
    "シロテンハナムグリ",
    "オオカマキリ",
    "ルリシジミ",
    "ツバメシジミ",
    "コシアキトンボ",
    "コカマキリ",
    "モンキアゲハ",
    "ショウリョウバッタモドキ",
    "ナガサキアゲハ",
    "チョウセンカマキリ",
)

# 3.1 also fixes which characters a field can hold. Everything not listed here
# is free Japanese handwriting and is kept exactly as the engine read it; the
# entries below are the fields where anything outside the set is a misread, and
# 5.2-2 says a misread is an empty string rather than a guess. 見つけた場所_丁目
# holding 山 is not a chome number, and carrying it through would put a wrong
# value in the xlsx where a blank would have told the operator to look.
#
# Matched after the folds listed in OCR_CHARACTER_FOLDS below, so a full-width
# ７ counts as 7 and a circled Ⓖ as G. A value that still does not fit is
# rejected whole, never edited down to the characters that do fit.
OCR_FIELD_CHARACTERS = {
    "found_month": string.digits,
    "found_day": string.digits,
    "location_chome": string.digits,
    "map_symbol": string.ascii_letters,
}

# 6.2 asks for out-of-domain character-class detection, with the rules coming
# from configuration rather than from the code. These are those rules for the
# free-text fields: the scripts a Japanese survey sheet can be written in.
# Anything outside them is a misread - Document OCR returns Korean for
# handwritten kana often enough to matter - and 5.2-2 makes a misread an empty
# string rather than a guess. Widen this list, do not touch the code, if a
# future sheet legitimately carries another script.
OCR_ALLOWED_SCRIPTS = (
    (0x0020, 0x007E),  # ASCII: digits, latin letters, plain punctuation
    (0x00B0, 0x00B0),  # degree sign
    (0x2460, 0x24FF),  # enclosed alphanumerics: 丸数字 such as 何してた's ④
    (0x3000, 0x303F),  # CJK punctuation: 、。〆々
    (0x3040, 0x309F),  # hiragana
    (0x30A0, 0x30FF),  # katakana, including the ー length mark and ・
    (0x3200, 0x32FF),  # enclosed CJK letters: 記号's circled kana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs: kanji
    (0xFF00, 0xFFEF),  # halfwidth and fullwidth forms
)

def _rings(*ranges: Iterable[int]) -> dict[str, str]:
    """Circled characters mapped to the character inside the ring.

    Derived with NFKC rather than typed out, over ranges that hold nothing but
    rings. What makes NFKC the wrong tool for a value is what it does to
    everything else - it folds ⁸ to 8 and ℊ to g as well - not what it does to
    a ring, and 47 circled katakana are not worth transcribing by hand when a
    transcription is the thing most likely to be wrong.
    """
    folds = {}
    for codes in ranges:
        for code in codes:
            ring = chr(code)
            plain = unicodedata.normalize("NFKC", ring)
            if plain != ring:
                folds[ring] = plain
    return folds


# The only rewrites allowed before a value is checked against its character
# set. Every entry is the same answer written another way: a width variant, or
# a ring drawn round a character. The sheet asks for rings - マップ記号 invites
# one round a letter and 何してた's answers are printed as ①-⑩ - and the ring
# is the paper's way of saying "pick this one", not part of the answer, so it
# comes off and only the character inside reaches the xlsx.
#
# Still not Unicode NFKC applied wholesale: what is not listed here is left
# alone, and therefore fails the character set and empties the cell. (5.2-2)
OCR_CHARACTER_FOLDS = {
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},  # ０-９ fullwidth digits
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},  # Ａ-Ｚ fullwidth capitals
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},  # ａ-ｚ fullwidth smalls
    **_rings(
        range(0x24B6, 0x24EA),  # Ⓐ-Ⓩ ⓐ-ⓩ circled latin
        range(0x3280, 0x328A),  # ㊀-㊉ circled kanji numerals
        range(0x32D0, 0x32FF),  # ㋐-㋾ circled katakana
    ),
}

# Circled digits, folded everywhere except the three columns whose answer is a
# bare number. There a ① is not a month written oddly: it is 何してた's answer
# from a neighbouring column bleeding in, and folding it would hand the xlsx a
# plausible month nobody wrote. Left alone it fails the digit set and the cell
# is emptied instead, which is what 5.2-2 asks for. 何してた itself, and every
# free-writing column, does fold them: there the ring is the printed answer's
# own decoration. (3.1, 5.2-2)
OCR_CIRCLED_DIGIT_FOLDS = {
    **_rings(range(0x2460, 0x2474), (0x24EA,)),  # ①-⑳ ⓪
    **{chr(0x2776 + i): str(i + 1) for i in range(10)},  # ❶-❿ negative circled
    **{chr(0x2780 + i): str(i + 1) for i in range(10)},  # ➀-➉ circled sans-serif
    **{chr(0x278A + i): str(i + 1) for i in range(10)},  # ➊-➓ negative sans-serif
}

# The columns a circled digit must never be folded into. Their whole answer is
# a number, so a folded ring would be indistinguishable from one.
OCR_NUMBER_ONLY_FIELDS = ("found_month", "found_day", "location_chome")

# One survey record per row, numbered 1 to 8 on the sheet.
ROWS_PER_PAGE = 8

# Horizontal ruling as fractions of the cropped page height: nine edges bounding
# the eight data rows. Measured off a 300 DPI pageNN.png.
CELL_ROW_EDGES = (
    0.0364,
    0.1526,
    0.2712,
    0.3877,
    0.5055,
    0.6227,
    0.7406,
    0.8582,
    0.9764,
)

# Per-field (left, right) fractions of the cropped page width. Deliberately not
# contiguous: the printed double rule between 何してた and 見つけた月 sits in the
# 0.5195-0.5248 gap and belongs to neither field. The No. column,
# 0.0419-0.0704, is absent because the row number is read off the row
# index and never OCRed by the coordinate engines - but it is inside the crop,
# and the Gemini prompt asks the model to read it so a row can be placed by the
# number printed beside it rather than by its position in the answer. (5.1)
CELL_COLUMNS = {
    "bug_name": (0.0704, 0.2424),
    "symbol": (0.2424, 0.2766),
    "where": (0.2766, 0.4138),
    "what_doing": (0.4138, 0.5195),
    "found_month": (0.5248, 0.5634),
    "found_day": (0.5634, 0.6027),
    "location_town": (0.6027, 0.6847),
    "location_chome": (0.6847, 0.7263),
    "location_name": (0.7263, 0.8355),
    "map_symbol": (0.8355, 0.8804),
    "notice": (0.8804, 0.9968),
}


# Gemini on Vertex AI. Only the key names are here; the values arrive from the
# real environment, which in a container means compose's env_file or the
# orchestrator, never a file baked into the image. Nothing about the project,
# the region or the model is hardcoded, so pointing the server somewhere else
# is a restart rather than a rebuild. (4.2, 6.2, 6.5)
#
# The location is its own key rather than a shared one because Vertex and
# Document AI do not name their regions the same way - Document AI takes us or
# eu, Vertex takes us-central1 or global - and pointing one at the other's
# value fails as a NOT_FOUND that reads like a broken project.
GEMINI_ENV_PROJECT = "GOOGLE_CLOUD_PROJECT"
GEMINI_ENV_LOCATION = "GEMINI_LOCATION"
GEMINI_ENV_MODEL = "GEMINI_MODEL"

# Which engine a request may name a model for. Document AI's reader is a
# processor and nemotron's is whichever deployment its URL points at; neither
# is a model id, so asking for one is a mistake worth a 400 rather than a
# setting to be quietly ignored. (6.5)
GEMINI_MODELS_ENGINE = "gemini"

# The models a request is allowed to ask for, and the list GET /v1/engines
# publishes so a client needs no model configuration of its own. GEMINI_MODEL
# is still what a request that names none is served with.
#
# It is an allow-list and not merely a menu. An engine is built once per model
# and kept, so a free-form model string would let anyone grow this server an
# unbounded number of SDK clients; only a name from this list is ever built.
#
# Vertex retires model ids on Google's own schedule, so this is expected to be
# edited - and GEMINI_MODELS overrides it from the environment, comma
# separated, for an operator who cannot wait for a release. Changing which
# model reads a page changes what the page says, so a batch read again under
# another model is a batch billed again. (3.2, 4.2, 6.5)
GEMINI_ENV_MODELS = "GEMINI_MODELS"

GEMINI_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
)

# How the Google engines authenticate. Empty means Application Default
# Credentials; a path names a service account key file; a value starting with {
# is the key itself. engines/auth.py reads it, and is the only place that does.
#
# For a container the mounted-file form is the one to use: the key stays
# outside the image, read-only, and can be rotated without a rebuild. (6.3)
GOOGLE_ENV_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"

# 5.2-4 fixes these so re-running the same image gives the same rows. They are
# constants and never request parameters: the moment a caller can send a
# temperature, the determinism this pipeline promises is gone, and it is gone
# silently, for whichever pages happened to be read that day.
#
# Note the tension, and leave it visible rather than resolving it in code:
# Google's guidance for the Gemini 3 models is to leave temperature at its
# default of 1.0 and warns that lowering it can make a model loop or reason
# worse. 5.2-4 says 0.0. The requirement wins, because a survey sheet needs
# transcription rather than reasoning. (5.2-4, 6.5)
GEMINI_TEMPERATURE = 0.0
GEMINI_TOP_P = 0.1

# Which engine a request that names none is served with. Only the key name and
# the fallback are here; the value is read where the engine is built, so a test
# can set the environment without reimporting this module. (6.5)
OCR_ENGINE_ENV = "OCR_ENGINE"
DEFAULT_OCR_ENGINE = "gemini"

# 6.4: a failing call is retried three times with exponential backoff.
OCR_RETRIES = 3
OCR_BACKOFF_SECONDS = 1.0

# The server's own settings. Key names, and the value used when the key is
# unset - which is every key, because a server that will not start until five
# variables are filled in is a server nobody can try.
SERVER_ENV_API_KEY = "SERVER_API_KEY"
SERVER_ENV_MAX_IMAGE_BYTES = "SERVER_MAX_IMAGE_BYTES"
SERVER_ENV_MAX_CONCURRENCY = "SERVER_MAX_CONCURRENCY"
SERVER_ENV_REQUEST_DEADLINE = "SERVER_REQUEST_DEADLINE_SECONDS"
SERVER_ENV_QUEUE_TIMEOUT = "SERVER_QUEUE_TIMEOUT_SECONDS"
SERVER_ENV_LOG_LEVEL = "LOG_LEVEL"

# A 300 DPI page of this sheet is around half a megabyte. The cap is generous
# rather than tight because the point is to refuse a body nobody meant to send,
# not to second-guess an operator's scanner.
SERVER_MAX_IMAGE_BYTES = 20 * 1024 * 1024

# How many engine calls may be in flight. 6.1 asks for a parallelism of 2 to 8
# set to suit the engine quota, and this is where that setting now lives: the
# desktop app no longer decides, because the quota belongs to whoever runs the
# server. Four sits in the middle of the range 6.1 names.
#
# The binding constraint is Vertex quota rather than this machine - a page is
# some forty seconds of waiting and almost no work - so raising it costs
# nothing here and may cost everything there. (6.1)
SERVER_MAX_CONCURRENCY = 4

# The whole budget for one request. A page measures 18 to 68 seconds, and 6.4
# allows three retries with 1 + 2 + 4 seconds of backoff, so a genuinely bad
# page can legitimately take about 280 seconds. An attempt that cannot finish
# inside what is left is not started, and the request ends as 504 rather than
# dying halfway through a call somebody has already been billed for.
#
# Every client and every proxy in front of this server needs a read timeout
# larger than this. A default nginx allows 60 seconds and would cut a perfectly
# healthy slow page in three. (6.1, 6.4)
SERVER_REQUEST_DEADLINE_SECONDS = 300.0

# How long a request waits for a concurrency slot before it is turned away with
# 429. Waiting is right; waiting for five minutes and concluding the server has
# hung is not.
SERVER_QUEUE_TIMEOUT_SECONDS = 30.0

# What a PNG starts with. An upload is identified by this rather than by the
# content type the client declared, because a declared type is a claim and this
# is a fact - and the type is passed on to the engine, so believing a wrong one
# would tell the model a lie about what it is looking at. (6.2)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sheet_definition() -> dict:
    """The sheet as the value checks see it, as plain JSON-able data.

    One function doing three jobs: it is the body of GET /v1/sheet, it is what
    ocr.checks_fingerprint hashes, and it is what a client can hold onto so
    both halves of a split pipeline describe the same piece of paper.

    Only what a check reads is in it. The cell grid is not: assignment is the
    server's own business now, and the grid already has a fingerprint of its
    own inside an engine's settings. Neither are the bug-name hints, which
    steer the prompt rather than a check and are covered by the prompt
    fingerprint.

    Everything comes out as a list, a string or a number, sorted wherever the
    source has no order of its own, so the same definition serialises to the
    same bytes on any interpreter. That matters because it is hashed: a digest
    taken over repr() of a dict would be promising insertion order rather than
    content. (6.5)
    """
    return {
        "rows_per_page": ROWS_PER_PAGE,
        "fields": [{"key": key, "header": header} for key, header in OCR_FIELDS],
        "field_choices": {
            field: dict(choices)
            for field, choices in sorted(OCR_FIELD_CHOICES.items())
        },
        "field_spellings": {
            field: dict(spellings)
            for field, spellings in sorted(OCR_FIELD_SPELLINGS.items())
        },
        "field_characters": dict(sorted(OCR_FIELD_CHARACTERS.items())),
        "allowed_scripts": [[low, high] for low, high in OCR_ALLOWED_SCRIPTS],
        "character_folds": dict(sorted(OCR_CHARACTER_FOLDS.items())),
        "circled_digit_folds": dict(sorted(OCR_CIRCLED_DIGIT_FOLDS.items())),
        "number_only_fields": list(OCR_NUMBER_ONLY_FIELDS),
        "paired_fields": [
            {
                "letter": letter_field,
                "wording": wording_field,
                "words": {answer: list(words) for answer, words in table.items()},
            }
            for letter_field, wording_field, table in OCR_PAIRED_FIELDS
        ],
    }
