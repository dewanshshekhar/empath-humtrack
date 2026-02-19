import os
import re
import sys
import numpy as np
from collections import Counter
from collections import defaultdict


from py3langid.langid import LanguageIdentifier, MODEL_FILE

from empath.language_segmentation.utils.num import num2str




# Word segmentation function:
# automatically identify and split the words (Chinese/English/Japanese/Korean) in the article or sentence according to different languages,
# making it more suitable for TTS processing.
# This code is designed for front-end text multi-lingual mixed annotation distinction, multi-language mixed training and inference of various TTS projects.
# This processing result is mainly for (Chinese = zh, Japanese = ja, English = en, Korean = ko), and can actually support up to 97 different language mixing processing.
class LangSSML:

    def __init__(self):
        # 纯数字
        self._zh_numerals_number = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }

    # 将2024/8/24, 2024-08, 08-24, 24 标准化“年月日”
    # Standardize 2024/8/24, 2024-08, 08-24, 24 to "year-month-day"
    def _format_chinese_data(self, date_str: str):
        input_date = date_str
        if date_str is None or date_str.strip() == "":
            return ""
        date_str = re.sub(r"[\/\._|年|月]", "-", date_str)
        date_str = re.sub(r"日", r"", date_str)
        date_arrs = date_str.split(" ")
        if len(date_arrs) == 1 and ":" in date_arrs[0]:
            time_str = date_arrs[0]
            date_arrs = []
        else:
            time_str = date_arrs[1] if len(date_arrs) >= 2 else ""

        def nonZero(num, cn, func=None):
            if func is not None:
                num = func(num)
            return f"{num}{cn}" if num is not None and num != "" and num != "0" else ""

        f_number = self.to_chinese_number
        f_currency = self.to_chinese_currency
        # year, month, day
        year_month_day = ""
        if len(date_arrs) > 0:
            year, month, day = "", "", ""
            parts = date_arrs[0].split("-")
            if len(parts) == 3:  # YYYY-MM-DD
                year, month, day = parts
            elif len(parts) == 2:  # MM-DD 或 YYYY-MM
                if len(parts[0]) == 4:  
                    year, month = parts
                else:
                    month, day = parts  
            elif len(parts[0]) > 0:  
                if len(parts[0]) == 4:
                    year = parts[0]
                else:
                    day = parts[0]
            year, month, day = (
                nonZero(year, "年", f_number),
                nonZero(month, "月", f_currency),
                nonZero(day, "日", f_currency),
            )
            year_month_day = re.sub(r"([年|月|日])+", r"\1", f"{year}{month}{day}")
        # hours, minutes, seconds
        time_str = re.sub(r"[\/\.\-：_]", ":", time_str)
        time_arrs = time_str.split(":")
        hours, minutes, seconds = "", "", ""
        if len(time_arrs) == 3:  # H/M/S
            hours, minutes, seconds = time_arrs
        elif len(time_arrs) == 2:  # H/M
            hours, minutes = time_arrs
        elif len(time_arrs[0]) > 0:
            hours = f"{time_arrs[0]}点"  # H
        if len(time_arrs) > 1:
            hours, minutes, seconds = (
                nonZero(hours, "点", f_currency),
                nonZero(minutes, "分", f_currency),
                nonZero(seconds, "秒", f_currency),
            )
        hours_minutes_seconds = re.sub(
            r"([点|分|秒])+", r"\1", f"{hours}{minutes}{seconds}"
        )
        output_date = f"{year_month_day}{hours_minutes_seconds}"
        return output_date

    
    def to_chinese_number(self, num: str):
        pattern = r"(\d+)"
        zh_numerals = self._zh_numerals_number
        arrs = re.split(pattern, num)
        output = ""
        for item in arrs:
            if re.match(pattern, item):
                output += "".join(
                    zh_numerals[digit] if digit in zh_numerals else ""
                    for digit in str(item)
                )
            else:
                output += item
        output = output.replace(".", "点")
        return output

    
    def to_chinese_telephone(self, num: str):
        output = self.to_chinese_number(num.replace("+86", ""))  # zh +86
        output = output.replace("一", "幺")
        return output

    
    def to_chinese_currency(self, num: str):
        pattern = r"(\d+)"
        arrs = re.split(pattern, num)
        output = ""
        for item in arrs:
            if re.match(pattern, item):
                output += num2str(item)
            else:
                output += item
        output = output.replace(".", "点")
        return output

    
    def to_chinese_date(self, num: str):
        chinese_date = self._format_chinese_data(num)
        return chinese_date


class LangSegment:

    def __init__(self):

        self.langid = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)

        self._text_cache = None
        self._text_lasts = None
        self._text_langs = None
        self._lang_count = None
        self._lang_eos = None
        self.SYMBOLS_PATTERN = r"(<([a-zA-Z|-]*)>(.*?)<\/*[a-zA-Z|-]*>)"

        self.DEFAULT_FILTERS = ["zh", "ja", "ko", "en"]

        self.Langfilters = self.DEFAULT_FILTERS[:]  # 创建副本

        self.isLangMerge = True
        self.EnablePreview = False

        self.LangPriorityThreshold = 0.89

        self.keepPinyin = False

        # DEFINITION
        self.PARSE_TAG = re.compile(r"(⑥\$*\d+[\d]{6,}⑥)")

        self.LangSSML = LangSSML()

    def _clears(self):
        self._text_cache = None
        self._text_lasts = None
        self._text_langs = None
        self._text_waits = None
        self._lang_count = None
        self._lang_eos = None

    def _is_english_word(self, word):
        return bool(re.match(r"^[a-zA-Z]+$", word))

    def _is_chinese(self, word):
        for char in word:
            if "\u4e00" <= char <= "\u9fff":
                return True
        return False

    def _is_japanese_kana(self, word):
        pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]+")
        matches = pattern.findall(word)
        return len(matches) > 0

    def _insert_english_uppercase(self, word):
        modified_text = re.sub(r"(?<!\b)([A-Z])", r" \1", word)
        modified_text = modified_text.strip("-")
        return modified_text + " "

    def _split_camel_case(self, word):
        return re.sub(r"(?<!^)(?=[A-Z])", " ", word)

    def _statistics(self, language, text):
        # Language word statistics:
        # Chinese characters usually occupy double bytes
        if self._lang_count is None or not isinstance(self._lang_count, defaultdict):
            self._lang_count = defaultdict(int)
        lang_count = self._lang_count
        if not "|" in language:
            lang_count[language] += (
                int(len(text) * 2) if language == "zh" else len(text)
            )
        self._lang_count = lang_count

    def _clear_text_number(self, text):
        if text == "\n":
            return text, False  # Keep Line Breaks
        clear_text = re.sub(r"([^\w\s]+)", "", re.sub(r"\n+", "", text)).strip()
        is_number = len(re.sub(re.compile(r"(\d+)"), "", clear_text)) == 0
        return clear_text, is_number

    def _saveData(self, words, language: str, text: str, score: float, symbol=None):
        # Pre-detection
        clear_text, is_number = self._clear_text_number(text)
        # Merge the same language and save the results
        preData = words[-1] if len(words) > 0 else None
        if symbol is not None:
            pass
        elif preData is not None and preData["symbol"] is None:
            if len(clear_text) == 0:
                language = preData["lang"]
            elif is_number == True:
                language = preData["lang"]
            _, pre_is_number = self._clear_text_number(preData["text"])
            if preData["lang"] == language:
                self._statistics(preData["lang"], text)
                text = preData["text"] + text
                preData["text"] = text
                return preData
            elif pre_is_number == True:
                text = f'{preData["text"]}{text}'
                words.pop()
        elif is_number == True:
            priority_language = self._get_filters_string()[:2]
            if priority_language in "ja-zh-en-ko-fr-vi":
                language = priority_language
        data = {"lang": language, "text": text, "score": score, "symbol": symbol}
        filters = self.Langfilters
        if (
            filters is None
            or len(filters) == 0
            or "?" in language
            or language in filters
            or language in filters[0]
            or filters[0] == "*"
            or filters[0] in "alls-mixs-autos"
        ):
            words.append(data)
            self._statistics(data["lang"], data["text"])
        return data

    def _addwords(self, words, language, text, score, symbol=None):
        if text == "\n":
            pass  # Keep Line Breaks
        elif text is None or len(text.strip()) == 0:
            return True
        if language is None:
            language = ""
        language = language.lower()
        if language == "en":
            text = self._insert_english_uppercase(text)
        # text = re.sub(r'[(（）)]', ',' , text) # Keep it.
        text_waits = self._text_waits
        ispre_waits = len(text_waits) > 0
        preResult = text_waits.pop() if ispre_waits else None
        if preResult is None:
            preResult = words[-1] if len(words) > 0 else None
        if preResult and ("|" in preResult["lang"]):
            pre_lang = preResult["lang"]
            if language in pre_lang:
                preResult["lang"] = language = language.split("|")[0]
            else:
                preResult["lang"] = pre_lang.split("|")[0]
            if ispre_waits:
                preResult = self._saveData(
                    words,
                    preResult["lang"],
                    preResult["text"],
                    preResult["score"],
                    preResult["symbol"],
                )
        pre_lang = preResult["lang"] if preResult else None
        if ("|" in language) and (
            pre_lang and not pre_lang in language and not "…" in language
        ):
            language = language.split("|")[0]
        if "|" in language:
            self._text_waits.append(
                {"lang": language, "text": text, "score": score, "symbol": symbol}
            )
        else:
            self._saveData(words, language, text, score, symbol)
        return False

    def _get_prev_data(self, words):
        data = words[-1] if words and len(words) > 0 else None
        if data:
            return (data["lang"], data["text"])
        return (None, "")

    def _match_ending(self, input, index):
        if input is None or len(input) == 0:
            return False, None
        input = re.sub(r"\s+", "", input)
        if len(input) == 0 or abs(index) > len(input):
            return False, None
        ending_pattern = re.compile(r'([「」“”‘’"\':：。.！!?．？])')
        return ending_pattern.match(input[index]), input[index]

    def _cleans_text(self, cleans_text):
        cleans_text = re.sub(r"(.*?)([^\w]+)", r"\1 ", cleans_text)
        cleans_text = re.sub(r"(.)\1+", r"\1", cleans_text)
        return cleans_text.strip()

    def _mean_processing(self, text: str):
        if text is None or (text.strip()) == "":
            return None, 0.0
        arrs = self._split_camel_case(text).split(" ")
        langs = []
        for t in arrs:
            if len(t.strip()) <= 3:
                continue
            language, score = self.langid.classify(t)
            langs.append({"lang": language})
        if len(langs) == 0:
            return None, 0.0
        return Counter([item["lang"] for item in langs]).most_common(1)[0][0], 1.0

    def _lang_classify(self, cleans_text):
        language, score = self.langid.classify(cleans_text)
        # fix: Huggingface is np.float32
        if (
            score is not None
            and isinstance(score, np.generic)
            and hasattr(score, "item")
        ):
            score = score.item()
        score = round(score, 3)
        return language, score

    def _get_filters_string(self):
        filters = self.Langfilters
        return "-".join(filters).lower().strip() if filters is not None else ""

    def _parse_language(self, words, segment):
        LANG_JA = "ja"
        LANG_ZH = "zh"
        LANG_ZH_JA = f"{LANG_ZH}|{LANG_JA}"
        LANG_JA_ZH = f"{LANG_JA}|{LANG_ZH}"
        language = LANG_ZH
        regex_pattern = re.compile(r"([^\w\s]+)")
        lines = regex_pattern.split(segment)
        lines_max = len(lines)
        LANG_EOS = self._lang_eos
        for index, text in enumerate(lines):
            if len(text) == 0:
                continue
            EOS = index >= (lines_max - 1)
            nextId = index + 1
            nextText = lines[nextId] if not EOS else ""
            nextPunc = (
                len(re.sub(regex_pattern, "", re.sub(r"\n+", "", nextText)).strip())
                == 0
            )
            textPunc = (
                len(re.sub(regex_pattern, "", re.sub(r"\n+", "", text)).strip()) == 0
            )
            if not EOS and (
                textPunc == True or (len(nextText.strip()) >= 0 and nextPunc == True)
            ):
                lines[nextId] = f"{text}{nextText}"
                continue
            number_tags = re.compile(r"(⑥\d{6,}⑥)")
            cleans_text = re.sub(number_tags, "", text)
            cleans_text = re.sub(r"\d+", "", cleans_text)
            cleans_text = self._cleans_text(cleans_text)
            # fix:Langid's recognition of short sentences is inaccurate, and it is spliced longer.
            if not EOS and len(cleans_text) <= 2:
                lines[nextId] = f"{text}{nextText}"
                continue
            language, score = self._lang_classify(cleans_text)
            prev_language, prev_text = self._get_prev_data(words)
            if language != LANG_ZH and all(
                "\u4e00" <= c <= "\u9fff" for c in re.sub(r"\s", "", cleans_text)
            ):
                language, score = LANG_ZH, 1
            if len(cleans_text) <= 5 and self._is_chinese(cleans_text):
                filters_string = self._get_filters_string()
                if score < self.LangPriorityThreshold and len(filters_string) > 0:
                    index_ja, index_zh = filters_string.find(
                        LANG_JA
                    ), filters_string.find(LANG_ZH)
                    if index_ja != -1 and index_ja < index_zh:
                        language = LANG_JA
                    elif index_zh != -1 and index_zh < index_ja:
                        language = LANG_ZH
                if self._is_japanese_kana(cleans_text):
                    language = LANG_JA
                elif len(cleans_text) > 2 and score > 0.90:
                    pass
                elif EOS and LANG_EOS:
                    language = LANG_ZH if len(cleans_text) <= 1 else language
                else:
                    LANG_UNKNOWN = (
                        LANG_ZH_JA
                        if language == LANG_ZH
                        or (len(cleans_text) <= 2 and prev_language == LANG_ZH)
                        else LANG_JA_ZH
                    )
                    match_end, match_char = self._match_ending(text, -1)
                    referen = (
                        prev_language in LANG_UNKNOWN or LANG_UNKNOWN in prev_language
                        if prev_language
                        else False
                    )
                    if match_char in "。.":
                        language = (
                            prev_language if referen and len(words) > 0 else language
                        )
                    else:
                        language = f"{LANG_UNKNOWN}|…"
            text, *_ = re.subn(number_tags, self._restore_number, text)
            self._addwords(words, language, text, score)

    def _process_symbol_SSML(self, words, data):
        tag, match = data
        language = SSML = match[1]
        text = match[2]
        score = 1.0
        if SSML == "telephone":
            language = "zh"
            text = self.LangSSML.to_chinese_telephone(text)
        elif SSML == "number":
            language = "zh"
            text = self.LangSSML.to_chinese_number(text)
        elif SSML == "currency":
            language = "zh"
            text = self.LangSSML.to_chinese_currency(text)
        elif SSML == "date":
            language = "zh"
            text = self.LangSSML.to_chinese_date(text)
        self._addwords(words, language, text, score, SSML)

    # ----------------------------------------------------------
    def _restore_number(self, matche):
        value = matche.group(0)
        text_cache = self._text_cache
        if value in text_cache:
            process, data = text_cache[value]
            tag, match = data
            value = match
        return value

    def _pattern_symbols(self, item, text):
        if text is None:
            return text
        tag, pattern, process = item
        matches = pattern.findall(text)
        if len(matches) == 1 and "".join(matches[0]) == text:
            return text
        for i, match in enumerate(matches):
            key = f"⑥{tag}{i:06d}⑥"
            text = re.sub(pattern, key, text, count=1)
            self._text_cache[key] = (process, (tag, match))
        return text

    def _process_symbol(self, words, data):
        tag, match = data
        language = match[1]
        text = match[2]
        score = 1.0
        filters = self._get_filters_string()
        if language not in filters:
            self._process_symbol_SSML(words, data)
        else:
            self._addwords(words, language, text, score, True)

    def _process_english(self, words, data):
        tag, match = data
        text = match[0]
        filters = self._get_filters_string()
        priority_language = filters[:2]
        # Preview feature, other language segmentation processing
        enablePreview = self.EnablePreview
        if enablePreview == True:
            # Experimental: Other language support
            regex_pattern = re.compile(r"(.*?[。.?？!！]+[\n]{,1})")
            lines = regex_pattern.split(text)
            for index, text in enumerate(lines):
                if len(text.strip()) == 0:
                    continue
                cleans_text = self._cleans_text(text)
                language, score = self._lang_classify(cleans_text)
                if language not in filters:
                    language, score = self._mean_processing(cleans_text)
                if language is None or score <= 0.0:
                    continue
                elif language in filters:
                    pass  # pass
                elif score >= 0.95:
                    continue  # High score, but not in the filter, excluded.
                elif score <= 0.15 and filters[:2] == "fr":
                    language = priority_language
                else:
                    language = "en"
                self._addwords(words, language, text, score)
        else:
            # Default is English
            language, score = "en", 1.0
            self._addwords(words, language, text, score)

    def _process_Russian(self, words, data):
        tag, match = data
        text = match[0]
        language = "ru"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_Thai(self, words, data):
        tag, match = data
        text = match[0]
        language = "th"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_korean(self, words, data):
        tag, match = data
        text = match[0]
        language = "ko"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_quotes(self, words, data):
        tag, match = data
        text = "".join(match)
        childs = self.PARSE_TAG.findall(text)
        if len(childs) > 0:
            self._process_tags(words, text, False)
        else:
            cleans_text = self._cleans_text(match[1])
            if len(cleans_text) <= 5:
                self._parse_language(words, text)
            else:
                language, score = self._lang_classify(cleans_text)
                self._addwords(words, language, text, score)

    def _process_pinyin(self, words, data):
        tag, match = data
        text = match
        language = "zh"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_number(self, words, data):  # "$0" process only
        """
        Numbers alone cannot accurately identify language.
        Because numbers are universal in all languages.
        So it won't be executed here, just for testing.
        """
        tag, match = data
        language = words[0]["lang"] if len(words) > 0 else "zh"
        text = match
        score = 0.0
        self._addwords(words, language, text, score)

    def _process_tags(self, words, text, root_tag):
        text_cache = self._text_cache
        segments = re.split(self.PARSE_TAG, text)
        segments_len = len(segments) - 1
        for index, text in enumerate(segments):
            if root_tag:
                self._lang_eos = index >= segments_len
            if self.PARSE_TAG.match(text):
                process, data = text_cache[text]
                if process:
                    process(words, data)
            else:
                self._parse_language(words, text)
        return words

    def _merge_results(self, words):
        new_word = []
        for index, cur_data in enumerate(words):
            if "symbol" in cur_data:
                del cur_data["symbol"]
            if index == 0:
                new_word.append(cur_data)
            else:
                pre_data = new_word[-1]
                if cur_data["lang"] == pre_data["lang"]:
                    pre_data["text"] = f'{pre_data["text"]}{cur_data["text"]}'
                else:
                    new_word.append(cur_data)
        return new_word

    def _parse_symbols(self, text):
        TAG_NUM = "00"  # "00" => default channels , "$0" => testing channel
        TAG_S1, TAG_S2, TAG_P1, TAG_P2, TAG_EN, TAG_KO, TAG_RU, TAG_TH = (
            "$1",
            "$2",
            "$3",
            "$4",
            "$5",
            "$6",
            "$7",
            "$8",
        )
        TAG_BASE = re.compile(rf'(([【《（(“‘"\']*[LANGUAGE]+[\W\s]*)+)')
        # Get custom language filter
        filters = self.Langfilters
        filters = filters if filters is not None else ""
        enablePreview = self.EnablePreview
        if "fr" in filters or "vi" in filters:
            enablePreview = True
        self.EnablePreview = enablePreview
        # 实验性：法语字符支持。Prise en charge des caractères français
        RE_FR = "" if not enablePreview else "àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ"
        # 实验性：越南语字符支持。Hỗ trợ ký tự tiếng Việt
        RE_VI = (
            ""
            if not enablePreview
            else "đơưăáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựôâêơưỷỹ"
        )
        # -------------------------------------------------------------------------------------------------------
        # Basic options:
        process_list = [
            (
                TAG_S1,
                re.compile(self.SYMBOLS_PATTERN),
                self._process_symbol,
            ),  # Symbol Tag
            (
                TAG_KO,
                re.compile(re.sub(r"LANGUAGE", f"\uac00-\ud7a3", TAG_BASE.pattern)),
                self._process_korean,
            ),  # Korean words
            (
                TAG_TH,
                re.compile(re.sub(r"LANGUAGE", f"\u0e00-\u0e7f", TAG_BASE.pattern)),
                self._process_Thai,
            ),  # Thai words support.
            (
                TAG_RU,
                re.compile(re.sub(r"LANGUAGE", f"А-Яа-яЁё", TAG_BASE.pattern)),
                self._process_Russian,
            ),  # Russian words support.
            (
                TAG_NUM,
                re.compile(r"(\W*\d+\W+\d*\W*\d*)"),
                self._process_number,
            ),  # Number words, Universal in all languages, Ignore it.
            (
                TAG_EN,
                re.compile(
                    re.sub(r"LANGUAGE", f"a-zA-Z{RE_FR}{RE_VI}", TAG_BASE.pattern)
                ),
                self._process_english,
            ),  # English words + Other language support.
            (
                TAG_P1,
                re.compile(r'(["\'])(.*?)(\1)'),
                self._process_quotes,
            ),  # Regular quotes
            (
                TAG_P2,
                re.compile(
                    r"([\n]*[【《（(“‘])([^【《（(“‘’”)）》】]{3,})([’”)）》】][\W\s]*[\n]{,1})"
                ),
                self._process_quotes,
            ),  # Special quotes, There are left and right.
        ]
        # Extended options: Default False
        if self.keepPinyin == True:
            process_list.insert(
                1,
                (
                    TAG_S2,
                    re.compile(r"([\(（{](?:\s*\w*\d\w*\s*)+[}）\)])"),
                    self._process_pinyin,
                ),  # Chinese Pinyin Tag.
            )
        # -------------------------------------------------------------------------------------------------------
        words = []
        lines = re.findall(r".*\n*", re.sub(self.PARSE_TAG, "", text))
        for index, text in enumerate(lines):
            if len(text.strip()) == 0:
                continue
            self._lang_eos = False
            self._text_cache = {}
            for item in process_list:
                text = self._pattern_symbols(item, text)
            cur_word = self._process_tags([], text, True)
            if len(cur_word) == 0:
                continue
            cur_data = cur_word[0] if len(cur_word) > 0 else None
            pre_data = words[-1] if len(words) > 0 else None
            if (
                cur_data
                and pre_data
                and cur_data["lang"] == pre_data["lang"]
                and cur_data["symbol"] == False
                and pre_data["symbol"]
            ):
                cur_data["text"] = f'{pre_data["text"]}{cur_data["text"]}'
                words.pop()
            words += cur_word
        if self.isLangMerge == True:
            words = self._merge_results(words)
        lang_count = self._lang_count
        if lang_count and len(lang_count) > 0:
            lang_count = dict(
                sorted(lang_count.items(), key=lambda x: x[1], reverse=True)
            )
            lang_count = list(lang_count.items())
            self._lang_count = lang_count
        return words

    def setfilters(self, filters):
        if self.Langfilters != filters:
            self._clears()
            self.Langfilters = filters

    def getfilters(self):
        return self.Langfilters

    def setPriorityThreshold(self, threshold: float):
        self.LangPriorityThreshold = threshold

    def getPriorityThreshold(self):
        return self.LangPriorityThreshold

    def getCounts(self):
        lang_count = self._lang_count
        if lang_count is not None:
            return lang_count
        text_langs = self._text_langs
        if text_langs is None or len(text_langs) == 0:
            return [("zh", 0)]
        lang_counts = defaultdict(int)
        for d in text_langs:
            lang_counts[d["lang"]] += (
                int(len(d["text"]) * 2) if d["lang"] == "zh" else len(d["text"])
            )
        lang_counts = dict(
            sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
        )
        lang_counts = list(lang_counts.items())
        self._lang_count = lang_counts
        return lang_counts

    def getTexts(self, text: str):
        if text is None or len(text.strip()) == 0:
            self._clears()
            return []
        # lasts
        text_langs = self._text_langs
        if self._text_lasts == text and text_langs is not None:
            return text_langs
        # parse
        self._text_waits = []
        self._lang_count = None
        self._text_lasts = text
        text = self._parse_symbols(text)
        self._text_langs = text
        return text

    def classify(self, text: str):
        return self.getTexts(text)


def printList(langlist):
    """
    功能：打印数组结果
    기능: 어레이 결과 인쇄
    機能:配列結果を印刷
    Function: Print array results
    """
    print("\n===================【打印结果】===================")
    if langlist is None or len(langlist) == 0:
        print("无内容结果,No content result")
        return
    for line in langlist:
        print(line)
    pass


def main():

    langsegment = LangSegment()
    langsegment.setfilters(["fr", "vi", "ja", "zh", "ko", "en", "ru", "th"])
    text = """
我喜欢在雨天里听音乐。
I enjoy listening to music on rainy days.
雨の日に音楽を聴くのが好きです。
비 오는 날에 음악을 듣는 것을 즐깁니다。
J'aime écouter de la musique les jours de pluie.
Tôi thích nghe nhạc vào những ngày mưa.
Мне нравится слушать музыку в дождливую погоду.
ฉันชอบฟังเพลงในวันที่ฝนตก
"""

    langlist = langsegment.getTexts(text)
    printList(langlist)

    print("\n===================【语种统计】===================")
    langCounts = langsegment.getCounts()
    print(langCounts, "\n")

    lang, count = langCounts[0]
    print(f"输入内容的主要语言为 = {lang} ，字数 = {count}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
