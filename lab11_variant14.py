"""
Лабораторная работа № 11. Вариант 14.
Наивный поиск, алгоритм КМП, расстояние Левенштейна, сопоставление записей.
Образец для ручного расчёта: aabaaba. Тип текста: с повторами,
n = 10^4, 10^5, 5·10^5. Порог сходства 0,70.
Дополнительное задание Л7: расширенная нормализация (транслитерация,
сокращения, кавычки разных видов) и прирост доли точных совпадений.
"""
import gc
import json
import random
import re
import statistics
import time

PATTERN = "aabaaba"
SIZES = [10_000, 100_000, 500_000]
THRESHOLD = 0.70


# ------------------------------------------------------- поиск подстроки
def naive_search(text, pattern):
    """Наивный поиск. Возвращает (позиции, число сравнений символов)."""
    n, m = len(text), len(pattern)
    positions, comparisons = [], 0
    for i in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1
            if text[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            positions.append(i)
    return positions, comparisons


def prefix_function(pattern):
    """π[i] — длина наибольшего собственного префикса P[0..i], равного его суффиксу."""
    pi = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = pi[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        pi[i] = k
    return pi


def kmp_search(text, pattern):
    """Алгоритм Кнута — Морриса — Пратта. Θ(n + m)."""
    n, m = len(text), len(pattern)
    pi = prefix_function(pattern)
    positions, comparisons = [], 0
    k = 0
    for i in range(n):
        while k > 0 and text[i] != pattern[k]:
            comparisons += 1
            k = pi[k - 1]
        comparisons += 1
        if text[i] == pattern[k]:
            k += 1
        if k == m:
            positions.append(i - m + 1)
            k = pi[k - 1]
    return positions, comparisons


def reference_positions(text, pattern):
    """Эталон для проверки: все (в том числе перекрывающиеся) вхождения через re."""
    return [m.start() for m in re.finditer("(?=" + re.escape(pattern) + ")", text)]


# ------------------------------------------------ расстояние Левенштейна
def levenshtein(a, b):
    """Θ(nm) времени, Θ(min(n, m)) памяти."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,                    # удаление
                               current[j - 1] + 1,                 # вставка
                               previous[j - 1] + (ca != cb)))      # замена
        previous = current
    return previous[-1]


def levenshtein_table(a, b):
    """Полная таблица D[i][j] (для демонстрации и проверки). Θ(nm) памяти."""
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i
    for j in range(len(b) + 1):
        D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return D


def similarity(a, b):
    """Нормированная мера сходства из [0; 1]."""
    if not a and not b:
        return 1.0
    return 1 - levenshtein(a, b) / max(len(a), len(b))


# ------------------------------------------------------------ нормализация
LEGAL_FORMS = {"ооо", "оао", "зао", "пао", "ао", "ип"}


def normalize_basic(name):
    """Базовая нормализация: регистр, «ё», кавычки и знаки препинания, правовые формы."""
    s = name.lower().replace("ё", "е")
    s = re.sub(r"[\"'«»„“”.,()]", " ", s)
    return " ".join(w for w in s.split() if w not in LEGAL_FORMS)


TRANSLIT_MULTI = [("shch", "щ"), ("sch", "щ"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"), ("ch", "ч"),
                  ("sh", "ш"), ("yu", "ю"), ("ya", "я"), ("yo", "е"), ("iy", "ий"), ("yy", "ый"),
                  ("ey", "ей")]
TRANSLIT_SINGLE = dict(a="а", b="б", c="к", d="д", e="е", f="ф", g="г", h="х", i="и", j="й", k="к",
                       l="л", m="м", n="н", o="о", p="п", q="к", r="р", s="с", t="т", u="у",
                       v="в", w="в", x="кс", y="ы", z="з")
ABBREVIATIONS = {"ул": "улица", "г": "город", "д": "дом", "просп": "проспект", "пер": "переулок",
                 "наб": "набережная", "корп": "корпус", "стр": "строение", "ш": "шоссе",
                 "обл": "область"}
QUOTES = "«»„“”‟‘’‚`´'\"‹›"


def transliterate(word):
    """Латиница → кириллица по упрощённой таблице (слово целиком из латинских букв)."""
    out, i = [], 0
    while i < len(word):
        for lat, cyr in TRANSLIT_MULTI:
            if word.startswith(lat, i):
                out.append(cyr)
                i += len(lat)
                break
        else:
            ch = word[i]
            out.append("э" if (ch == "e" and i == 0) else TRANSLIT_SINGLE[ch])
            i += 1
    return "".join(out)


def normalize_extended(name):
    """Расширенная нормализация (задание Л7): кавычки всех видов, транслитерация,
    раскрытие сокращений, затем базовые шаги."""
    s = name.lower().replace("ё", "е")
    s = re.sub(r"\bпр-к?т\b\.?", " проспект ", s)             # «пр-т», «пр-кт» до замены дефиса
    s = "".join(" " if c in QUOTES else c for c in s)
    s = re.sub(r"[.,;:()/\\\-–—]", " ", s)
    tokens = []
    for t in s.split():
        if re.fullmatch("[a-z]+", t):
            t = transliterate(t)
        tokens.append(ABBREVIATIONS.get(t, t))
    return " ".join(w for w in tokens if w not in LEGAL_FORMS)


# ------------------------------------------------------ сопоставление записей
def match_records(list_a, list_b, threshold, normalize):
    """Для каждой записи A — лучшая запись B: точное совпадение после нормализации
    или наибольшее сходство не ниже порога. Возвращает (B-индекс | None, сходство, вид)."""
    norm_b = [normalize(b) for b in list_b]
    result = []
    for a in list_a:
        na = normalize(a)
        best, best_sim, kind = None, 0.0, ""
        for j, nb in enumerate(norm_b):
            if na == nb:
                best, best_sim, kind = j, 1.0, "точное"
                break
            s = similarity(na, nb)
            if s > best_sim:
                best, best_sim, kind = j, s, "нечёткое"
        if best_sim >= threshold:
            result.append((best, best_sim, kind))
        else:
            result.append((None, best_sim, "не найдено"))
    return result


# Система А (учёт) и система Б (CRM). Эталонное соответствие: A[i] ↔ B[i] для i < 20.
SYSTEM_A = [
    'ООО «Ромашка», г. Москва, ул. Ленина, д. 5', 'АО «Василёк», г. Казань, ул. Баумана, д. 12',
    'ПАО «Гранит», г. Самара, пр-т Ленина, д. 3', 'ООО «Северный ветер», г. Омск, ул. Мира, д. 8',
    'ЗАО «Орион», г. Пермь, ул. Комсомольская, д. 21', 'ООО «Горизонт», г. Тула, ул. Советская, д. 7',
    'ООО «Сатурн», г. Уфа, ул. Пушкина, д. 15', 'АО «Лукоморье», г. Томск, пер. Тихий, д. 2',
    'ООО «Техносервис», г. Рязань, ул. Гагарина, д. 40', 'ОАО «Прогресс», г. Курск, ул. Дзержинского, д. 9',
    'ООО «Вектор», г. Воронеж, ул. Кирова, д. 33', 'АО «Эталон», г. Липецк, ул. Первомайская, д. 18',
    'ООО «Меридиан», г. Ярославль, ул. Свободы, д. 6', 'ПАО «Звезда», г. Иваново, пр-т Строителей, д. 11',
    'ООО «Колос», г. Орёл, ул. Ленина, д. 27', 'ООО «Заря», г. Пенза, ул. Московская, д. 4',
    'АО «Алмаз», г. Брянск, ул. Фокина, д. 19', 'ООО «Искра», г. Тамбов, ул. Интернациональная, д. 50',
    'ЗАО «Фортуна», г. Киров, ул. Труда, д. 23', 'ООО «Мастерская Иванова», г. Сочи, ул. Навагинская, д. 1',
    'ООО «Янтарь», г. Калининград, ул. Л. Чайкиной, д. 10', 'АО «Нева-Плюс», г. Псков, ул. Некрасова, д. 5',
    'ООО «Кедр», г. Чита, ул. Амурская, д. 14']
SYSTEM_B = [
    'Ромашка, ООО, город Москва, улица Ленина, дом 5', 'ВАСИЛЕК АО, г. Казань, ул. Баумана, д. 12',
    'ПАО "Гранит", город Самара, проспект Ленина, дом 3', 'Северный Ветер ООО, г. Омск, ул. Мира, д. 8',
    'Orion ZAO, г. Пермь, ул. Комсомольская, д. 21', 'Gorizont OOO, г. Тула, ул. Советская, д. 7',
    'ООО ‘Сатурн’, г. Уфа, ул. Пушкина, д. 15', 'АО „Лукоморье“, г. Томск, переулок Тихий, д. 2',
    'Технасервис ООО, г. Рязань, ул. Гагарина, д. 40', 'ОАО «Прогрес», г. Курск, улица Дзержинского, дом 9',
    'ВЕКТОР ООО, Воронеж, ул. Кирова, д. 33', 'Etalon AO, г. Липецк, ул. Первомайская, д. 18',
    'ООО «Меридиан», г. Ярославль, ул. Свободы, дом 6', 'ПАО Звезда, город Иваново, проспект Строителей, д. 11',
    'ООО "Колос", г. Орел, ул. Ленина, д. 27', 'Zarya OOO, г. Пенза, ул. Московская, д. 4',
    'АО Алмаз, г. Брянск, ул. Фокина, д. 19', 'ООО «Искра» г. Тамбов ул. Интернациональная д. 50',
    'Фортуна ЗАО, Киров, ул. Труда, 23', 'Мастерская Иванова ООО, г. Сочи, ул. Навагинская, д. 1',
    'ООО «Ромашка-Сервис», г. Санкт-Петербург, ул. Блохина, д. 7', 'ПАО «Гранит-Строй», г. Самара, ул. Мичурина, д. 9',
    'ООО «Нева», г. Псков, ул. Гоголя, д. 2', 'ООО «Янтарный берег», г. Калининград, ул. Ленина, д. 10',
    'ООО «Кедровый бор», г. Чита, ул. Ленина, д. 14']
TRUTH = list(range(20)) + [None, None, None]


def evaluate(results, truth=TRUTH):
    """Точность и полнота: верно — найден правильный партнёр либо правильно отказано."""
    tp = sum(1 for r, t in zip(results, truth) if t is not None and r[0] == t)
    fp = sum(1 for r, t in zip(results, truth) if r[0] is not None and r[0] != t)
    fn = sum(1 for r, t in zip(results, truth) if t is not None and r[0] != t)
    exact = sum(1 for r, t in zip(results, truth) if t is not None and r[0] == t and r[2] == "точное")
    p = tp / (tp + fp) if tp + fp else 1.0
    rc = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "exact": exact, "precision": p, "recall": rc}


def typo(s, rng):
    """Случайная опечатка: замена, удаление или вставка символа."""
    if len(s) < 3:
        return s
    i = rng.randrange(len(s))
    op = rng.choice("sdi")
    if op == "s":
        return s[:i] + rng.choice("абвгдежз") + s[i + 1:]
    if op == "d":
        return s[:i] + s[i + 1:]
    return s[:i] + rng.choice("абвгдежз") + s[i:]


# ------------------------------------------------------------ замеры
def timed(func, *args, repeats=3):
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["search"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylabel in ((axes[0], "t", "время, с"), (axes[1], "c", "число сравнений символов")):
        for scen, ls in (("periodic", "-"), ("worst", "--")):
            for alg, col in (("naive", "C0"), ("kmp", "C1")):
                pts = [(r["n"], r[scen][alg][key]) for r in rows]
                lab = f"{'наивный' if alg == 'naive' else 'КМП'}, {'период aab' if scen == 'periodic' else 'aⁿ / a⁵⁰b'}"
                ax.loglog(*zip(*pts), ls, color=col, marker="o", label=lab)
        ax.set_xlabel("длина текста n"); ax.set_ylabel(ylabel); ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("img/lr11_search.png", dpi=170); plt.close()

    sw = res["sweep"]
    plt.figure(figsize=(6.6, 4.2))
    ths = [s["threshold"] for s in sw]
    plt.plot(ths, [s["precision"] * 100 for s in sw], "o-", label="точность")
    plt.plot(ths, [s["recall"] * 100 for s in sw], "s-", label="полнота")
    plt.axvline(THRESHOLD, color="gray", ls="--", label="порог варианта (0,70)")
    plt.xlabel("порог сходства"); plt.ylabel("%"); plt.grid(True, alpha=.3); plt.legend()
    plt.title("Качество сопоставления и порог")
    plt.tight_layout(); plt.savefig("img/lr11_threshold.png", dpi=170); plt.close()

    sc = res["scaling"]
    plt.figure(figsize=(6.4, 4.2))
    plt.loglog([r["N"] for r in sc], [r["t"] for r in sc], "o-")
    plt.xlabel("число записей N в каждой системе"); plt.ylabel("время сопоставления, с")
    plt.title("Сравнение «каждый с каждым»"); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig("img/lr11_scaling.png", dpi=170); plt.close()


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    res = {}
    pi = prefix_function(PATTERN)
    assert pi == [0, 1, 0, 1, 2, 3, 4]
    res["pi"] = pi
    print("π для", PATTERN, pi)

    rng = random.Random(14)
    res["search"] = []
    print(f"{'n':>8}{'текст':>10}{'наивн., с':>11}{'КМП, с':>9}{'сравн. (наив.)':>16}{'сравн. (КМП)':>14}{'вхожд.':>9}")
    for n in SIZES:
        texts = {"periodic": ("aab" * (n // 3 + 1))[:n],
                 "random": "".join(rng.choice("ab") for _ in range(n)),
                 "worst": "a" * n}
        patterns = {"periodic": PATTERN, "random": PATTERN, "worst": "a" * 50 + "b"}
        row = {"n": n}
        for scen, text in texts.items():
            p = patterns[scen]
            tn, (pos_n, cn) = timed(naive_search, text, p)
            tk, (pos_k, ck) = timed(kmp_search, text, p)
            ref = reference_positions(text, p)
            assert pos_n == pos_k == ref, scen
            row[scen] = {"naive": {"t": tn, "c": cn}, "kmp": {"t": tk, "c": ck}, "occ": len(ref), "pattern": p}
            print(f"{n:>8}{scen:>10}{tn:>11.4f}{tk:>9.4f}{cn:>16}{ck:>14}{len(ref):>9}", flush=True)
        res["search"].append(row)

    a, b = "платёж", "плата"
    D = levenshtein_table(a, b)
    assert D[-1][-1] == levenshtein(a, b) == 2
    res["table"] = {"a": a, "b": b, "D": D}
    rng = random.Random(3)
    for _ in range(300):                               # сверка двух реализаций
        x = "".join(rng.choice("абвг") for _ in range(rng.randint(0, 9)))
        y = "".join(rng.choice("абвг") for _ in range(rng.randint(0, 9)))
        assert levenshtein(x, y) == levenshtein_table(x, y)[-1][-1]
    pairs = [("кот", "скот"), ("Ромашка", "Ромашки"), ("аналитика", "аналитик"), ("платёж", "плата"),
             ("договор", "договорённость")]
    res["pairs"] = [(x, y, levenshtein(x, y), similarity(x, y)) for x, y in pairs]
    print(res["pairs"])

    res["match"] = {}
    for name, norm in (("basic", normalize_basic), ("extended", normalize_extended)):
        out = match_records(SYSTEM_A, SYSTEM_B, THRESHOLD, norm)
        res["match"][name] = {"rows": out, "eval": evaluate(out)}
        print(name, res["match"][name]["eval"])
    res["norm"] = {"a_basic": [normalize_basic(s) for s in SYSTEM_A], "b_basic": [normalize_basic(s) for s in SYSTEM_B],
                   "a_ext": [normalize_extended(s) for s in SYSTEM_A], "b_ext": [normalize_extended(s) for s in SYSTEM_B]}

    res["sweep"] = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out = match_records(SYSTEM_A, SYSTEM_B, th, normalize_extended)
        res["sweep"].append({"threshold": th, **evaluate(out)})
        print(th, res["sweep"][-1])

    rng = random.Random(5)
    res["scaling"] = []
    for N in (50, 100, 200, 400):
        la = [SYSTEM_A[i % len(SYSTEM_A)] + f" {i}" for i in range(N)]
        lb = [typo(SYSTEM_B[i % len(SYSTEM_B)], rng) + f" {i}" for i in range(N)]
        t, _ = timed(match_records, la, lb, THRESHOLD, normalize_extended, repeats=1)
        res["scaling"].append({"N": N, "pairs": N * N, "t": t})
        print(N, t, flush=True)

    with open("res/lr11.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    make_plots(res)
