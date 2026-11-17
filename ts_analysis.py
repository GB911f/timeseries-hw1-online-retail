"""Функции для анализа и предобработки временного ряда продаж (датасет Online Retail, UCI).

Модуль разбит на блоки в порядке шагов анализа:
загрузка и очистка -> построение ряда -> пропуски и выбросы -> описательный анализ ->
декомпозиция и сезонность -> стационарность -> графики.
"""

from __future__ import annotations

import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

DATA_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
)
XLSX_NAME = "Online Retail.xlsx"

# Частоты агрегации. Неделя заканчивается в субботу: магазин работает с воскресенья
# по пятницу, поэтому каждая неделя содержит целый торговый цикл.
FREQS = {"D": "день", "W-SAT": "неделя", "MS": "месяц"}
WEEKDAYS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


# ---------------------------------------------------------------------------
# 1. Загрузка и очистка
# ---------------------------------------------------------------------------

def download_dataset(data_dir: str | Path) -> Path:
    """Скачивает Online Retail.xlsx в data_dir, если файла там ещё нет."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    xlsx = data_dir / XLSX_NAME
    if xlsx.exists():
        return xlsx
    for url in DATA_URLS:
        target = data_dir / Path(url).name.replace("%20", " ")
        try:
            urllib.request.urlretrieve(url, target)
        except OSError:
            continue
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as zf:
                zf.extract(XLSX_NAME, data_dir)
        if xlsx.exists():
            return xlsx
    raise RuntimeError(f"Не удалось скачать датасет; положите {XLSX_NAME} в {data_dir} вручную")


def load_transactions(xlsx_path: str | Path) -> pd.DataFrame:
    """Читает Excel-файл (≈10 с) и кэширует результат в pickle рядом с ним."""
    xlsx_path = Path(xlsx_path)
    cache = xlsx_path.with_suffix(".pkl")
    if cache.exists() and cache.stat().st_mtime >= xlsx_path.stat().st_mtime:
        return pd.read_pickle(cache)
    df = pd.read_excel(xlsx_path, dtype={"InvoiceNo": str, "StockCode": str})
    df["StockCode"] = df["StockCode"].str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    df.to_pickle(cache)
    return df


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оставляет только фактические продажи.

    Удаляются отменённые счета (номер начинается с «C»), строки с Quantity <= 0
    (возвраты и складские корректировки) и строки с UnitPrice <= 0 (списания, подарки,
    исправления остатков). Возвращает очищенную таблицу и отчёт о том, сколько строк
    отсеяно каждым правилом.
    """
    rules = {
        "Отменённый счёт (InvoiceNo начинается с C)": df["InvoiceNo"].str.startswith("C"),
        "Quantity <= 0": df["Quantity"] <= 0,
        "UnitPrice <= 0": df["UnitPrice"] <= 0,
    }
    drop = np.zeros(len(df), dtype=bool)
    rows = []
    for name, mask in rules.items():
        mask = mask.to_numpy()
        rows.append({"правило": name, "строк": int(mask.sum()), "из них новых": int((mask & ~drop).sum())})
        drop |= mask
    report = pd.DataFrame(rows)
    report.loc[len(report)] = {"правило": "Итого удалено", "строк": int(drop.sum()), "из них новых": int(drop.sum())}
    clean = df.loc[~drop].copy()
    clean["Revenue"] = clean["Quantity"] * clean["UnitPrice"]
    clean["Date"] = clean["InvoiceDate"].dt.normalize()
    return clean, report


def product_overview(clean: pd.DataFrame, top: int = 15, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Сводка по товарам: число дней с продажами, объём, число клиентов и стран."""
    g = clean[~clean["StockCode"].isin(exclude)].groupby("StockCode")
    out = pd.DataFrame({
        "описание": g["Description"].agg(lambda s: s.mode().iat[0] if s.notna().any() else ""),
        "дней с продажами": g["Date"].nunique(),
        "строк": g.size(),
        "штук": g["Quantity"].sum(),
        "выручка": g["Revenue"].sum().round(0),
        "клиентов": g["CustomerID"].nunique(),
        "стран": g["Country"].nunique(),
        "первая продажа": g["Date"].min(),
        "последняя продажа": g["Date"].max(),
    })
    return out.sort_values("дней с продажами", ascending=False).head(top)


# ---------------------------------------------------------------------------
# 2. Построение временного ряда
# ---------------------------------------------------------------------------

def trading_calendar(clean: pd.DataFrame) -> pd.DatetimeIndex:
    """Дни, когда в магазине была хотя бы одна продажа любого товара."""
    return pd.DatetimeIndex(np.sort(clean["Date"].unique()), name="Date")


def build_daily_series(clean: pd.DataFrame, stock_code: str, calendar: pd.DatetimeIndex,
                       value: str = "Quantity") -> pd.Series:
    """Дневной ряд товара на полном календаре.

    * торговый день без продаж товара -> 0 (магазин работал, спроса не было);
    * день, когда магазин не работал вовсе (субботы, праздники) -> NaN: наблюдения нет,
      и считать его нулевым спросом нельзя.
    """
    sales = clean.loc[clean["StockCode"] == stock_code].groupby("Date")[value].sum()
    full = pd.date_range(calendar.min(), calendar.max(), freq="D", name="Date")
    s = sales.reindex(full)
    s.loc[s.index.isin(calendar) & s.isna()] = 0.0
    s.name = value
    return s.astype(float)


def aggregate_series(daily: pd.Series, freq: str) -> pd.DataFrame:
    """Агрегирует дневной ряд до недели или месяца.

    Кроме суммы возвращает число торговых дней в периоде, их долю от нормы
    (6 дней в неделе, медиана по месяцам) и среднее за торговый день. Период без
    торговых дней получает NaN, а не 0.
    """
    if freq == "D":
        out = daily.to_frame("value")
        out["trading_days"] = daily.notna().astype(int)
        out["per_day"] = daily
        out["complete"] = daily.notna()
        return out
    r = daily.resample(freq)
    out = pd.DataFrame({"value": r.sum(min_count=1), "trading_days": r.count()})
    norm = 6 if freq.startswith("W") else out["trading_days"].median()
    out["coverage"] = out["trading_days"] / norm
    out["per_day"] = out["value"] / out["trading_days"].replace(0, np.nan)
    # первый и последний период обрезаны границами датасета
    calendar_days = r.size() if hasattr(r, "size") else daily.resample(freq).apply(len)
    full_len = 7 if freq.startswith("W") else out.index.days_in_month
    out["complete"] = (calendar_days.to_numpy() == np.asarray(full_len))
    out.loc[out["trading_days"] == 0, "complete"] = False
    return out


# ---------------------------------------------------------------------------
# 3. Пропуски и выбросы
# ---------------------------------------------------------------------------

def missing_runs(s: pd.Series) -> pd.DataFrame:
    """Непрерывные отрезки пропусков: начало, конец, длина."""
    isna = s.isna().to_numpy()
    if not isna.any():
        return pd.DataFrame(columns=["начало", "конец", "длина"])
    edges = np.diff(np.r_[0, isna.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return pd.DataFrame({"начало": s.index[starts], "конец": s.index[ends], "длина": ends - starts + 1})


def missing_report(s: pd.Series) -> pd.Series:
    """Количество и доля пропусков, а также нулей среди наблюдённых значений."""
    obs = s.dropna()
    return pd.Series({
        "длина ряда": len(s),
        "пропусков": int(s.isna().sum()),
        "доля пропусков": s.isna().mean(),
        "нулей среди наблюдений": int((obs == 0).sum()),
        "доля нулей": (obs == 0).mean(),
    })


def hampel_outliers(s: pd.Series, window: int = 7, n_sigmas: float = 3.0) -> pd.Series:
    """Фильтр Хампеля: точка — выброс, если она дальше n_sigmas * 1,4826 * MAD
    от скользящей медианы окна 2*window+1. Устойчив к самим выбросам и к тренду."""
    med = s.rolling(2 * window + 1, center=True, min_periods=window).median()
    mad = (s - med).abs().rolling(2 * window + 1, center=True, min_periods=window).median()
    scale = 1.4826 * mad
    return ((s - med).abs() > n_sigmas * scale) & scale.gt(0)


def iqr_outliers(s: pd.Series, k: float = 1.5, log: bool = False) -> pd.Series:
    """Правило Тьюки (Q1 − k·IQR, Q3 + k·IQR) по всему ряду сразу: не учитывает
    ни тренд, ни день недели, поэтому служит только грубой отправной точкой."""
    x = np.log1p(s) if log else s
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def stl_residual_outliers(s: pd.Series, period: int, k: float = 3.0) -> pd.Series:
    """Выбросы по остатку робастной STL: |R| > k * 1,4826 * MAD(R)."""
    res = STL(s, period=period, robust=True).fit()
    r = res.resid
    mad = 1.4826 * (r - r.median()).abs().median()
    return (r - r.median()).abs() > k * mad


def outlier_table(s: pd.Series, masks: dict[str, pd.Series]) -> pd.DataFrame:
    """Сводит разметку нескольких методов в одну таблицу (только отмеченные точки)."""
    df = pd.DataFrame(masks).reindex(s.index).fillna(False).astype(bool)
    df.insert(0, "значение", s)
    df["методов"] = df[list(masks)].sum(axis=1)
    return df[df["методов"] > 0].sort_values("значение", ascending=False)


def transactions_on(clean: pd.DataFrame, stock_code: str, dates) -> pd.DataFrame:
    """Строки счетов товара за указанные дни: помогает понять природу выброса."""
    dates = pd.DatetimeIndex(dates)
    cols = ["Date", "InvoiceNo", "CustomerID", "Country", "Quantity", "UnitPrice"]
    t = clean.loc[(clean["StockCode"] == stock_code) & clean["Date"].isin(dates), cols]
    return t.sort_values(["Date", "Quantity"], ascending=[True, False])


def order_profile(clean: pd.DataFrame, stock_code: str, big: int = 200, freq: str = "MS") -> pd.DataFrame:
    """Структура продаж по периодам: строки заказов, клиенты, доля штук в крупных
    (оптовых) строках с Quantity >= big, доля трёх крупнейших клиентов и медианная цена."""
    t = clean.loc[clean["StockCode"] == stock_code].set_index("Date").sort_index()

    def top3_share(g):
        by_c = g.groupby("CustomerID")["Quantity"].sum().nlargest(3).sum()
        return by_c / g["Quantity"].sum()

    r = t.groupby(pd.Grouper(freq=freq))
    return pd.DataFrame({
        "строк": r.size(),
        "клиентов": r["CustomerID"].nunique(),
        "штук": r["Quantity"].sum(),
        "доля штук в строках ≥ %d" % big: r.apply(lambda g: g.loc[g["Quantity"] >= big, "Quantity"].sum() / g["Quantity"].sum()),
        "доля топ-3 клиентов": r.apply(top3_share),
        "медианная цена": r["UnitPrice"].median(),
    })


def split_bulk(clean: pd.DataFrame, stock_code: str, calendar: pd.DatetimeIndex, big: int = 200) -> pd.DataFrame:
    """Дневные продажи, разделённые на обычные строки и крупные (Quantity >= big)."""
    t = clean.loc[clean["StockCode"] == stock_code]
    full = pd.date_range(calendar.min(), calendar.max(), name="Date")
    out = pd.DataFrame({
        "обычные": t.loc[t["Quantity"] < big].groupby("Date")["Quantity"].sum(),
        "крупные": t.loc[t["Quantity"] >= big].groupby("Date")["Quantity"].sum(),
    }).reindex(full)
    out.loc[out.index.isin(calendar)] = out.loc[out.index.isin(calendar)].fillna(0)
    return out


def cap_outliers(s: pd.Series, mask: pd.Series, window: int = 7) -> pd.Series:
    """Заменяет отмеченные точки скользящей медианой (используется только для
    проверки чувствительности выводов; основной анализ идёт по исходному ряду)."""
    med = s.where(~mask).rolling(2 * window + 1, center=True, min_periods=1).median()
    return s.where(~mask, med)


# ---------------------------------------------------------------------------
# 4. Описательный анализ
# ---------------------------------------------------------------------------

def describe_series(s: pd.Series) -> pd.Series:
    """Описательные статистики ряда продаж и характеристики прерывистости спроса.

    ADI — средний интервал между ненулевыми продажами (в периодах), CV² — квадрат
    коэффициента вариации ненулевых значений. По Syntetos–Boylan (2005):
    ADI < 1,32 и CV² < 0,49 — ровный спрос, ADI < 1,32 и CV² >= 0,49 — неустойчивый,
    ADI >= 1,32 и CV² < 0,49 — прерывистый, иначе — комковатый (lumpy).
    """
    x = s.dropna()
    nz = x[x > 0]
    adi = len(x) / len(nz) if len(nz) else np.inf
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
    if adi < 1.32:
        kind = "ровный (smooth)" if cv2 < 0.49 else "неустойчивый (erratic)"
    else:
        kind = "прерывистый (intermittent)" if cv2 < 0.49 else "комковатый (lumpy)"
    return pd.Series({
        "наблюдений": len(x), "среднее": x.mean(), "медиана": x.median(), "ст. откл.": x.std(),
        "минимум": x.min(), "Q1": x.quantile(0.25), "Q3": x.quantile(0.75), "максимум": x.max(),
        "коэф. вариации": x.std() / x.mean(), "асимметрия": stats.skew(x), "эксцесс": stats.kurtosis(x),
        "доля нулей": (x == 0).mean(), "ADI": adi, "CV² ненулевых": cv2, "тип спроса": kind,
    })


def describe_many(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Таблица describe_series для нескольких рядов (по столбцу на ряд)."""
    return pd.DataFrame({name: describe_series(s) for name, s in series.items()})


def rolling_stats(s: pd.Series, window: int) -> pd.DataFrame:
    """Скользящие среднее, медиана, стандартное отклонение и межквартильный размах.
    IQR устойчив к единичным пикам, в отличие от стандартного отклонения."""
    r = s.rolling(window, min_periods=max(2, window // 2))
    return pd.DataFrame({"value": s, "mean": r.mean(), "median": r.median(), "std": r.std(),
                         "iqr": r.quantile(0.75) - r.quantile(0.25)})


def compare_halves(s: pd.Series) -> pd.DataFrame:
    """Сравнение уровня и разброса первой и второй половины ряда:
    тест Манна–Уитни (уровень) и тест Брауна–Форсайта (разброс, Levene с медианой)."""
    x = s.dropna()
    a, b = x.iloc[: len(x) // 2], x.iloc[len(x) // 2:]
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    lv = stats.levene(a, b, center="median")
    return pd.DataFrame({
        "1-я половина": [a.index.min().date(), a.median(), a.std()],
        "2-я половина": [b.index.min().date(), b.median(), b.std()],
        "p-value": [np.nan, mw.pvalue, lv.pvalue],
    }, index=["начало", "медиана", "ст. откл."])


def weekday_profile(daily: pd.Series) -> pd.DataFrame:
    """Статистики продаж по дням недели (только торговые дни)."""
    x = daily.dropna()
    g = x.groupby(x.index.day_of_week)
    out = pd.DataFrame({"дней": g.size(), "среднее": g.mean(), "медиана": g.median(), "доля нулей": g.apply(lambda v: (v == 0).mean())})
    out.index = [WEEKDAYS_RU[i] for i in out.index]
    return out


def weekday_test(daily: pd.Series) -> dict:
    """Критерий Краскела–Уоллиса: одинаково ли распределены продажи по дням недели."""
    x = daily.dropna()
    groups = [v.to_numpy() for _, v in x.groupby(x.index.day_of_week)]
    h = stats.kruskal(*groups)
    return {"H": h.statistic, "p-value": h.pvalue, "групп": len(groups)}


# ---------------------------------------------------------------------------
# 5. Декомпозиция и сезонность
# ---------------------------------------------------------------------------

def trading_days_only(daily: pd.Series) -> pd.Series:
    """Только фактические наблюдения: торговые дни без суббот и праздников."""
    return daily.dropna()


def weekly_grid(daily: pd.Series, start=None, end=None, closed_weekday: int = 5) -> pd.Series:
    """Регулярная сетка «6 дней в неделю» (все дни, кроме субботы).

    Если просто выкинуть нерабочие дни, после каждого праздника фаза недельного
    цикла сдвигается, и STL / сезонная разность начинают сравнивать разные дни недели.
    На сетке позиция дня в неделе постоянна; праздничные закрытия остаются NaN."""
    s = daily.loc[start:end]
    return s[s.index.day_of_week != closed_weekday]


def fill_for_decomposition(grid: pd.Series, period: int = 6) -> tuple[pd.Series, pd.Series]:
    """Заполняет редкие NaN на сетке средним того же дня недели соседних недель
    (±1 цикл; если их нет — ±2). Нужно только для методов, не допускающих пропусков
    (STL, ACF, разности). Возвращает заполненный ряд и маску заполненных точек."""
    filled = grid.copy()
    mask = grid.isna()
    for k in (1, 2):
        neigh = pd.concat([grid.shift(k * period), grid.shift(-k * period)], axis=1).mean(axis=1)
        filled = filled.fillna(neigh)
    filled = filled.interpolate(limit_direction="both")
    return filled, mask


def stl_decompose(s: pd.Series, period: int, robust: bool = True, **kw):
    """Робастная STL-декомпозиция (Cleveland et al., 1990). Ряд не должен иметь NaN."""
    if s.isna().any():
        raise ValueError("STL не работает с пропусками: передайте ряд без NaN")
    return STL(s.to_numpy(), period=period, robust=robust, **kw).fit()


def trend_lowess(s: pd.Series, frac: float = 0.3) -> pd.DataFrame:
    """Разложение «тренд + остаток» сглаживанием LOWESS. Для недельного и месячного
    рядов длиной около года годовую сезонность отделить нельзя (нужно хотя бы
    два полных цикла), поэтому выделяем только тренд."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    x = s.dropna()
    tr = lowess(x.to_numpy(), np.arange(len(x)), frac=frac, return_sorted=False)
    return pd.DataFrame({"observed": x, "trend": tr, "resid": x.to_numpy() - tr}, index=x.index)


def linear_detrend(s: pd.Series) -> pd.Series:
    """Остаток после вычитания линейного тренда, оценённого МНК."""
    t = np.arange(len(s))
    beta = np.polyfit(t, s.to_numpy(), 1)
    return pd.Series(s.to_numpy() - np.polyval(beta, t), index=s.index, name=s.name)


def component_strength(res) -> dict:
    """Сила тренда и сезонности по Wang, Smith & Hyndman (2006):
    F = max(0, 1 - Var(R) / Var(X + R)), где X — тренд или сезонная компонента."""
    r, t, s = np.asarray(res.resid), np.asarray(res.trend), np.asarray(res.seasonal)
    return {
        "сила тренда": max(0.0, 1 - r.var() / (t + r).var()),
        "сила сезонности": max(0.0, 1 - r.var() / (s + r).var()),
    }


def seasonal_strength_null(s: pd.Series, period: int, n_perm: int = 200, seed: int = 0, **stl_kw) -> dict:
    """Перестановочная проверка силы сезонности.

    Наблюдаемая F_s сравнивается с распределением F_s на рядах, где внутри каждого
    полного цикла значения случайно перемешаны: тренд (уровень цикла) при этом
    сохраняется, а привязка к фазе цикла разрушается. p-value — доля перестановок
    с F_s не меньше наблюдаемой.
    """
    rng = np.random.default_rng(seed)
    x = s.to_numpy()
    n = len(x) // period * period
    obs = component_strength(stl_decompose(pd.Series(x[:n]), period, **stl_kw))["сила сезонности"]
    null = np.empty(n_perm)
    blocks = x[:n].reshape(-1, period)
    for i in range(n_perm):
        perm = rng.permuted(blocks, axis=1).ravel()
        null[i] = component_strength(stl_decompose(pd.Series(perm), period, **stl_kw))["сила сезонности"]
    return {"F_s": obs, "F_s при H0 (95%)": np.quantile(null, 0.95), "p-value": (1 + (null >= obs).sum()) / (1 + n_perm)}


def acf_at(s: pd.Series, lags) -> pd.Series:
    """Выборочная автокорреляция на заданных лагах и граница ±1,96/√n."""
    from statsmodels.tsa.stattools import acf
    x = s.dropna().to_numpy()
    vals = acf(x, nlags=max(lags), fft=True)
    out = pd.Series({f"lag {k}": vals[k] for k in lags})
    out["граница 95%"] = 1.96 / np.sqrt(len(x))
    return out


# ---------------------------------------------------------------------------
# 6. Стационарность
# ---------------------------------------------------------------------------

def adf_test(s: pd.Series, regression: str = "c", autolag: str | None = "AIC", maxlag: int | None = None) -> dict:
    """Тест Дики–Фуллера (расширенный). H0: есть единичный корень (ряд нестационарен)."""
    stat, p, lags, nobs, crit, *_ = adfuller(s.dropna(), maxlag=maxlag, regression=regression,
                                             autolag=autolag, result_object=False)
    return {"тест": "ADF", "H0": "единичный корень", "статистика": stat, "p-value": p, "лагов": lags,
            "n": nobs, "крит. 5%": crit["5%"]}


def kpss_test(s: pd.Series, regression: str = "c") -> dict:
    """Тест KPSS. H0: ряд стационарен (вокруг константы или тренда при regression='ct').

    p-value statsmodels берёт из таблицы и обрезает его до [0,01; 0,1]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # предупреждение об обрезке p-value
        stat, p, lags, crit = kpss(s.dropna(), regression=regression, nlags="auto")
    return {"тест": "KPSS", "H0": "стационарность", "статистика": stat, "p-value": p, "лагов": lags,
            "n": int(s.notna().sum()), "крит. 5%": crit["5%"]}


def stationarity_verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    """Совместная интерпретация ADF и KPSS."""
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        return "стационарен"
    if not adf_rej and kpss_rej:
        return "нестационарен (единичный корень)"
    if adf_rej and kpss_rej:
        return "противоречие: нет единичного корня, но есть сдвиг уровня/тренд или неоднородная дисперсия"
    return "тесты не различают гипотезы (мало данных)"


def stationarity_report(series: dict[str, pd.Series], regression: str = "c", alpha: float = 0.05,
                        maxlag: int | None = None) -> pd.DataFrame:
    """ADF + KPSS для набора рядов и общий вывод по каждому.

    Для коротких рядов (недели, месяцы) можно ограничить maxlag: иначе AIC-подбор
    лагов съедает большую часть и так небольшой выборки."""
    rows = []
    for name, s in series.items():
        a, k = adf_test(s, regression=regression, maxlag=maxlag), kpss_test(s, regression=regression)
        rows.append({
            "ряд": name, "n": a["n"] + a["лагов"] + 1,
            "ADF стат.": a["статистика"], "ADF p": a["p-value"], "ADF лагов": a["лагов"],
            "KPSS стат.": k["статистика"], "KPSS p": k["p-value"],
            "вывод": stationarity_verdict(a["p-value"], k["p-value"], alpha),
        })
    return pd.DataFrame(rows).set_index("ряд")


def difference(s: pd.Series, lag: int = 1) -> pd.Series:
    """Разность с заданным лагом (lag=6 на торговых днях — сезонная разность)."""
    return s.diff(lag).dropna()


def boxcox_lambda(s: pd.Series) -> float:
    """Оценка λ преобразования Бокса–Кокса для x + 1 методом максимального правдоподобия:
    λ ≈ 1 — преобразование не нужно, λ ≈ 0,5 — корень, λ ≈ 0 — логарифм."""
    return float(stats.boxcox(s.dropna().to_numpy() + 1)[1])


def robustness_table(variants: dict[str, pd.Series], period: int, alpha: float = 0.05, **stl_kw) -> pd.DataFrame:
    """Ключевые выводы шагов 4–5 для нескольких вариантов ряда (в log1p):
    p-value ADF и KPSS (константа и линейный тренд), KPSS после разности,
    автокорреляция разности на лаге 1 и сила недельной сезонности."""
    from statsmodels.tsa.stattools import acf
    rows = {}
    for name, s in variants.items():
        x = np.log1p(s)
        d1 = difference(x)
        rows[name] = {
            "ADF p (c)": adf_test(x)["p-value"],
            "KPSS p (c)": kpss_test(x)["p-value"],
            "ADF p (ct)": adf_test(x, regression="ct")["p-value"],
            "KPSS p (ct)": kpss_test(x, regression="ct")["p-value"],
            "KPSS p, разность": kpss_test(d1)["p-value"],
            "ACF(1) разности": acf(d1.to_numpy(), nlags=1)[1],
            "сила сезонности": component_strength(stl_decompose(x, period, **stl_kw))["сила сезонности"],
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 7. Графики
# ---------------------------------------------------------------------------

def _finish(ax, title: str, ylabel: str, xlabel: str = "Дата", legend: bool = True):
    """Общее оформление осей: заголовок, подписи, легенда, сетка."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return ax


def _break_long_gaps(s: pd.Series, max_gap_days: int = 7) -> pd.Series:
    """Убирает NaN, но вставляет разрыв там, где между наблюдениями больше max_gap_days дней.

    Субботы и одиночные праздники линия перешагивает, а рождественские каникулы
    (23.12–04.01) показываются разрывом, а не ровной полкой."""
    obs = s.dropna()
    gaps = obs.index.to_series().diff() > pd.Timedelta(days=max_gap_days)
    if not gaps.any():
        return obs
    breaks = obs.index[gaps.to_numpy()] - pd.Timedelta(days=1)
    return pd.concat([obs, pd.Series(np.nan, index=breaks)]).sort_index()


def plot_series(s: pd.Series, title: str, ylabel: str, ax=None, mark: pd.Series | None = None,
                mark_label: str = "выброс", label: str = "продажи", closed: pd.Series | None = None):
    """Линейный график ряда; опционально отмечает выбросы и нерабочие дни."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.8))
    obs = _break_long_gaps(s)  # линия через наблюдения, нерабочие дни отмечаются отдельно
    ax.plot(obs.index, obs.to_numpy(), lw=0.9, label=label)
    if mark is not None and mark.any():
        ax.scatter(s.index[mark.to_numpy()], s[mark].to_numpy(), color="C3", s=22, zorder=3, label=mark_label)
    if closed is not None and closed.any():
        y0 = np.nanmin(s.to_numpy())
        ax.scatter(closed.index[closed.to_numpy()], np.full(int(closed.sum()), y0), marker="|", color="0.5",
                   s=40, label="магазин закрыт (NaN)")
    return _finish(ax, title, ylabel)


def plot_aggregations(aggs: dict[str, pd.DataFrame], unit: str):
    """Три уровня агрегации друг под другом; неполные периоды отмечены отдельно."""
    fig, axes = plt.subplots(len(aggs), 1, figsize=(12, 3.2 * len(aggs)))
    for ax, (name, a) in zip(np.atleast_1d(axes), aggs.items()):
        y = _break_long_gaps(a["value"]) if len(a) > 100 else a["value"]  # день: линия через торговые дни
        ax.plot(y.index, y.to_numpy(), lw=0.9 if len(a) > 100 else 1.5, marker=None if len(a) > 100 else "o",
                ms=4, label=f"сумма за {name}")
        inc = ~a["complete"] & a["value"].notna()
        if inc.any() and len(a) < 100:
            ax.scatter(a.index[inc], a.loc[inc, "value"], s=60, facecolors="none", edgecolors="C3",
                       zorder=3, label="неполный период")
        _finish(ax, f"Продажи за {name}", unit)
    fig.tight_layout()
    return fig


def plot_rolling(rs: pd.DataFrame, window: int, title: str, ylabel: str):
    """Ряд со скользящими средним и медианой, ниже — скользящие ст. откл. и IQR."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    raw = _break_long_gaps(rs["value"])
    a1.plot(raw.index, raw.to_numpy(), lw=0.6, alpha=0.5, label="ряд")
    a1.plot(rs.index, rs["mean"], lw=1.6, label=f"скользящее среднее ({window})")
    a1.plot(rs.index, rs["median"], lw=1.6, label=f"скользящая медиана ({window})")
    _finish(a1, title, ylabel, xlabel="")
    a2.plot(rs.index, rs["std"], color="C3", lw=1.4, label=f"скользящее ст. откл. ({window})")
    a2.plot(rs.index, rs["iqr"], color="C4", lw=1.4, label=f"скользящий IQR ({window})")
    _finish(a2, "Изменение разброса", ylabel)
    fig.tight_layout()
    return fig


def plot_distribution(s: pd.Series, title: str, xlabel: str):
    """Гистограммы ряда на исходной шкале и на шкале log(1 + x)."""
    x = s.dropna()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    a1.hist(x, bins=40, color="C0", alpha=0.8, label="торговые дни")
    a1.axvline(x.mean(), color="C3", ls="--", label=f"среднее {x.mean():.0f}")
    a1.axvline(x.median(), color="C2", ls="--", label=f"медиана {x.median():.0f}")
    _finish(a1, title, "Число дней", xlabel=xlabel)
    a2.hist(np.log1p(x), bins=40, color="C1", alpha=0.8, label="log(1 + x)")
    _finish(a2, title + " (лог-шкала)", "Число дней", xlabel=f"log(1 + {xlabel.lower()})")
    fig.tight_layout()
    return fig


def plot_weekday(daily: pd.Series, title: str, ylabel: str):
    """Ящики с усами log(1 + продажи) по дням недели."""
    x = daily.dropna()
    days = sorted(x.index.day_of_week.unique())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = [np.log1p(x[x.index.day_of_week == d]) for d in days]
    ax.boxplot(data, tick_labels=[WEEKDAYS_RU[d] for d in days], showmeans=True,
               meanprops=dict(marker="D", markerfacecolor="C3", markeredgecolor="C3", markersize=5))
    ax.plot([], [], "D", color="C3", label="среднее")
    ax.plot([], [], color="C1", label="медиана")
    return _finish(ax, title, ylabel, xlabel="День недели")


def plot_acf_pacf(s: pd.Series, lags: int, title: str, lag_unit: str = "торговых дней", ylim: float = 0.6):
    """ACF и PACF рядом; ось Y ограничена ±ylim, чтобы были видны небольшие корреляции."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    x = s.dropna()
    plot_acf(x, lags=lags, ax=a1, zero=False)
    plot_pacf(x, lags=lags, ax=a2, zero=False, method="ywm")
    for ax, name in ((a1, "ACF"), (a2, "PACF")):
        ax.set_ylim(-ylim, ylim)
        ax.plot([], [], color="C0", alpha=0.25, lw=8, label="95% доверительная полоса")
        _finish(ax, f"{name}: {title}", "Корреляция", xlabel=f"Лаг, {lag_unit}")
    fig.tight_layout()
    return fig


def plot_stl(res, index, title: str, ylabel: str):
    """Четыре панели STL: наблюдения, тренд, сезонность, остаток."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    parts = [("наблюдения", res.observed), ("тренд", res.trend), ("сезонность", res.seasonal), ("остаток", res.resid)]
    for ax, (name, v) in zip(axes, parts):
        if name == "остаток":
            ax.scatter(index, v, s=5, label=name)
            ax.axhline(0, color="0.3", lw=0.8)
        else:
            ax.plot(index, v, lw=0.9, label=name)
        _finish(ax, "", ylabel, xlabel="")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Дата")
    fig.tight_layout()
    return fig


def plot_seasonal_profile(res, index: pd.DatetimeIndex, title: str, ylabel: str, n_parts: int = 3):
    """Сезонная компонента STL по дням недели, отдельно для частей ряда:
    видно, насколько недельный профиль устойчив во времени."""
    seas = pd.Series(np.asarray(res.seasonal), index=index)
    order = [6, 0, 1, 2, 3, 4]  # Вс, Пн, ..., Пт — порядок торговой недели
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for part in np.array_split(np.arange(len(seas)), n_parts):
        chunk = seas.iloc[part]
        prof = chunk.groupby(chunk.index.day_of_week).mean().reindex(order)
        lbl = f"{chunk.index[0]:%d.%m} – {chunk.index[-1]:%d.%m.%Y}"
        ax.plot([WEEKDAYS_RU[d] for d in order], prof.to_numpy(), marker="o", label=lbl)
    ax.axhline(0, color="0.3", lw=0.8)
    return _finish(ax, title, ylabel, xlabel="День недели")


def plot_trend_lowess(tr: pd.DataFrame, title: str, ylabel: str):
    """Ряд с трендом LOWESS и остатком под ним."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[2, 1])
    a1.plot(tr.index, tr["observed"], marker="o", ms=3, lw=0.9, label="наблюдения")
    a1.plot(tr.index, tr["trend"], lw=2, label="тренд (LOWESS)")
    _finish(a1, title, ylabel, xlabel="")
    a2.bar(tr.index, tr["resid"], width=5, color="C2", label="остаток")
    a2.axhline(0, color="0.3", lw=0.8)
    _finish(a2, "", ylabel)
    fig.tight_layout()
    return fig


def plot_bulk(split: pd.DataFrame, freq: str, title: str, ylabel: str):
    """Продажи по периодам, разделённые на обычные и крупные строки заказов."""
    agg = split.resample(freq).sum(min_count=1)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    width = 5 if freq.startswith("W") else 20
    small, large = agg.iloc[:, 0], agg.iloc[:, 1]
    ax.bar(agg.index, small, width=width, label=agg.columns[0])
    ax.bar(agg.index, large, width=width, bottom=small, label=agg.columns[1])
    return _finish(ax, title, ylabel)


def plot_transform_comparison(series: dict[str, pd.Series], title: str):
    """Ряды до и после преобразований друг под другом, с линией среднего."""
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 2.6 * len(series)), sharex=True)
    for ax, (name, s) in zip(np.atleast_1d(axes), series.items()):
        ax.plot(s.index, s.to_numpy(), lw=0.7, label=name)
        ax.axhline(s.mean(), color="C3", lw=0.8, ls="--", label="среднее")
        _finish(ax, "", "Значение", xlabel="")
    np.atleast_1d(axes)[0].set_title(title)
    np.atleast_1d(axes)[-1].set_xlabel("Дата")
    fig.tight_layout()
    return fig


@dataclass
class SeriesBundle:
    """Ряды выбранного товара на всех уровнях агрегации."""
    stock_code: str
    description: str
    daily: pd.Series
    weekly: pd.DataFrame
    monthly: pd.DataFrame

    @classmethod
    def build(cls, clean: pd.DataFrame, stock_code: str, value: str = "Quantity") -> "SeriesBundle":
        cal = trading_calendar(clean)
        daily = build_daily_series(clean, stock_code, cal, value)
        desc = clean.loc[clean["StockCode"] == stock_code, "Description"].mode().iat[0]
        return cls(stock_code, desc, daily, aggregate_series(daily, "W-SAT"), aggregate_series(daily, "MS"))

    @property
    def trading(self) -> pd.Series:
        return trading_days_only(self.daily)
