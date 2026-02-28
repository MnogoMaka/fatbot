import csv
import os
import logging
import calendar
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from badlist import BAD_LIST
import pytz
from matplotlib.patches import Rectangle, FancyBboxPatch
from dotenv import load_dotenv

load_dotenv()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    InputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ConversationHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- Конфигурация логгирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Конфигурация доступа и путей ---
ALLOWED_USERNAMES = os.getenv("USERNAMES").split(';')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

USERS_CSV = DATA_DIR / "users.csv"
ENTRIES_CSV = DATA_DIR / "entries.csv"
CALENDAR_DIR = DATA_DIR / "calendars"
CALENDAR_DIR.mkdir(exist_ok=True)

MOSCOW_TZ = pytz.timezone("Europe/Moscow")

# Небольшой список колких фраз — можно расширять



def get_bad_phrase() -> str:
    if not BAD_LIST:
        return ""
    return random.choice(BAD_LIST)


# --- Модели данных ---

class Role(str, Enum):
    FATTY = "Жиртрест"
    PIG = "Кабан"
    ALMOST_FIT = "Почти соска"
    NORMAL = "Норм чел"


@dataclass
class UserProfile:
    user_id: int
    username: str
    current_weight: float
    target_weight: float
    calorie_limit: int
    height_cm: int = 180
    age: int = 25
    gender: str = "male"
    activity_level: float = 1.375
    start_weight: Optional[float] = None

    def __post_init__(self):
        if self.start_weight is None:
            self.start_weight = self.current_weight

    def calculate_bmr(self) -> float:
        """Формула Миффлина-Сан Жеора (сколько организм тратит в покое)"""
        if self.gender == "female":
            return 10 * self.current_weight + 6.25 * self.height_cm - 5 * self.age - 161
        return 10 * self.current_weight + 6.25 * self.height_cm - 5 * self.age + 5

    def calculate_tdee(self) -> float:
        """Общий расход энергии с учётом активности (TDEE)"""
        return self.calculate_bmr() * self.activity_level

    def get_deficit_progress(self, today_calories: int = 0) -> Dict[str, float]:
        """
        Расчёт прогресса дефицита калорий.
        today_calories — фактически потреблённые калории сегодня (из entries.csv)
        """
        kcal_per_kg = 7700
        start = self.start_weight if self.start_weight else self.current_weight

        # Всего нужно сжечь для достижения цели
        total_deficit = max(0, (start - self.target_weight)) * kcal_per_kg

        # Уже сожжено — по факту потери веса (объективный показатель)
        achieved = max(0, (start - self.current_weight)) * kcal_per_kg

        # Остаток
        remaining = max(0, total_deficit - achieved)

        # TDEE и ежедневный дефицит
        tdee = self.calculate_tdee()
        daily_deficit = max(0, tdee - today_calories)  # ✅ Используем фактическое потребление!

        # Прогноз
        days_to_goal = remaining / daily_deficit if daily_deficit > 0 else float('inf')

        return {
            'total_deficit_needed': total_deficit,
            'deficit_achieved': achieved,
            'deficit_remaining': remaining,
            'daily_deficit': daily_deficit,
            'days_to_goal': days_to_goal,
            'tdee': tdee,
            'bmr': self.calculate_bmr(),
            'today_calories': today_calories,
        }

    @property
    def progress_percent(self) -> float:
        start = self.start_weight if self.start_weight else max(self.current_weight, self.target_weight * 1.5)
        if start <= self.target_weight:
            return 100.0
        progress = (start - self.current_weight) / (start - self.target_weight)
        return max(0.0, min(100.0, progress * 100))

    @property
    def role(self) -> Role:
        p = self.progress_percent
        if p < 25:
            return Role.FATTY
        if p < 50:
            return Role.PIG
        if p < 80:
            return Role.ALMOST_FIT
        return Role.NORMAL


@dataclass
class DailyEntry:
    date: date
    user_id: int
    username: str
    calories: int
    weight: Optional[float] = None
    exercises: str = ""


# --- Работа с CSV ---

def ensure_csv_files() -> None:
    if not USERS_CSV.exists():
        logger.info("Создание файла users.csv")
        with USERS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "user_id", "username", "current_weight", "target_weight", "calorie_limit",
                "height_cm", "age", "gender", "activity_level", "start_weight"
            ])

    if not ENTRIES_CSV.exists():
        logger.info("Создание файла entries.csv")
        with ENTRIES_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "user_id", "username", "calories", "weight", "exercises"])


def load_users() -> Dict[int, UserProfile]:
    ensure_csv_files()
    users: Dict[int, UserProfile] = {}
    try:
        with USERS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    user_id = int(row["user_id"])
                    users[user_id] = UserProfile(
                        user_id=user_id,
                        username=row["username"],
                        current_weight=float(row["current_weight"]),
                        target_weight=float(row["target_weight"]),
                        calorie_limit=int(row["calorie_limit"]),
                        height_cm=int(row.get("height_cm") or 175),
                        age=int(row.get("age") or 30),
                        gender=row.get("gender") or "male",
                        activity_level=float(row.get("activity_level") or 1.375),
                        start_weight=float(row["start_weight"]) if row.get("start_weight") else None,
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Ошибка парсинга строки пользователя: {row}, ошибка: {e}")
                    continue
        logger.info(f"Загружено пользователей: {len(users)}")
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке users.csv: {e}")
    return users


def save_users(users: Dict[int, UserProfile]) -> None:
    ensure_csv_files()
    try:
        with USERS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "user_id", "username", "current_weight", "target_weight", "calorie_limit",
                "height_cm", "age", "gender", "activity_level", "start_weight"
            ])
            for u in users.values():
                writer.writerow([
                    u.user_id,
                    u.username,
                    f"{u.current_weight:.2f}",
                    f"{u.target_weight:.2f}",
                    u.calorie_limit,
                    u.height_cm,
                    u.age,
                    u.gender,
                    f"{u.activity_level:.3f}",
                    f"{u.start_weight:.2f}" if u.start_weight else "",
                ])
        logger.info(f"Сохранено пользователей: {len(users)}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении users.csv: {e}")


def append_or_update_entry(entry: DailyEntry) -> None:
    ensure_csv_files()
    rows: List[Dict[str, str]] = []
    found = False
    try:
        with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["user_id"] == str(entry.user_id) and row["date"] == entry.date.isoformat():
                    old_cal = int(row.get("calories") or 0)
                    row["calories"] = str(old_cal + entry.calories)
                    if entry.weight is not None:
                        row["weight"] = f"{entry.weight:.2f}"
                    if entry.exercises:
                        row["exercises"] = entry.exercises
                    found = True
                rows.append(row)

        if not found:
            rows.append({
                "date": entry.date.isoformat(),
                "user_id": str(entry.user_id),
                "username": entry.username,
                "calories": str(entry.calories),
                "weight": f"{entry.weight:.2f}" if entry.weight is not None else "",
                "exercises": entry.exercises,
            })

        with ENTRIES_CSV.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["date", "user_id", "username", "calories", "weight", "exercises"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Запись обновлена/добавлена для user_id {entry.user_id} за {entry.date}")
    except Exception as e:
        logger.error(f"Ошибка при работе с entries.csv: {e}")


def load_entries_for_month(year: int, month: int, user_id: Optional[int] = None) -> List[DailyEntry]:
    ensure_csv_files()
    result: List[DailyEntry] = []
    try:
        with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    d = date.fromisoformat(row["date"])
                except (TypeError, ValueError):
                    continue
                if d.year != year or d.month != month:
                    continue
                if user_id is not None and int(row["user_id"]) != user_id:
                    continue
                try:
                    calories = int(row["calories"])
                except (TypeError, ValueError):
                    calories = 0
                weight = None
                if row.get("weight"):
                    try:
                        weight = float(row["weight"])
                    except (TypeError, ValueError):
                        weight = None
                result.append(DailyEntry(
                    date=d,
                    user_id=int(row["user_id"]),
                    username=row["username"],
                    calories=calories,
                    weight=weight,
                    exercises=row.get("exercises", ""),
                ))
    except Exception as e:
        logger.error(f"Ошибка загрузки записей за {year}-{month}: {e}")
    return result


def get_available_months(user_id: Optional[int] = None) -> List[Tuple[int, int]]:
    """Возвращает список уникальных (year, month) из entries.csv."""
    ensure_csv_files()
    months_set: Set[Tuple[int, int]] = set()
    try:
        with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if user_id is not None and int(row["user_id"]) != user_id:
                    continue
                try:
                    d = date.fromisoformat(row["date"])
                    months_set.add((d.year, d.month))
                except (TypeError, ValueError):
                    continue
    except Exception as e:
        logger.error(f"Ошибка при получении доступных месяцев: {e}")

    return sorted(months_set, key=lambda x: (x[0], x[1]), reverse=True)


def load_entries_for_user(user_id: int) -> List[DailyEntry]:
    """Загружает все записи для конкретного пользователя."""
    ensure_csv_files()
    result: List[DailyEntry] = []
    try:
        with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(row["user_id"]) != user_id:
                        continue
                    d = date.fromisoformat(row["date"])
                except (TypeError, ValueError, KeyError):
                    continue
                try:
                    calories = int(row["calories"])
                except (TypeError, ValueError):
                    calories = 0
                weight = None
                if row.get("weight"):
                    try:
                        weight = float(row["weight"])
                    except (TypeError, ValueError):
                        weight = None
                result.append(
                    DailyEntry(
                        date=d,
                        user_id=user_id,
                        username=row.get("username", ""),
                        calories=calories,
                        weight=weight,
                        exercises=row.get("exercises", ""),
                    )
                )
    except Exception as e:
        logger.error(f"Ошибка загрузки записей пользователя {user_id}: {e}")
    return result


def compute_deficit_with_history(
    profile: UserProfile,
    entries: List[DailyEntry],
) -> Dict[str, float]:
    """
    Расчёт дефицита:
    - по весу (объективно),
    - по калориям (накопленный дефицит),
    - прогноз по среднему дефициту за 7 дней.
    """
    kcal_per_kg = 7700
    start = profile.start_weight if profile.start_weight else profile.current_weight

    total_deficit_needed = max(0.0, (start - profile.target_weight)) * kcal_per_kg
    deficit_achieved_weight = max(0.0, (start - profile.current_weight)) * kcal_per_kg

    # Калории по дням
    daily_cals: Dict[date, int] = defaultdict(int)
    for e in entries:
        if e.user_id != profile.user_id:
            continue
        daily_cals[e.date] += e.calories

    tdee = profile.calculate_tdee()
    bmr = profile.calculate_bmr()

    today = date.today()
    today_calories = daily_cals.get(today, 0)
    daily_deficit_today = max(0.0, tdee - today_calories)

    # Накопленный дефицит по калориям за всё время
    deficit_achieved_calories = 0.0
    for d, cals in daily_cals.items():
        day_def = max(0.0, tdee - cals)
        deficit_achieved_calories += day_def

    # Эффективно засчитываем максимум из «по весу» и «по калориям»
    deficit_achieved_effective = max(deficit_achieved_weight, deficit_achieved_calories)
    deficit_remaining = max(0.0, total_deficit_needed - deficit_achieved_effective)

    # Средний дефицит за последние 7 дней (включая сегодня),
    # но считаем только по дням, где есть записи по калориям.
    total_def_7 = 0.0
    days_counted = 0
    for i in range(7):
        d = today - timedelta(days=i)
        if d in daily_cals:
            cals = daily_cals[d]
            day_def = max(0.0, tdee - cals)
            total_def_7 += day_def
            days_counted += 1

    if days_counted:
        avg_daily_def_7d = total_def_7 / days_counted
    else:
        # Если записей вообще нет, ориентируемся на плановый дефицит (TDEE - лимит),
        # а не на нереалистичный вариант «сегодня ничего не ел».
        planned_deficit = max(0.0, tdee - profile.calorie_limit)
        avg_daily_def_7d = planned_deficit

    days_to_goal = deficit_remaining / avg_daily_def_7d if avg_daily_def_7d > 0 else float("inf")

    return {
        "total_deficit_needed": total_deficit_needed,
        "deficit_achieved_weight": deficit_achieved_weight,
        "deficit_achieved_calories": deficit_achieved_calories,
        "deficit_achieved_effective": deficit_achieved_effective,
        "deficit_remaining": deficit_remaining,
        "daily_deficit_today": daily_deficit_today,
        "avg_daily_deficit_7d": avg_daily_def_7d,
        "days_to_goal": days_to_goal,
        "tdee": tdee,
        "bmr": bmr,
        "today_calories": today_calories,
    }


# --- Построение календаря ---

def build_calendar_image(
        *,
        year: int,
        month: int,
        users: Dict[int, UserProfile],
        entries: List[DailyEntry],
        personal_user_id: Optional[int] = None,
) -> Path:
    cal = calendar.Calendar(firstweekday=0)
    month_days = [d for d in cal.itermonthdates(year, month) if d.month == month]

    daily_user_calories: Dict[date, Dict[int, int]] = {d: {} for d in month_days}

    for e in entries:
        if e.date in daily_user_calories and e.calories > 0:
            if e.user_id not in daily_user_calories[e.date]:
                daily_user_calories[e.date][e.user_id] = 0
            daily_user_calories[e.date][e.user_id] += e.calories

    weeks = calendar.monthcalendar(year, month)
    n_weeks = len(weeks)
    fig, ax = plt.subplots(figsize=(14, 2.2 + 1.6 * n_weeks))

    for week_idx, week in enumerate(weeks):
        for dow_idx, day_num in enumerate(week):
            if day_num == 0:
                continue
            d = date(year, month, day_num)

            has_any_data = any(daily_user_calories[d].values())
            cell_bg = "#FFFFFF" if has_any_data else "#FAFAFA"
            rect = plt.Rectangle(
                (dow_idx, n_weeks - week_idx - 1),
                1, 1,
                facecolor=cell_bg,
                edgecolor="#E0E0E0",
                linewidth=0.5
            )
            ax.add_patch(rect)

            ax.text(
                dow_idx + 0.02,
                n_weeks - week_idx - 0.15,
                str(day_num),
                ha="left", va="top",
                fontsize=10, color="#666666", weight="bold"
            )

            user_cals = daily_user_calories[d]

            if personal_user_id is not None:
                total = user_cals.get(personal_user_id, 0)
                user = users.get(personal_user_id)
                limit = user.calorie_limit if user else 0

                if total > 0:
                    if total <= limit * 0.8:
                        bg_color = "#C8E6C9"
                        text_color = "#2E7D32"
                    elif total <= limit:
                        bg_color = "#FFF9C4"
                        text_color = "#F57F17"
                    else:
                        bg_color = "#FFCDD2"
                        text_color = "#C62828"

                    box_w, box_h = 0.9, 0.5
                    # ✅ Исправлено: используем FancyBboxPatch вместо Rectangle
                    box = FancyBboxPatch(
                        (dow_idx + 0.05, n_weeks - week_idx - 0.9),
                        box_w, box_h,
                        facecolor=bg_color, edgecolor="none",
                        boxstyle="round,pad=0.1",
                        mutation_aspect=0.5
                    )
                    ax.add_patch(box)
                    ax.text(
                        dow_idx + 0.5,
                        n_weeks - week_idx - 0.65,
                        f"{total}",
                        ha="center", va="center",
                        fontsize=11, color=text_color, weight="bold"
                    )
            else:
                if user_cals:
                    sorted_users = sorted(
                        user_cals.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    max_display = 3
                    to_show = sorted_users[:max_display]
                    hidden_count = len(sorted_users) - max_display

                    y_start = n_weeks - week_idx - 0.35
                    line_height = 0.22

                    for i, (uid, cals) in enumerate(to_show):
                        user = users.get(uid)
                        username = user.username if user and user.username else f"user_{uid}"
                        limit = user.calorie_limit if user else 2000

                        if cals <= limit * 0.8:
                            status_color = "#4CAF50"
                        elif cals <= limit:
                            status_color = "#FFC107"
                        else:
                            status_color = "#F44336"

                        display_text = f"@{username[:8]}: {cals}"

                        ax.text(
                            dow_idx + 0.03,
                            y_start - i * line_height,
                            display_text,
                            ha="left", va="center",
                            fontsize=8,
                            color=status_color if cals > limit else "#333333",
                            weight="bold" if cals > limit else "normal",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFFF80", edgecolor="none")
                        )

                    if hidden_count > 0:
                        ax.text(
                            dow_idx + 0.03,
                            y_start - len(to_show) * line_height,
                            f"…ещё {hidden_count}",
                            ha="left", va="center",
                            fontsize=7, color="#999999",
                            style="italic"
                        )

    ax.set_xlim(0, 7)
    ax.set_ylim(0, n_weeks)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"], fontsize=9)
    ax.set_yticks([])

    if personal_user_id is None:
        legend_text = "🟢 в норме  🟡 на грани  🔴 перебор"
        ax.text(
            3.5, -0.4, legend_text,
            ha="center", va="center",
            fontsize=8, color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#DDD")
        )

    month_name = calendar.month_name[month].capitalize()
    title = f"{'Личная' if personal_user_id else 'Общая'} статистика — {month_name} {year}"
    ax.set_title(title, fontsize=13, pad=20)
    ax.axis("off")
    fig.tight_layout()

    filename = (
        f"personal_{personal_user_id}_{year}_{month}.png"
        if personal_user_id
        else f"global_{year}_{month}.png"
    )
    path = CALENDAR_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# --- Telegram-бот ---

(
    ONBOARD_WEIGHT, ONBOARD_TARGET, ONBOARD_LIMIT,
    ONBOARD_HEIGHT, ONBOARD_AGE, ONBOARD_GENDER, ONBOARD_ACTIVITY,
    ADD_CALORIES, UPDATE_WEIGHT,
    STATS_SCOPE, STATS_MONTH_SELECT,
    SETTINGS_CHOICE, SETTINGS_NEW_TARGET, SETTINGS_NEW_LIMIT,
    SETTINGS_EDIT_BIOMETRICS, SETTINGS_EDIT_ACTIVITY,
) = range(16)

MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["🍔 Добавить калории", "⚖️ Обновить вес"],
        ["📊 Мой статус", "📅 Статистика"],
        ["⚙️ Настройки", "⚡ Получить заряд бодрости"],
    ],
    resize_keyboard=True,
)

STATS_SCOPE_KEYBOARD = ReplyKeyboardMarkup(
    [["👤 Моя статистика"], ["🌍 Общая статистика"], ["❌ Отмена"]],
    resize_keyboard=True,
)

SETTINGS_KEYBOARD = ReplyKeyboardMarkup(
    [["🎯 Изменить цель (вес)"], ["🔥 Изменить лимит (ккл)"],
     ["📏 Рост/возраст/пол"], ["🏃 Активность"], ["❌ Отмена"]],
    resize_keyboard=True,
)


def is_allowed(update: Update) -> bool:
    user = update.effective_user
    if not user or not user.username:
        return False
    allowed = user.username in ALLOWED_USERNAMES
    if not allowed:
        logger.warning(f"Попытка доступа от запрещенного пользователя: {user.username} ({user.id})")
    return allowed


async def deny_access(update: Update) -> None:
    if update.message:
        await update.message.reply_text("🚫 У тебя нет доступа к этому боту.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        await deny_access(update)
        return ConversationHandler.END

    ensure_csv_files()
    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    logger.info(f"Команда /start от пользователя {tg_user.username}")

    if tg_user.id in users:
        profile = users[tg_user.id]
        all_entries = load_entries_for_user(tg_user.id)
        deficit = compute_deficit_with_history(profile, all_entries)
        await update.message.reply_text(
            f"С возвращением, {tg_user.first_name}!\n\n"
            f"⚖️ Вес: {profile.current_weight:.1f} кг (цель: {profile.target_weight:.1f})\n"
            f"🔥 Лимит: {profile.calorie_limit} ккал | TDEE: {deficit['tdee']:.0f} ккал\n"
            f"📉 Осталось сжечь: {format_ru_number(deficit['deficit_remaining'])} ккал\n"
            f"🏆 Звание: {profile.role.value}",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Привет! Это твой жирный помощник.\n"
        "Сначала настроим профиль.\n\n"
        "Введи текущий вес в кг (например: 83.5):",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ONBOARD_WEIGHT


# --- ОНБОРДИНГ (ИСПРАВЛЕННЫЙ) ---

async def onboard_weight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        weight = float(update.message.text.replace(",", "."))
        if weight <= 0:
            raise ValueError
    except (TypeError, ValueError):
        await update.message.reply_text("Не понял. Введи положительное число, например: 83.5")
        return ONBOARD_WEIGHT
    context.user_data["current_weight"] = weight
    context.user_data["start_weight"] = weight
    await update.message.reply_text("Ок. Теперь введи целевой вес в кг:")
    return ONBOARD_TARGET


async def onboard_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        weight = float(update.message.text.replace(",", "."))
        if weight <= 0:
            raise ValueError
    except (TypeError, ValueError):
        await update.message.reply_text("Не понял. Введи положительное число, например: 75")
        return ONBOARD_TARGET
    context.user_data["target_weight"] = weight
    await update.message.reply_text("Отлично. Теперь введи дневной лимит калорий (целое число):")
    return ONBOARD_LIMIT


async def onboard_limit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        limit = int(update.message.text)
        if limit <= 0:
            raise ValueError
    except (TypeError, ValueError):
        await update.message.reply_text("Не понял. Введи положительное целое число, например: 2200")
        return ONBOARD_LIMIT
    context.user_data["calorie_limit"] = limit
    await update.message.reply_text(
        "Теперь для расчёта метаболизма.\n"
        "Введи свой рост в см (например: 180):",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ONBOARD_HEIGHT


async def onboard_height(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        height = int(update.message.text)
        if not 100 <= height <= 250:
            raise ValueError
        context.user_data["height_cm"] = height
    except (TypeError, ValueError):
        await update.message.reply_text("Рост должен быть числом от 100 до 250 см. Попробуй ещё раз:")
        return ONBOARD_HEIGHT

    # Сразу переходим к следующему шагу
    await update.message.reply_text("Введи возраст в годах (например: 28):")
    return ONBOARD_AGE


async def onboard_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        age = int(update.message.text)
        if not 10 <= age <= 100:
            raise ValueError
        context.user_data["age"] = age
    except (TypeError, ValueError):
        await update.message.reply_text("Возраст должен быть от 10 до 100 лет. Попробуй ещё раз:")
        return ONBOARD_AGE

    keyboard = ReplyKeyboardMarkup([["Мужской", "Женский"]], resize_keyboard=True)
    await update.message.reply_text("Выбери пол:", reply_markup=keyboard)
    return ONBOARD_GENDER


async def onboard_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    text = update.message.text.strip().lower()
    if text in ["мужской", "м", "male"]:
        context.user_data["gender"] = "male"
    elif text in ["женский", "ж", "female"]:
        context.user_data["gender"] = "female"
    else:
        await update.message.reply_text("Выбери 'Мужской' или 'Женский' с клавиатуры.")
        return ONBOARD_GENDER

    keyboard = ReplyKeyboardMarkup([
        ["🪑 1.2", "🚶 1.375"],
        ["🏃 1.55", "🔥 1.725"],
        ["/skip"]
    ], resize_keyboard=True)
    await update.message.reply_text(
        "Выбери уровень активности:\n"
        "🪑 1.2 — сидячий (офис, без спорта)\n"
        "🚶 1.375 — лёгкая (тренировки 1-3 раза/нед)\n"
        "🏃 1.55 — средняя (3-5 раз/нед)\n"
        "🔥 1.725 — активная (ежедневно)\n"
        "Или /skip:",
        reply_markup=keyboard
    )
    return ONBOARD_ACTIVITY

def format_ru_number(num: float) -> str:
    """Форматирует число с пробелами как разделитель тысяч: 100100 → '100 100'"""
    return f"{int(num):,}".replace(",", " ")

async def onboard_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END

    text = update.message.text.strip()
    activity_map = {
        "🪑 1.2": 1.2, "1.2": 1.2,
        "🚶 1.375": 1.375, "1.375": 1.375,
        "🏃 1.55": 1.55, "1.55": 1.55,
        "🔥 1.725": 1.725, "1.725": 1.725,
    }

    # Проверяем корректность ввода
    new_activity = activity_map.get(text, None)
    if new_activity is None:
        await update.message.reply_text(
            "Пожалуйста, выбери один из вариантов в меню:\n"
            "🪑 1.2 — сидячий\n"
            "🚶 1.375 — лёгкая\n"
            "🏃 1.55 — средняя\n"
            "🔥 1.725 — активная"
        )
        return ONBOARD_ACTIVITY

    context.user_data["activity_level"] = new_activity

    # === ФИНАЛИЗАЦИЯ ОНБОРДИНГА (встроена напрямую) ===
    ensure_csv_files()
    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    profile = UserProfile(
        user_id=tg_user.id,
        username=tg_user.username or "",
        current_weight=context.user_data["current_weight"],
        target_weight=context.user_data["target_weight"],
        calorie_limit=context.user_data["calorie_limit"],
        height_cm=context.user_data.get("height_cm", 175),
        age=context.user_data.get("age", 30),
        gender=context.user_data.get("gender", "male"),
        activity_level=context.user_data.get("activity_level", 1.375),
        start_weight=context.user_data.get("start_weight"),
    )
    users[tg_user.id] = profile
    save_users(users)

    # На старте используем чисто расчётный дефицит (по весу ещё рано судить)
    base_deficit = profile.get_deficit_progress()
    days_forecast = (
        f"~{base_deficit['days_to_goal']:.0f} дней"
        if base_deficit["daily_deficit"] > 0
        else "❌ Нет дефицита"
    )

    logger.info(
        f"Новый пользователь: {tg_user.username}, "
        f"BMR={base_deficit['bmr']:.0f}, TDEE={base_deficit['tdee']:.0f}"
    )

    await update.message.reply_text(
        f"✅ Профиль готов!\n\n"
        f"🔥 Твой метаболизм:\n"
        f"   BMR (покой): {base_deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE (с активностью): {base_deficit['tdee']:.0f} ккал/день\n\n"
        f"🎯 Для цели нужно сжечь: {format_ru_number(base_deficit['total_deficit_needed'])} ккал\n"
        f"📊 При лимите {profile.calorie_limit} ккал/день:\n"
        f"   Ежедневный дефицит: ~{base_deficit['daily_deficit']:.0f} ккал\n"
        f"   Прогноз до цели: {days_forecast}\n\n"
        f"🏆 Звание: {profile.role.value}",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


async def _finalize_onboarding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ensure_csv_files()
    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    profile = UserProfile(
        user_id=tg_user.id,
        username=tg_user.username or "",
        current_weight=context.user_data["current_weight"],
        target_weight=context.user_data["target_weight"],
        calorie_limit=context.user_data["calorie_limit"],
        height_cm=context.user_data.get("height_cm", 175),
        age=context.user_data.get("age", 30),
        gender=context.user_data.get("gender", "male"),
        activity_level=context.user_data.get("activity_level", 1.375),
        start_weight=context.user_data.get("start_weight"),
    )
    users[tg_user.id] = profile
    save_users(users)

    deficit = profile.get_deficit_progress()
    days_forecast = f"~{deficit['days_to_goal']:.0f} дней" if deficit['daily_deficit'] > 0 else "❌ Нет дефицита"

    logger.info(f"Новый пользователь: {tg_user.username}, BMR={deficit['bmr']:.0f}, TDEE={deficit['tdee']:.0f}")

    await update.message.reply_text(
        f"✅ Профиль готов!\n\n"
        f"🔥 Твой метаболизм:\n"
        f"   BMR (покой): {deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE (с активностью): {deficit['tdee']:.0f} ккал/день\n\n"
        f"🎯 Для цели нужно сжечь: {format_ru_number(deficit['total_deficit_needed'])} ккал\n"
        f"📊 При лимите {profile.calorie_limit} ккал/день:\n"
        f"   Ежедневный дефицит: ~{format_ru_number(deficit['daily_deficit'])} ккал\n"
        f"   Прогноз до цели: {days_forecast}\n\n"
        f"🏆 Звание: {profile.role.value}",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


# --- Калории ---

async def add_calories_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    await update.message.reply_text("Сколько калорий в этом приёме пищи? (число суммируется к сегодняшнему):")
    return ADD_CALORIES


async def handle_add_calories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        calories = int(update.message.text)
        if calories < 0:
            raise ValueError
    except (TypeError, ValueError):
        await update.message.reply_text("Введи положительное целое число.")
        return ADD_CALORIES

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    if tg_user.id not in users:
        await update.message.reply_text("Сначала введи профиль командой /start.")
        return ConversationHandler.END

    entry = DailyEntry(date=date.today(), user_id=tg_user.id, username=tg_user.username or "", calories=calories)
    append_or_update_entry(entry)
    logger.info(f"User {tg_user.username} added {calories} kcal")

    profile = users[tg_user.id]
    # Пересчитываем дефицит с учётом всей истории и сегодняшних калорий
    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    insult = get_bad_phrase()
    await update.message.reply_text(
        f"Записал +{calories} ккал.\n"
        f"{insult}\n"
        f"Осталось сжечь до цели: {format_ru_number(deficit['deficit_remaining'])} ккал",
        reply_markup=MAIN_MENU_KEYBOARD,
    )

    # Если уже превысил лимит по калориям — отдельная «наградная» фраза
    today_cals = int(deficit["today_calories"])
    if today_cals > profile.calorie_limit:
        await update.message.reply_text("АХАХАХА ну ты и лох, жри дальше. Теперь все об этом знают")
        for user_id, profile in users.items():
            if user_id != tg_user.id:
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=(
                            f"Поздравьте ЖИРОБАСА {tg_user.username}. Он сегодня объелся как свинья.\n "
                            f"Он перебрал на {(profile.calorie_limit - today_cals) * -1} от нормы 🤬🤬🤬"
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Не удалось отправить напоминание пользователю {profile.username} ({user_id}): {e}")


    return ConversationHandler.END


# --- Вес ---

async def update_weight_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    await update.message.reply_text("Введи новый текущий вес в кг:")
    return UPDATE_WEIGHT


async def handle_update_weight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        weight = float(update.message.text.replace(",", "."))
        if weight <= 0:
            raise ValueError
    except (TypeError, ValueError):
        await update.message.reply_text("Введи положительное число.")
        return UPDATE_WEIGHT

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    if tg_user.id not in users:
        await update.message.reply_text("Сначала введи профиль командой /start.")
        return ConversationHandler.END

    profile = users[tg_user.id]
    old_weight = profile.current_weight
    profile.current_weight = weight
    save_users(users)
    logger.info(f"User {tg_user.username} updated weight: {old_weight} -> {weight}")

    entry = DailyEntry(date=date.today(), user_id=tg_user.id, username=tg_user.username or "", calories=0,
                       weight=weight)
    append_or_update_entry(entry)

    # Пересчитываем метаболизм с новым весом
    deficit = profile.get_deficit_progress()

    await update.message.reply_text(
        f"⚖️ Вес обновлен: {old_weight:.1f} ➡️ {weight:.1f} кг\n\n"
        f"🔥 Метаболизм пересчитан:\n"
        f"   BMR: {deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE: {deficit['tdee']:.0f} ккал/день\n\n"
        f"📉 Осталось сжечь: {format_ru_number(deficit['deficit_remaining'])} ккал\n"
        f"🏆 Звание: {profile.role.value}",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


# --- Статус ---

async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    if tg_user.id not in users:
        await update.message.reply_text("Профиль не найден. Используйте /start")
        return ConversationHandler.END

    profile = users[tg_user.id]
    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    today_calories = int(deficit["today_calories"])

    if deficit["avg_daily_deficit_7d"] > 0 and deficit["deficit_remaining"] > 0:
        days_forecast = f"~{deficit['days_to_goal']:.0f} дней"
    elif deficit["deficit_remaining"] <= 0:
        days_forecast = "🎉 Цель достигнута!"
    else:
        days_forecast = "❌ Дефицита нет (лимит ≥ расход)"

    total = deficit["total_deficit_needed"]
    achieved_effective = deficit["deficit_achieved_effective"]
    if total > 0:
        pct = min(100, achieved_effective / total * 100)
        bar_len = 20
        filled = int(bar_len * pct / 100)
        progress_bar = "█" * filled + "░" * (bar_len - filled)
        progress_text = f"[{progress_bar}] {pct:.1f}%"
    else:
        progress_text = "─" * 22 + " 100%"

    text = (
        f"📊 *Статус на {date.today().strftime('%d.%m.%Y')}*\n\n"
        f"🔥 *Баланс калорий*:\n"
        f"   Потреблено сегодня: {today_calories} ккал\n"
        f"   TDEE (расход): {deficit['tdee']:.0f} ккал\n"
        f"   Дефицит за сегодня: *{deficit['daily_deficit_today']:.0f} ккал*\n\n"
        f"🎯 *Путь к цели*:\n"
        f"   Всего нужно сжечь: {format_ru_number(deficit['total_deficit_needed'])} ккал\n"
        f"   ✅ Уже сжжено по весу: {format_ru_number(deficit['deficit_achieved_weight'])} ккал\n"
        f"   📉 Уже сжжено по калориям: {format_ru_number(deficit['deficit_achieved_calories'])} ккал\n"
        f"   🔥 В зачёт идёт: {format_ru_number(deficit['deficit_achieved_effective'])} ккал\n"
        f"   Осталось сжечь: {format_ru_number(deficit['deficit_remaining'])} ккал\n"
        f"   🗓️ Прогноз (по среднему дефициту 7 дней): {days_forecast}\n\n"
        f"📈 Прогресс: {progress_text}"
    )
    await update.message.reply_text(text, reply_markup=MAIN_MENU_KEYBOARD, parse_mode="Markdown")
    return ConversationHandler.END


# --- Статистика с inline-кнопками ---

async def stats_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    await update.message.reply_text("Что показать?", reply_markup=STATS_SCOPE_KEYBOARD)
    return STATS_SCOPE


async def stats_scope_choose(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END

    text = update.message.text.strip().lower()
    if "моя" in text or "👤" in text:
        context.user_data["stats_scope"] = "personal"
        user_id = update.effective_user.id
    elif "общая" in text or "🌍" in text:
        context.user_data["stats_scope"] = "global"
        user_id = None
    else:
        await update.message.reply_text("Ок, отмена.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END

    available_months = get_available_months(user_id=user_id)

    if not available_months:
        await update.message.reply_text(
            "📭 Пока нет данных для отображения.\n"
            "Добавь калории или вес, чтобы появилась статистика.",
            reply_markup=MAIN_MENU_KEYBOARD
        )
        return ConversationHandler.END

    keyboard = []
    current_year = None

    today = date.today()
    keyboard.append([
        InlineKeyboardButton(
            f"📅 Текущий ({today.month:02d}.{today.year})",
            callback_data=f"stats_{today.year}_{today.month:02d}"
        )
    ])
    keyboard.append([InlineKeyboardButton("──────────────", callback_data="ignore")])

    for year, month in available_months[:12]:
        if year != current_year:
            current_year = year
            keyboard.append([InlineKeyboardButton(f"📆 {year}", callback_data="ignore")])

        month_name = calendar.month_name[month].capitalize()
        keyboard.append([
            InlineKeyboardButton(
                f"{month_name} {year}",
                callback_data=f"stats_{year}_{month:02d}"
            )
        ])

    if len(available_months) > 12:
        keyboard.append([
            InlineKeyboardButton("◀️ Больше месяцев", callback_data="stats_more_12")
        ])

    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="stats_cancel")])

    await update.message.reply_text(
        f"📊 Выбери месяц для просмотра статистики:\n"
        f"Всего доступно: {len(available_months)} мес.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return STATS_MONTH_SELECT


async def stats_month_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "stats_cancel":
        await query.edit_message_text("Отмена.", reply_markup=None)
        await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END

    if data == "ignore":
        return STATS_MONTH_SELECT

    if data.startswith("stats_more_"):
        await query.answer("Показаны все доступные месяцы выше ⬆️")
        return STATS_MONTH_SELECT

    try:
        parts = data.replace("stats_", "").split("_")
        year = int(parts[0])
        month = int(parts[1])
    except (ValueError, IndexError):
        await query.answer("❌ Ошибка формата даты")
        return STATS_MONTH_SELECT

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    scope = context.user_data.get("stats_scope", "personal")

    try:
        if scope == "personal":
            entries = load_entries_for_month(year, month, user_id=tg_user.id)
            img_path = build_calendar_image(
                year=year, month=month, users=users,
                entries=entries, personal_user_id=tg_user.id
            )
            title = "👤 Твоя статистика"
        else:
            entries = load_entries_for_month(year, month, user_id=None)
            img_path = build_calendar_image(
                year=year, month=month, users=users,
                entries=entries, personal_user_id=None
            )
            title = "🌍 Общая статистика"

        month_name = calendar.month_name[month].capitalize()

        with img_path.open("rb") as f:
            await query.message.reply_photo(
                photo=InputFile(f),
                caption=f"{title} за {month_name} {year}",
                reply_markup=MAIN_MENU_KEYBOARD
            )

        await query.edit_message_text(
            f"✅ Отправлена статистика за {month_name} {year}",
            reply_markup=None
        )

    except Exception as e:
        logger.error(f"Ошибка генерации календаря: {e}")
        await query.answer("❌ Ошибка при построении графика")

    return ConversationHandler.END


# --- Настройки ---

async def settings_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None
    if tg_user.id not in users:
        await update.message.reply_text("Сначала создай профиль через /start")
        return ConversationHandler.END

    await update.message.reply_text("Что хотите изменить?", reply_markup=SETTINGS_KEYBOARD)
    return SETTINGS_CHOICE


async def settings_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if "цель" in text.lower():
        await update.message.reply_text("Введи новый целевой вес (кг):", reply_markup=ReplyKeyboardRemove())
        return SETTINGS_NEW_TARGET
    elif "лимит" in text.lower():
        await update.message.reply_text("Введи новый лимит калорий:", reply_markup=ReplyKeyboardRemove())
        return SETTINGS_NEW_LIMIT
    elif "рост" in text.lower() or "возраст" in text.lower() or "пол" in text.lower():
        users = load_users()
        profile = users.get(update.effective_user.id)
        if profile:
            await update.message.reply_text(
                f"Текущие параметры:\n"
                f"📏 Рост: {profile.height_cm} см\n"
                f"🎂 Возраст: {profile.age} лет\n"
                f"👤 Пол: {'Мужской' if profile.gender == 'male' else 'Женский'}\n\n"
                f"Введи новые данные в формате: `рост возраст пол`\n"
                f"Пример: `180 28 male`",
                reply_markup=ReplyKeyboardRemove(),
            )
        return SETTINGS_EDIT_BIOMETRICS
    elif "актив" in text.lower():
        keyboard = ReplyKeyboardMarkup([
            ["🪑 1.2", "🚶 1.375"],
            ["🏃 1.55", "🔥 1.725"],
            ["❌ Отмена"]
        ], resize_keyboard=True)
        await update.message.reply_text("Выбери уровень активности:", reply_markup=keyboard)
        return SETTINGS_EDIT_ACTIVITY
    else:
        await update.message.reply_text("Отмена.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END


async def settings_new_target(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        weight = float(update.message.text.replace(",", "."))
        if weight <= 0:
            raise ValueError
    except:
        await update.message.reply_text("Ошибка. Введи положительное число.")
        return SETTINGS_NEW_TARGET

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    if tg_user.id not in users:
        return ConversationHandler.END

    profile = users[tg_user.id]
    old_target = profile.target_weight
    profile.target_weight = weight
    save_users(users)
    logger.info(f"User {tg_user.username} changed target: {old_target} -> {weight}")

    # Для оценки остатка используем историю калорий, если она есть
    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    await update.message.reply_text(
        f"Цель изменена: {old_target:.1f} ➡️ {weight:.1f} кг\n"
        f"Осталось сжечь: {format_ru_number(deficit['deficit_remaining'])} ккал",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


async def settings_new_limit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        limit = int(update.message.text)
        if limit <= 0:
            raise ValueError
    except:
        await update.message.reply_text("Ошибка. Введи положительное целое число.")
        return SETTINGS_NEW_LIMIT

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    if tg_user.id not in users:
        return ConversationHandler.END

    profile = users[tg_user.id]
    old_limit = profile.calorie_limit
    profile.calorie_limit = limit
    save_users(users)
    logger.info(f"User {tg_user.username} changed limit: {old_limit} -> {limit}")

    # Обновляем прогноз с учётом среднего дефицита за 7 дней
    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    days_forecast = (
        f"~{deficit['days_to_goal']:.0f} дней"
        if deficit['avg_daily_deficit_7d'] > 0 and deficit['deficit_remaining'] > 0
        else ("🎉 Цель достигнута!" if deficit['deficit_remaining'] <= 0 else "❌ Нет дефицита")
    )

    await update.message.reply_text(
        f"Лимит изменен: {old_limit} ➡️ {limit} ккал\n"
        f"Средний дефицит за 7 дней: {deficit['avg_daily_deficit_7d']:.0f} ккал/день\n"
        f"Прогноз: {days_forecast}",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


async def settings_edit_biometrics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        parts = update.message.text.strip().split()
        if len(parts) != 3:
            raise ValueError
        height = int(parts[0])
        age = int(parts[1])
        gender = parts[2].lower()
        if gender not in ["male", "female", "м", "ж"]:
            raise ValueError
        if gender in ["м", "ж"]:
            gender = "male" if gender == "м" else "female"
    except:
        await update.message.reply_text("Ошибка. Формат: `рост возраст пол`\nПример: `180 28 male`")
        return SETTINGS_EDIT_BIOMETRICS

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    if tg_user.id not in users:
        return ConversationHandler.END

    profile = users[tg_user.id]
    profile.height_cm = height
    profile.age = age
    profile.gender = gender
    save_users(users)
    logger.info(f"User {tg_user.username} updated biometrics: {height}cm, {age}yo, {gender}")

    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    await update.message.reply_text(
        f"Параметры обновлены:\n"
        f"📏 Рост: {height} см\n"
        f"🎂 Возраст: {age} лет\n"
        f"👤 Пол: {'Мужской' if gender == 'male' else 'Женский'}\n\n"
        f"🔥 Новый метаболизм:\n"
        f"   BMR: {deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE: {deficit['tdee']:.0f} ккал/день\n"
        f"   Средний дефицит за 7 дней: {deficit['avg_daily_deficit_7d']:.0f} ккал/день",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


async def settings_edit_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    activity_map = {
        "🪑 1.2": 1.2, "1.2": 1.2,
        "🚶 1.375": 1.375, "1.375": 1.375,
        "🏃 1.55": 1.55, "1.55": 1.55,
        "🔥 1.725": 1.725, "1.725": 1.725,
    }

    if text.lower() in ["отмена", "❌ отмена", "cancel"]:
        await update.message.reply_text("Отмена.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END

    new_activity = activity_map.get(text, None)
    if new_activity is None:
        await update.message.reply_text("Выбери одно из значений в меню.")
        return SETTINGS_EDIT_ACTIVITY

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None

    if tg_user.id not in users:
        return ConversationHandler.END

    profile = users[tg_user.id]
    old_activity = profile.activity_level
    profile.activity_level = new_activity
    save_users(users)
    logger.info(f"User {tg_user.username} updated activity: {old_activity} -> {new_activity}")

    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    await update.message.reply_text(
        f"Активность изменена: {old_activity} ➡️ {new_activity}\n"
        f"🔥 Новый метаболизм:\n"
        f"   BMR: {deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE: {deficit['tdee']:.0f} ккал/день\n"
        f"   Средний дефицит за 7 дней: {deficit['avg_daily_deficit_7d']:.0f} ккал/день",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        await update.message.reply_text("Отменено.", reply_markup=MAIN_MENU_KEYBOARD)
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Краткая справка по боту и расчётам."""
    if not is_allowed(update):
        return
    text = (
        "🧠 Как считает бот:\n"
        "- *По весу*: сколько кг ты уже сбросил, × 7 700 ккал → «уже сжжено по весу».\n"
        "- *По калориям*: по каждому дню считает: TDEE − съеденные ккал (если в минусе — 0) "
        "и копит это как «уже сжжено по калориям».\n"
        "- В зачёт идёт максимум из этих двух величин.\n"
        "- Прогноз дней до цели считается по *среднему дефициту за последние 7 дней*.\n\n"
        "Команды:\n"
        "/start — создать или показать профиль\n"
        "/add — добавить калории\n"
        "/weight — обновить вес\n"
        "/status — текущий статус и прогресс\n"
        "/stats — календарь со статистикой\n"
        "/settings — настройки цели, лимита и параметров\n"
        "/help — эта справка"
    )
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=MAIN_MENU_KEYBOARD)


async def calories_reminder_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Периодическое напоминание внести калории (для всех известных пользователей)."""
    try:
        users = load_users()
        if not users:
            return
        for user_id, profile in users.items():
            insult = get_bad_phrase()
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=(
                        "🕒 Напоминание внести калории за сегодня.\n"
                        "Зайди в бота и нажми «🍔 Добавить калории».\n\n"
                        f"{insult}"
                    ),
                )
            except Exception as e:
                logger.warning(f"Не удалось отправить напоминание пользователю {profile.username} ({user_id}): {e}")
    except Exception as e:
        logger.error(f"Ошибка в задаче напоминаний: {e}")


async def send_energy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отправляет случайную колкую фразу из BAD_LIST."""
    if not is_allowed(update):
        return ConversationHandler.END
    insult = get_bad_phrase()
    if not insult:
        await update.message.reply_text("Сегодня без подколов, но калории всё равно запиши.", reply_markup=MAIN_MENU_KEYBOARD)
    else:
        await update.message.reply_text(insult, reply_markup=MAIN_MENU_KEYBOARD)
    return ConversationHandler.END

from telegram.error import TimedOut, NetworkError
from httpcore import ConnectTimeout

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок"""
    logger.error(f"Ошибка при обработке обновления: {context.error}")

    if isinstance(context.error, (TimedOut, NetworkError, ConnectTimeout)):
        logger.warning("Проблемы с соединением Telegram API (таймаут)")
        return

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ Произошла ошибка. Попробуйте позже."
        )
def build_application() -> "ApplicationBuilder":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")

    app = ApplicationBuilder().token(token).build()

    app.add_error_handler(error_handler)

    onboard_conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ONBOARD_WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_weight)],
            ONBOARD_TARGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_target)],
            ONBOARD_LIMIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_limit)],
            ONBOARD_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_height)],
            ONBOARD_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_age)],
            ONBOARD_GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_gender)],
            ONBOARD_ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_activity)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    add_cal_conv = ConversationHandler(
        entry_points=[
            CommandHandler("add", add_calories_entry),
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^🍔 Добавить калории$"), add_calories_entry),
        ],
        states={ADD_CALORIES: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_calories)]},
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    update_weight_conv = ConversationHandler(
        entry_points=[
            CommandHandler("weight", update_weight_start),
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^⚖️ Обновить вес$"), update_weight_start),
        ],
        states={UPDATE_WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_update_weight)]},
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("status", show_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^📊 Мой статус$"), show_status))

    stats_conv = ConversationHandler(
        entry_points=[
            CommandHandler("stats", stats_start),
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^📅 Статистика$"), stats_start),
        ],
        states={
            STATS_SCOPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, stats_scope_choose)],
            STATS_MONTH_SELECT: [CallbackQueryHandler(stats_month_callback)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    settings_conv = ConversationHandler(
        entry_points=[
            CommandHandler("settings", settings_start),
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^⚙️ Настройки$"), settings_start),
        ],
        states={
            SETTINGS_CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, settings_choice)],
            SETTINGS_NEW_TARGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, settings_new_target)],
            SETTINGS_NEW_LIMIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, settings_new_limit)],
            SETTINGS_EDIT_BIOMETRICS: [MessageHandler(filters.TEXT & ~filters.COMMAND, settings_edit_biometrics)],
            SETTINGS_EDIT_ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, settings_edit_activity)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(onboard_conv)
    app.add_handler(add_cal_conv)
    app.add_handler(update_weight_conv)
    app.add_handler(stats_conv)
    app.add_handler(settings_conv)
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("energy", send_energy))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.Regex("^⚡ Получить заряд бодрости$"),
            send_energy,
        )
    )

    # Планировщик напоминаний: каждый день в 15:00 и 22:00 по Москве
    job_queue = app.job_queue
    if job_queue is None:
        logger.warning(
            "JobQueue недоступен (установите python-telegram-bot с extra 'job-queue', "
            "или проверьте окружение), напоминания о калориях работать не будут."
        )
    else:
        job_queue.run_daily(
            calories_reminder_job,
            time=dtime(hour=15, minute=14, tzinfo=MOSCOW_TZ),
            name="calories_reminder_15",
        )
        job_queue.run_daily(
            calories_reminder_job,
            time=dtime(hour=22, minute=0, tzinfo=MOSCOW_TZ),
            name="calories_reminder_22",
        )

    return app


def main() -> None:
    ensure_csv_files()
    logger.info("Инициализация бота...")
    app = build_application()
    logger.info("Бот запущен (Polling)")
    app.run_polling()


if __name__ == "__main__":
    main()