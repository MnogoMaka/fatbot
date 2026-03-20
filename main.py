import csv
import io
import os
import logging
import tempfile
import calendar
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from badlist import BAD_LIST
from scaner_qr import get_calories_from_image
import pytz

try:
    from giga import generate_answer
except ImportError:
    generate_answer = None
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
WORKOUTS_CSV = DATA_DIR / "workouts.csv"
CALENDAR_DIR = DATA_DIR / "calendars"
CALENDAR_DIR.mkdir(exist_ok=True)

MOSCOW_TZ = pytz.timezone("Europe/Moscow")

# Небольшой список колких фраз — можно расширять



def get_bad_phrase() -> str:
    if not BAD_LIST:
        return ""
    return random.choice(BAD_LIST)


# --- Модели данных ---

class GoalMode(str, Enum):
    LOSS = "loss"
    GAIN = "gain"


class Role(str, Enum):
    # --- Режим похудения ---
    FATTY = "Жиртрест"
    PIG = "Кабан"
    ALMOST_FIT = "Почти соска"
    NORMAL = "Норм чел"
    # --- Режим набора ---
    SKELETON = "Скелет"
    THIN = "Дрищ"
    ALMOST_BUFF = "Почти качок"
    BUFF = "Качок"


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
    cheat_meals: int = 0

    def __post_init__(self):
        if self.start_weight is None:
            self.start_weight = self.current_weight

    @property
    def goal_mode(self) -> "GoalMode":
        """Определяет режим: похудение или набор веса."""
        start = self.start_weight if self.start_weight else self.current_weight
        if self.target_weight > start:
            return GoalMode.GAIN
        return GoalMode.LOSS

    @property
    def is_gaining(self) -> bool:
        return self.goal_mode == GoalMode.GAIN

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
        Универсальный расчёт прогресса — работает и для похудения, и для набора.
        today_calories — фактически потреблённые калории сегодня (из entries.csv)
        """
        kcal_per_kg = 7700
        start = self.start_weight if self.start_weight else self.current_weight
        tdee = self.calculate_tdee()

        if self.is_gaining:
            # === РЕЖИМ НАБОРА ВЕСА ===
            total_surplus_needed = max(0, (self.target_weight - start)) * kcal_per_kg
            achieved = max(0, (self.current_weight - start)) * kcal_per_kg
            remaining = max(0, total_surplus_needed - achieved)
            daily_surplus = max(0, today_calories - tdee)
            days_to_goal = remaining / daily_surplus if daily_surplus > 0 else float('inf')
            return {
                'total_deficit_needed': total_surplus_needed,
                'deficit_achieved': achieved,
                'deficit_remaining': remaining,
                'daily_deficit': daily_surplus,
                'days_to_goal': days_to_goal,
                'tdee': tdee,
                'bmr': self.calculate_bmr(),
                'today_calories': today_calories,
            }
        else:
            # === РЕЖИМ ПОХУДЕНИЯ ===
            total_deficit = max(0, (start - self.target_weight)) * kcal_per_kg
            achieved = max(0, (start - self.current_weight)) * kcal_per_kg
            remaining = max(0, total_deficit - achieved)
            daily_deficit = max(0, tdee - today_calories)
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
        """Прогресс к цели в процентах (работает для обоих режимов)."""
        start = self.start_weight if self.start_weight else self.current_weight
        if self.is_gaining:
            # Набор: прогресс = (current - start) / (target - start)
            total = self.target_weight - start
            if total <= 0:
                return 100.0
            progress = (self.current_weight - start) / total
        else:
            # Похудение: прогресс = (start - current) / (start - target)
            if start <= self.target_weight:
                return 100.0
            progress = (start - self.current_weight) / (start - self.target_weight)
        return max(0.0, min(100.0, progress * 100))

    @property
    def weight_progress_percent(self) -> float:
        """Прогресс по весу (алиас для progress_percent)."""
        return self.progress_percent

    @property
    def role(self) -> Role:
        p = self.progress_percent
        if self.is_gaining:
            if p < 25:
                return Role.SKELETON
            if p < 50:
                return Role.THIN
            if p < 80:
                return Role.ALMOST_BUFF
            return Role.BUFF
        else:
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
                "height_cm", "age", "gender", "activity_level", "start_weight", "cheat_meals"
            ])

    if not ENTRIES_CSV.exists():
        logger.info("Создание файла entries.csv")
        with ENTRIES_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "user_id", "username", "calories", "weight", "exercises"])

    if not WORKOUTS_CSV.exists():
        logger.info("Создание файла workouts.csv")
        with WORKOUTS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "user_id", "username", "description"])


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
                    cheat_meals=int(row.get("cheat_meals") or 0),
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
                "height_cm", "age", "gender", "activity_level", "start_weight", "cheat_meals"
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
                    str(u.cheat_meals),
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


def set_entry_calories_for_day(user_id: int, username: str, d: date, calories: int) -> None:
    """Устанавливает (перезаписывает) калории за указанный день для пользователя."""
    ensure_csv_files()
    rows: List[Dict[str, str]] = []
    found = False
    with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["user_id"] == str(user_id) and row["date"] == d.isoformat():
                row["calories"] = str(calories)
                found = True
            rows.append(row)
    if not found:
        rows.append({
            "date": d.isoformat(),
            "user_id": str(user_id),
            "username": username,
            "calories": str(calories),
            "weight": "",
            "exercises": "",
        })
    with ENTRIES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "user_id", "username", "calories", "weight", "exercises"])
        writer.writeheader()
        writer.writerows(rows)


def set_entry_exercises_for_day(user_id: int, username: str, d: date, exercises_text: str) -> None:
    """Записывает тренировку за день в workouts.csv (и дублирует в entries.exercises для совместимости)."""
    ensure_csv_files()
    text = exercises_text.strip()
    if not text:
        return
    # Сохраняем в отдельную таблицу тренировок
    rows: List[Dict[str, str]] = []
    found = False
    if WORKOUTS_CSV.exists():
        with WORKOUTS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("user_id") == str(user_id) and row.get("date") == d.isoformat():
                    row["description"] = text
                    row["username"] = username
                    found = True
                rows.append(row)
    if not found:
        rows.append({
            "date": d.isoformat(),
            "user_id": str(user_id),
            "username": username,
            "description": text,
        })
    with WORKOUTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "user_id", "username", "description"])
        writer.writeheader()
        writer.writerows(rows)
    # Дублируем в entries для обратной совместимости (одна строка на день)
    rows_ent: List[Dict[str, str]] = []
    found_ent = False
    with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row.setdefault("exercises", "")
            if row["user_id"] == str(user_id) and row["date"] == d.isoformat():
                row["exercises"] = text
                found_ent = True
            rows_ent.append(row)
    if not found_ent:
        rows_ent.append({
            "date": d.isoformat(),
            "user_id": str(user_id),
            "username": username,
            "calories": "0",
            "weight": "",
            "exercises": text,
        })
    with ENTRIES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "user_id", "username", "calories", "weight", "exercises"])
        writer.writeheader()
        writer.writerows(rows_ent)


def load_workouts(
    user_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> List[Tuple[date, int, str, str]]:
    """Возвращает список (date, user_id, username, description) из workouts.csv."""
    ensure_csv_files()
    result: List[Tuple[date, int, str, str]] = []
    if not WORKOUTS_CSV.exists():
        return result
    try:
        with WORKOUTS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    d = date.fromisoformat(row["date"])
                    uid = int(row["user_id"])
                    if user_id is not None and uid != user_id:
                        continue
                    if year is not None and d.year != year:
                        continue
                    if month is not None and d.month != month:
                        continue
                    result.append((d, uid, row.get("username", ""), row.get("description", "").strip()))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        logger.error(f"Ошибка загрузки workouts.csv: {e}")
    return result


def get_available_workout_months(user_id: Optional[int] = None) -> List[Tuple[int, int]]:
    """Месяцы, в которых есть хотя бы одна запись о тренировке."""
    workouts = load_workouts(user_id=user_id)
    months_set: Set[Tuple[int, int]] = set((d.year, d.month) for d, _, _, desc in workouts if desc)
    return sorted(months_set, reverse=True)[:24]


def compute_deficit_with_history(
    profile: UserProfile,
    entries: List[DailyEntry],
) -> Dict[str, float]:
    """
    Универсальный расчёт прогресса (и для похудения, и для набора):
    - по весу (объективно),
    - по калориям (накопленный дефицит / профицит),
    - прогноз по среднему за 7 дней.
    """
    kcal_per_kg = 7700
    start = profile.start_weight if profile.start_weight else profile.current_weight
    gaining = profile.is_gaining

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

    if gaining:
        # === РЕЖИМ НАБОРА ВЕСА ===
        total_needed = max(0.0, (profile.target_weight - start)) * kcal_per_kg
        achieved_weight = max(0.0, (profile.current_weight - start)) * kcal_per_kg
        daily_surplus_today = max(0.0, today_calories - tdee)

        # Накопленный профицит по калориям за всё время
        achieved_calories = 0.0
        for d, cals in daily_cals.items():
            day_surplus = max(0.0, cals - tdee)
            achieved_calories += day_surplus

        achieved_effective = max(achieved_weight, achieved_calories)
        remaining = max(0.0, total_needed - achieved_effective)

        # Средний профицит за 7 дней
        total_7 = 0.0
        days_counted = 0
        for i in range(7):
            d = today - timedelta(days=i)
            if d in daily_cals:
                day_surplus = max(0.0, daily_cals[d] - tdee)
                total_7 += day_surplus
                days_counted += 1

        if days_counted:
            avg_daily_7d = total_7 / days_counted
        else:
            planned_surplus = max(0.0, profile.calorie_limit - tdee)
            avg_daily_7d = planned_surplus

        days_to_goal = remaining / avg_daily_7d if avg_daily_7d > 0 else float("inf")

        return {
            "total_deficit_needed": total_needed,       # (в данном случае = total_surplus_needed)
            "deficit_achieved_weight": achieved_weight,  # (= surplus_achieved_weight)
            "deficit_achieved_calories": achieved_calories,
            "deficit_achieved_effective": achieved_effective,
            "deficit_remaining": remaining,              # (= surplus_remaining)
            "daily_deficit_today": daily_surplus_today,   # (= daily_surplus_today)
            "avg_daily_deficit_7d": avg_daily_7d,
            "days_to_goal": days_to_goal,
            "tdee": tdee,
            "bmr": bmr,
            "today_calories": today_calories,
        }
    else:
        # === РЕЖИМ ПОХУДЕНИЯ ===
        total_needed = max(0.0, (start - profile.target_weight)) * kcal_per_kg
        achieved_weight = max(0.0, (start - profile.current_weight)) * kcal_per_kg
        daily_deficit_today = max(0.0, tdee - today_calories)

        achieved_calories = 0.0
        for d, cals in daily_cals.items():
            day_def = max(0.0, tdee - cals)
            achieved_calories += day_def

        achieved_effective = max(achieved_weight, achieved_calories)
        remaining = max(0.0, total_needed - achieved_effective)

        total_7 = 0.0
        days_counted = 0
        for i in range(7):
            d = today - timedelta(days=i)
            if d in daily_cals:
                day_def = max(0.0, tdee - daily_cals[d])
                total_7 += day_def
                days_counted += 1

        if days_counted:
            avg_daily_7d = total_7 / days_counted
        else:
            planned_deficit = max(0.0, tdee - profile.calorie_limit)
            avg_daily_7d = planned_deficit

        days_to_goal = remaining / avg_daily_7d if avg_daily_7d > 0 else float("inf")

        return {
            "total_deficit_needed": total_needed,
            "deficit_achieved_weight": achieved_weight,
            "deficit_achieved_calories": achieved_calories,
            "deficit_achieved_effective": achieved_effective,
            "deficit_remaining": remaining,
            "daily_deficit_today": daily_deficit_today,
            "avg_daily_deficit_7d": avg_daily_7d,
            "days_to_goal": days_to_goal,
            "tdee": tdee,
            "bmr": bmr,
            "today_calories": today_calories,
        }


def compute_streak(user_id: int, entries: List[DailyEntry], calorie_limit: int) -> int:
    """
    Считает стрик — кол-во последовательных дней С ЗАПИСЯМИ и без превышения лимита.
    Дни без записей НЕ считаются «чистыми»: если за прошлый день нет данных — стрик прерван.
    Сегодняшний день пропускается, если ещё не было записей (день не закончен).
    """
    by_date: Dict[date, int] = {}
    for e in entries:
        if e.user_id == user_id:
            by_date[e.date] = by_date.get(e.date, 0) + e.calories

    today = date.today()
    streak = 0
    d = today

    while (today - d).days <= 90:
        if d not in by_date:
            if d == today:
                # Сегодня ещё нет записей — пропускаем, идём на вчера
                d -= timedelta(days=1)
                continue
            else:
                # Прошлый день без записи — стрик прерывается
                break
        if by_date[d] > calorie_limit:
            break
        streak += 1
        d -= timedelta(days=1)

    return streak


def check_and_award_cheat_meal(profile: "UserProfile", entries: List[DailyEntry], users: Dict) -> bool:
    """
    Начисляет читмил если стрик >= 14 и нет нарушения сегодня. Макс 2 читмила.
    Возвращает True, если читмил был начислен.
    """
    today_cals = sum(e.calories for e in entries if e.user_id == profile.user_id and e.date == date.today())
    if today_cals > profile.calorie_limit:
        return False
    streak = compute_streak(profile.user_id, entries, profile.calorie_limit)
    if streak >= 14 and profile.cheat_meals < 2:
        profile.cheat_meals += 1
        save_users(users)
        return True
    return False



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


def build_sports_calendar_image(
    *,
    year: int,
    month: int,
    workouts: List[Tuple[date, int, str, str]],
    personal_user_id: Optional[int] = None,
) -> Path:
    """Календарь: дни с тренировками отмечены (зелёный/иконка). В общем — подписи ников."""
    cal = calendar.Calendar(firstweekday=0)
    month_days = [d for d in cal.itermonthdates(year, month) if d.month == month]
    daily_has_sport: Dict[date, bool] = {d: False for d in month_days}
    daily_usernames: Dict[date, List[str]] = {d: [] for d in month_days}
    for d, uid, username, desc in workouts:
        if d not in daily_has_sport:
            continue
        if personal_user_id is not None and uid != personal_user_id:
            continue
        if desc:
            daily_has_sport[d] = True
            nick = f"@{username}" if username else f"id{uid}"
            if nick not in daily_usernames[d]:
                daily_usernames[d].append(nick)
    weeks = calendar.monthcalendar(year, month)
    n_weeks = len(weeks)
    fig, ax = plt.subplots(figsize=(14, 2.2 + 1.6 * n_weeks))
    for week_idx, week in enumerate(weeks):
        for dow_idx, day_num in enumerate(week):
            if day_num == 0:
                continue
            d = date(year, month, day_num)
            has_sport = daily_has_sport.get(d, False)
            cell_bg = "#C8E6C9" if has_sport else "#FAFAFA"
            rect = plt.Rectangle(
                (dow_idx, n_weeks - week_idx - 1), 1, 1,
                facecolor=cell_bg, edgecolor="#E0E0E0", linewidth=0.5
            )
            ax.add_patch(rect)
            ax.text(dow_idx + 0.02, n_weeks - week_idx - 0.15, str(day_num),
                    ha="left", va="top", fontsize=10, color="#666666", weight="bold")
            if has_sport:
                if personal_user_id is None and daily_usernames.get(d):
                    # Общая статистика: подписываем ники
                    label = "\n".join(daily_usernames[d][:5])
                    if len(daily_usernames[d]) > 5:
                        label += "\n..."
                    ax.text(dow_idx + 0.5, n_weeks - week_idx - 0.65, label,
                            ha="center", va="center", fontsize=6, color="#1B5E20")
                else:
                    ax.text(dow_idx + 0.5, n_weeks - week_idx - 0.65, "🏃",
                            ha="center", va="center", fontsize=14)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, n_weeks)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"], fontsize=9)
    ax.set_yticks([])
    month_name = calendar.month_name[month].capitalize()
    title = f"{'Мои' if personal_user_id else 'Общие'} тренировки — {month_name} {year}"
    ax.set_title(title, fontsize=13, pad=20)
    ax.axis("off")
    fig.tight_layout()
    filename = f"sport_personal_{personal_user_id}_{year}_{month}.png" if personal_user_id else f"sport_global_{year}_{month}.png"
    path = CALENDAR_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# --- Telegram-бот ---

(
    ONBOARD_WEIGHT, ONBOARD_TARGET, ONBOARD_LIMIT,
    ONBOARD_HEIGHT, ONBOARD_AGE, ONBOARD_GENDER, ONBOARD_ACTIVITY,
    ADD_CALORIES, ADD_CALORIES_CHOICE, ADD_CALORIES_BARCODE, ADD_CALORIES_GRAMMS,
    UPDATE_WEIGHT,
    STATS_SCOPE, STATS_MONTH_SELECT,
    SETTINGS_CHOICE, SETTINGS_NEW_TARGET, SETTINGS_NEW_LIMIT,
    SETTINGS_EDIT_BIOMETRICS, SETTINGS_EDIT_ACTIVITY,
    AGENT_CHAT,
    EDIT_CAL_MONTH, EDIT_CAL_DAY, EDIT_CAL_VALUE,
    SPORT_MONTH, SPORT_DAY, SPORT_DESC,
    SPORTS_CAL_SCOPE, SPORTS_CAL_MONTH,
    CHANGE_DIET_CONFIRM,
) = range(29)

MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["🍔 Добавить калории", "⚖️ Обновить вес"],
        ["📊 Мой статус", "📅 Статистика"],
        ["✏️ Изменить ККЛ за день", "🏃 Добавить тренировку"],
        ["📋 Календарь тренировок", "📋 Мои записи об упражнениях"],
        ["🏆 Рейтинг"],
        ["🔄 Изменить диету"],
        ["⚙️ Настройки", "⚡ Получить заряд бодрости", "💬 Агент"],
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

ADD_CALORIES_CHOICE_KEYBOARD = ReplyKeyboardMarkup(
    [["1️⃣ Ввести число"], ["2️⃣ Отсканировать штрихкод"]],
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
    context.user_data.clear()
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
        gaining = profile.is_gaining
        remaining_label = "📈 Осталось набрать" if gaining else "📉 Осталось сжечь"
        await update.message.reply_text(
            f"С возвращением, {tg_user.first_name}!\n\n"
            f"{'📈 Режим: набор веса' if gaining else '📉 Режим: похудение'}\n"
            f"⚖️ Вес: {profile.current_weight:.1f} кг (цель: {profile.target_weight:.1f})\n"
            f"🔥 Лимит: {profile.calorie_limit} ккал | TDEE: {deficit['tdee']:.0f} ккал\n"
            f"{remaining_label}: {format_ru_number(deficit['deficit_remaining'])} ккал\n"
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

    # На старте используем чисто расчётный показатель
    base_deficit = profile.get_deficit_progress()
    gaining = profile.is_gaining
    daily_val = base_deficit["daily_deficit"]
    days_forecast = (
        f"~{base_deficit['days_to_goal']:.0f} дней"
        if daily_val > 0
        else ("❌ Нет профицита" if gaining else "❌ Нет дефицита")
    )

    logger.info(
        f"Новый пользователь: {tg_user.username}, "
        f"BMR={base_deficit['bmr']:.0f}, TDEE={base_deficit['tdee']:.0f}, "
        f"mode={'gain' if gaining else 'loss'}"
    )

    action_word = "набрать" if gaining else "сжечь"
    daily_label = "Ежедневный профицит" if gaining else "Ежедневный дефицит"
    mode_label = "📈 Режим: набор веса" if gaining else "📉 Режим: похудение"

    await update.message.reply_text(
        f"✅ Профиль готов!\n"
        f"{mode_label}\n\n"
        f"🔥 Твой метаболизм:\n"
        f"   BMR (покой): {base_deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE (с активностью): {base_deficit['tdee']:.0f} ккал/день\n\n"
        f"🎯 Для цели нужно {action_word}: {format_ru_number(base_deficit['total_deficit_needed'])} ккал\n"
        f"📊 При лимите {profile.calorie_limit} ккал/день:\n"
        f"   {daily_label}: ~{daily_val:.0f} ккал\n"
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
    gaining = profile.is_gaining
    daily_val = deficit['daily_deficit']
    days_forecast = (
        f"~{deficit['days_to_goal']:.0f} дней"
        if daily_val > 0
        else ("❌ Нет профицита" if gaining else "❌ Нет дефицита")
    )

    logger.info(f"Новый пользователь: {tg_user.username}, BMR={deficit['bmr']:.0f}, TDEE={deficit['tdee']:.0f}")

    action_word = "набрать" if gaining else "сжечь"
    daily_label = "Ежедневный профицит" if gaining else "Ежедневный дефицит"
    mode_label = "📈 Режим: набор веса" if gaining else "📉 Режим: похудение"

    await update.message.reply_text(
        f"✅ Профиль готов!\n"
        f"{mode_label}\n\n"
        f"🔥 Твой метаболизм:\n"
        f"   BMR (покой): {deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE (с активностью): {deficit['tdee']:.0f} ккал/день\n\n"
        f"🎯 Для цели нужно {action_word}: {format_ru_number(deficit['total_deficit_needed'])} ккал\n"
        f"📊 При лимите {profile.calorie_limit} ккал/день:\n"
        f"   {daily_label}: ~{format_ru_number(daily_val)} ккал\n"
        f"   Прогноз до цели: {days_forecast}\n\n"
        f"🏆 Звание: {profile.role.value}",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


# --- Калории ---

async def add_calories_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    await update.message.reply_text(
        "Введи количество калорий (число суммируется к дню):",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ADD_CALORIES


async def handle_add_calories_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    text = (update.message.text or "").strip()
    if "1" in text or "число" in text.lower() or "ввести" in text.lower():
        await update.message.reply_text(
            "Сколько калорий в этом приёме пищи? (число суммируется к сегодняшнему):",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ADD_CALORIES
    if "2" in text or "штрихкод" in text.lower() or "отсканировать" in text.lower():
        await update.message.reply_text(
            "Отправь фото со штрихкодом продукта.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ADD_CALORIES_BARCODE
    await update.message.reply_text("Выбери вариант: 1 — ввести число, 2 — отсканировать штрихкод.", reply_markup=ADD_CALORIES_CHOICE_KEYBOARD)
    return ADD_CALORIES_CHOICE


async def handle_add_calories_barcode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    if not update.message.photo:
        await update.message.reply_text("Нужно отправить именно фото со штрихкодом. Либо введи калории числом:")
        return ADD_CALORIES

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None
    if tg_user.id not in users:
        await update.message.reply_text("Сначала введи профиль командой /start.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END

    photo = update.message.photo[-1]
    tg_file = await photo.get_file()
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
        await tg_file.download_to_drive(path)
        result = get_calories_from_image(path)
    except Exception as e:
        logger.exception("Barcode scan failed: %s", e)
        result = {"error": str(e)}
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass

    if "error" in result:
        await update.message.reply_text(
            f"❌ {result['error']}\nВведи калории вручную (число):",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return ADD_CALORIES

    calories_per_100g = result.get("calories_per_100g")
    if calories_per_100g is None:
        await update.message.reply_text(
            "У этого продукта в базе нет калорийности. Введи калории вручную (число):",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return ADD_CALORIES

    context.user_data["barcode_product"] = result
    await update.message.reply_text(
        f"✅ Найден продукт: *{result.get('product_name', '?')}*\n"
        f"Калорийность: {calories_per_100g} ккал/100 г.\n\n"
        "Введи граммовку (сколько грамм съел):",
        parse_mode="Markdown",
    )
    return ADD_CALORIES_GRAMMS


async def handle_add_calories_gramms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        grams = float(update.message.text.replace(",", "."))
        if grams <= 0:
            raise ValueError
    except (TypeError, ValueError):
        await update.message.reply_text("Введи положительное число грамм (например: 150).")
        return ADD_CALORIES_GRAMMS

    product = context.user_data.get("barcode_product")
    if not product:
        await update.message.reply_text("Сессия сброшена. Введи калории числом:", reply_markup=MAIN_MENU_KEYBOARD)
        return ADD_CALORIES

    calories_per_100g = product.get("calories_per_100g") or 0
    calories = int(round(grams * calories_per_100g / 100.0))

    users = load_users()
    tg_user = update.effective_user
    assert tg_user is not None
    if tg_user.id not in users:
        await update.message.reply_text("Сначала введи профиль командой /start.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END

    entry = DailyEntry(date=date.today(), user_id=tg_user.id, username=tg_user.username or "", calories=calories)
    append_or_update_entry(entry)
    logger.info(f"User {tg_user.username} added {calories} kcal (barcode product, {grams}g)")

    profile = users[tg_user.id]
    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    insult = get_bad_phrase()
    gaining = profile.is_gaining
    remaining_label = "Осталось набрать до цели" if gaining else "Осталось сжечь до цели"
    await update.message.reply_text(
        f"Записал +{calories} ккал ({product.get('product_name', '?')}, {grams:.0f} г).\n"
        f"{insult}\n"
        f"{remaining_label}: {format_ru_number(deficit['deficit_remaining'])} ккал",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    today_cals = int(deficit["today_calories"])
    ccl = profile.calorie_limit
    if not profile.is_gaining and today_cals > ccl * 1.10:
        if profile.cheat_meals > 0:
            profile.cheat_meals -= 1
            save_users(users)
            await update.message.reply_text(
                f"🍕 Ты превысил лимит, но у тебя был читмил! Он использован, стрик сохранён.\n"
                f"Осталось читмилов: {profile.cheat_meals}"
            )
        else:
            await update.message.reply_text(f"АХАХАХА ну ты и лох, жри дальше. Теперь все об этом знают.")
            for other_uid, prof in users.items():
                if other_uid != tg_user.id:
                    try:
                        await context.bot.send_message(
                            chat_id=other_uid,
                            text=(
                                f"Поздравьте ЖИРОБАСА @{tg_user.username}. Он сегодня объелся как свинья."
                                f"Он перебрал на {today_cals - ccl} ккал от нормы 🤬🤬🤬"
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Не удалось отправить сообщение {prof.username} ({other_uid}): {e}")
    awarded_bar = check_and_award_cheat_meal(profile, all_entries, users)
    if awarded_bar:
        streak_now = compute_streak(tg_user.id, all_entries, profile.calorie_limit)
        await update.message.reply_text(
            f"🎉 Стрик {streak_now} дней без нарушений! +1 читмил. Всего: {profile.cheat_meals}"
        )
    context.user_data.pop("barcode_product", None)
    return ConversationHandler.END


async def _do_save_calories(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    tg_user,
    users,
    calories: int,
    target_date,
) -> int:
    """Saves calories for target_date and sends confirmation."""
    entry = DailyEntry(date=target_date, user_id=tg_user.id, username=tg_user.username or "", calories=calories)
    append_or_update_entry(entry)
    logger.info(f"User {tg_user.username} added {calories} kcal for {target_date}")

    profile = users[tg_user.id]
    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    insult = get_bad_phrase()
    gaining = profile.is_gaining
    remaining_label = "Осталось набрать до цели" if gaining else "Осталось сжечь до цели"
    date_label = f" (за {target_date.strftime('%d.%m')})" if target_date != date.today() else ""
    await update.message.reply_text(
        f"Записал +{calories} ккал{date_label}.\n"
        f"{insult}\n"
        f"{remaining_label}: {format_ru_number(deficit['deficit_remaining'])} ккал",
        reply_markup=MAIN_MENU_KEYBOARD,
    )

    today_cals = int(deficit["today_calories"])
    ccl = profile.calorie_limit
    if not profile.is_gaining and today_cals > ccl * 1.10:
        if profile.cheat_meals > 0:
            profile.cheat_meals -= 1
            save_users(users)
            await update.message.reply_text(
                f"🍕 Ты превысил лимит, но у тебя был читмил! Он использован, стрик сохранён.\n"
                f"Осталось читмилов: {profile.cheat_meals}"
            )
        else:
            await update.message.reply_text("АХАХАХА ну ты и лох, жри дальше. Теперь все об этом знают")
            for other_uid, prof in users.items():
                if other_uid != tg_user.id:
                    try:
                        await context.bot.send_message(
                            chat_id=other_uid,
                            text=(
                                f"Поздравьте ЖИРОБАСА @{tg_user.username}. Он сегодня объелся как свинья.\n"
                                f"Он перебрал на {today_cals - ccl} ккал от нормы 🤬🤬🤬"
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Не удалось отправить сообщение {prof.username} ({other_uid}): {e}")
    awarded_cal = check_and_award_cheat_meal(profile, all_entries, users)
    if awarded_cal:
        streak_now = compute_streak(tg_user.id, all_entries, profile.calorie_limit)
        await update.message.reply_text(
            f"🎉 Стрик {streak_now} дней без нарушений! +1 читмил. Всего: {profile.cheat_meals}"
        )
    return ConversationHandler.END


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

    # Check if time is between 00:00 and 02:00 — offer to log for yesterday
    now_msk = datetime.now(MOSCOW_TZ)
    if 0 <= now_msk.hour < 2:
        context.user_data["pending_calories"] = calories
        context.user_data["pending_users"] = users
        today = date.today()
        yesterday = today - timedelta(days=1)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    f"📅 Сегодня ({today.strftime('%d.%m')})",
                    callback_data="add_cal_day_today"
                ),
                InlineKeyboardButton(
                    f"📅 Вчера ({yesterday.strftime('%d.%m')})",
                    callback_data="add_cal_day_yesterday"
                ),
            ]
        ])
        await update.message.reply_text(
            f"За какой день записать {calories} ккал?",
            reply_markup=keyboard,
        )
        return ADD_CALORIES_CHOICE

    return await _do_save_calories(update, context, tg_user, users, calories, date.today())


async def handle_add_calories_day_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Callback for the today/yesterday inline keyboard after midnight."""
    query = update.callback_query
    await query.answer()

    calories = context.user_data.get("pending_calories")
    users = context.user_data.get("pending_users")
    tg_user = update.effective_user

    if calories is None or users is None or tg_user is None:
        await query.edit_message_text("Что-то пошло не так, попробуй снова.")
        return ConversationHandler.END

    if query.data == "add_cal_day_yesterday":
        target_date = date.today() - timedelta(days=1)
    else:
        target_date = date.today()

    await query.edit_message_reply_markup(reply_markup=None)
    # Reload users to get fresh data
    users = load_users()
    return await _do_save_calories(query, context, tg_user, users, calories, target_date)


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
    gaining = profile.is_gaining

    if gaining:
        remaining_label = "📈 Осталось набрать"
    else:
        remaining_label = "📉 Осталось сжечь"

    await update.message.reply_text(
        f"⚖️ Вес обновлен: {old_weight:.1f} ➡️ {weight:.1f} кг\n\n"
        f"🔥 Метаболизм пересчитан:\n"
        f"   BMR: {deficit['bmr']:.0f} ккал/день\n"
        f"   TDEE: {deficit['tdee']:.0f} ккал/день\n\n"
        f"{remaining_label}: {format_ru_number(deficit['deficit_remaining'])} ккал\n"
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
    gaining = profile.is_gaining

    # --- Прогноз дней ---
    if deficit["avg_daily_deficit_7d"] > 0 and deficit["deficit_remaining"] > 0:
        days_forecast = f"~{deficit['days_to_goal']:.0f} дней"
    elif deficit["deficit_remaining"] <= 0:
        days_forecast = "🎉 Цель достигнута!"
    else:
        if gaining:
            days_forecast = "❌ Профицита нет (лимит ≤ расход)"
        else:
            days_forecast = "❌ Дефицита нет (лимит ≥ расход)"

    # --- Прогресс-бар по ВЕСУ ---
    weight_pct = profile.weight_progress_percent
    bar_len = 20
    w_filled = int(bar_len * weight_pct / 100)
    weight_bar = "█" * w_filled + "░" * (bar_len - w_filled)
    weight_bar_text = f"[{weight_bar}] {weight_pct:.1f}%"

    # --- Прогресс-бар по КАЛОРИЯМ ---
    total = deficit["total_deficit_needed"]
    achieved_cals = deficit["deficit_achieved_calories"]
    if total > 0:
        cal_pct = min(100, achieved_cals / total * 100)
        c_filled = int(bar_len * cal_pct / 100)
        cal_bar = "█" * c_filled + "░" * (bar_len - c_filled)
        cal_bar_text = f"[{cal_bar}] {cal_pct:.1f}%"
    else:
        cal_bar_text = "─" * 22 + " 100%"

    if gaining:
        # === РЕЖИМ НАБОРА ===
        action_word = "набрать"
        action_done_weight = "Набрано по весу"
        action_done_cal = "Набрано по калориям"
        action_effective = "В зачёт идёт"
        action_remaining = "Осталось набрать"
        daily_label = "Профицит за сегодня"
        avg_label = "среднему профициту"
    else:
        # === РЕЖИМ ПОХУДЕНИЯ ===
        action_word = "сжечь"
        action_done_weight = "Сожжено по весу"
        action_done_cal = "Сожжено по калориям"
        action_effective = "В зачёт идёт"
        action_remaining = "Осталось сжечь"
        daily_label = "Дефицит за сегодня"
        avg_label = "среднему дефициту"

    # --- Стрик, читмилы, звание ---
    users_now = load_users()
    profile_fresh = users_now.get(tg_user.id, profile)
    streak = compute_streak(profile_fresh.user_id, all_entries, profile_fresh.calorie_limit)
    awarded = check_and_award_cheat_meal(profile_fresh, all_entries, users_now)
    if awarded:
        await update.message.reply_text(
            f"🎉 Стрик {streak} дней без нарушений! +1 читмил. Всего: {profile_fresh.cheat_meals}"
        )

    streak_str = f"\n🔥 *Стрик*: {streak} дней без нарушений"
    cheat_str = ""
    if profile_fresh.cheat_meals > 0:
        cheat_str = f"\n🍕 *Читмилы*: {profile_fresh.cheat_meals} шт. (защищают стрик при превышении лимита)"

    today_status = ""
    if not gaining and today_calories > profile_fresh.calorie_limit * 1.10:
        overage = today_calories - profile_fresh.calorie_limit
        today_status = f"\n🚨 *Статус сегодня*: ЖИРОБАС — перебор на {overage} ккал!"
    elif gaining and profile_fresh.calorie_limit > 0 and 0 < today_calories < profile_fresh.calorie_limit * 0.80:
        today_status = f"\n⚠️ *Статус сегодня*: СКЕЛЕТ — недобор {profile_fresh.calorie_limit - today_calories} ккал"

    text = (
        f"📊 *Статус на {date.today().strftime('%d.%m.%Y')}*\n"
        f"{'📈 Режим: набор веса' if gaining else '📉 Режим: похудение'}\n"
        f"🏆 *Звание*: {profile_fresh.role.value}"
        f"{today_status}"
        f"{streak_str}"
        f"{cheat_str}\n\n"
        f"🔥 *Баланс калорий*:\n"
        f"   Потреблено сегодня: {today_calories} ккал\n"
        f"   TDEE (расход): {deficit['tdee']:.0f} ккал\n"
        f"   {daily_label}: *{deficit['daily_deficit_today']:.0f} ккал*\n\n"
        f"🎯 *Путь к цели*:\n"
        f"   Всего нужно {action_word}: {format_ru_number(deficit['total_deficit_needed'])} ккал\n"
        f"   ✅ {action_done_weight}: {format_ru_number(deficit['deficit_achieved_weight'])} ккал\n"
        f"   {'📈' if gaining else '📉'} {action_done_cal}: {format_ru_number(deficit['deficit_achieved_calories'])} ккал\n"
        f"   🔥 {action_effective}: {format_ru_number(deficit['deficit_achieved_effective'])} ккал\n"
        f"   {action_remaining}: {format_ru_number(deficit['deficit_remaining'])} ккал\n"
        f"   🗓️ Прогноз (по {avg_label} 7 дней): {days_forecast}\n\n"
        f"⚖️ *Прогресс по весу*: {weight_bar_text}\n"
        f"🔥 *Прогресс по калориям*: {cal_bar_text}"
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
    for year, month in available_months[:12]:
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
        "📊 Выбери месяц для просмотра статистики:",
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

        photo_bytes = img_path.read_bytes()
        await query.message.reply_photo(
            photo=InputFile(io.BytesIO(photo_bytes)),
            caption=f"{title} за {month_name} {year}",
            reply_markup=MAIN_MENU_KEYBOARD
        )

        await query.edit_message_text(
            f"✅ Отправлена статистика за {month_name} {year}",
            reply_markup=None
        )

    except Exception as e:
        logger.exception("Ошибка генерации календаря калорий: %s", e)
        try:
            await query.message.reply_text("❌ Ошибка при построении графика. Попробуй ещё раз.", reply_markup=MAIN_MENU_KEYBOARD)
        except Exception:
            pass

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
    gaining = profile.is_gaining
    remaining_label = "Осталось набрать" if gaining else "Осталось сжечь"
    mode_label = "📈 Режим: набор веса" if gaining else "📉 Режим: похудение"
    await update.message.reply_text(
        f"Цель изменена: {old_target:.1f} ➡️ {weight:.1f} кг\n"
        f"{mode_label}\n"
        f"{remaining_label}: {format_ru_number(deficit['deficit_remaining'])} ккал",
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

    # Обновляем прогноз с учётом среднего дефицита/профицита за 7 дней
    all_entries = load_entries_for_user(tg_user.id)
    deficit = compute_deficit_with_history(profile, all_entries)
    gaining = profile.is_gaining
    no_progress_label = "❌ Нет профицита" if gaining else "❌ Нет дефицита"
    avg_label = "Средний профицит за 7 дней" if gaining else "Средний дефицит за 7 дней"
    days_forecast = (
        f"~{deficit['days_to_goal']:.0f} дней"
        if deficit['avg_daily_deficit_7d'] > 0 and deficit['deficit_remaining'] > 0
        else ("🎉 Цель достигнута!" if deficit['deficit_remaining'] <= 0 else no_progress_label)
    )

    await update.message.reply_text(
        f"Лимит изменен: {old_limit} ➡️ {limit} ккал\n"
        f"{avg_label}: {deficit['avg_daily_deficit_7d']:.0f} ккал/день\n"
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
        f"   {'Средний профицит' if profile.is_gaining else 'Средний дефицит'} за 7 дней: {deficit['avg_daily_deficit_7d']:.0f} ккал/день",
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
        f"   {'Средний профицит' if profile.is_gaining else 'Средний дефицит'} за 7 дней: {deficit['avg_daily_deficit_7d']:.0f} ккал/день",
        reply_markup=MAIN_MENU_KEYBOARD,
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        if update.message:
            await update.message.reply_text("Отменено.", reply_markup=MAIN_MENU_KEYBOARD)
    except Exception as e:
        # Логируем ошибку, но всё равно завершаем диалог
        logger.error(f"Ошибка при отправке 'Отменено.': {e}")
    finally:
        return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Краткая справка по боту и расчётам."""
    if not is_allowed(update):
        return
    text = (
        "🧠 Как считает бот:\n"
        "Бот поддерживает два режима: *похудение* и *набор веса*.\n"
        "Режим определяется автоматически: если целевой вес > стартового — набор, иначе — похудение.\n\n"
        "*Режим похудения:*\n"
        "- *По весу*: сколько кг ты уже сбросил × 7 700 ккал → «сожжено по весу».\n"
        "- *По калориям*: по каждому дню: TDEE − съеденные ккал (мин 0) → «сожжено по калориям».\n\n"
        "*Режим набора:*\n"
        "- *По весу*: сколько кг ты уже набрал × 7 700 ккал → «набрано по весу».\n"
        "- *По калориям*: по каждому дню: съеденные ккал − TDEE (мин 0) → «набрано по калориям».\n\n"
        "- В зачёт идёт максимум из двух величин.\n"
        "- Прогноз считается по среднему за *последние 7 дней*.\n"
        "- Прогресс показывается двумя барами: по весу и по калориям.\n\n"
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


def build_user_context(user_id: int) -> str:
    """Собирает текст о пользователе из таблиц для контекста агента."""
    users = load_users()
    profile = users.get(user_id)
    if not profile:
        return "Пользователь не найден в базе."
    entries = load_entries_for_user(user_id)
    deficit = compute_deficit_with_history(profile, entries)
    gaining = profile.is_gaining
    mode_str = "набор веса" if gaining else "похудение"
    remaining_str = "Осталось набрать до цели" if gaining else "Осталось сжечь до цели"
    lines = [
        "Данные пользователя из бота:",
        f"Режим: {mode_str}.",
        f"Вес: {profile.current_weight:.1f} кг, цель: {profile.target_weight:.1f} кг.",
        f"Лимит калорий: {profile.calorie_limit} ккал/день.",
        f"Рост: {profile.height_cm} см, возраст: {profile.age}, пол: {profile.gender}.",
        f"TDEE (расход): {deficit.get('tdee', profile.calculate_tdee()):.0f} ккал.",
        f"{remaining_str}: {format_ru_number(deficit.get('deficit_remaining', 0))} ккал.",
    ]
    if entries:
        by_date = defaultdict(int)
        for e in entries:
            by_date[e.date] += e.calories
        recent = sorted(by_date.items(), reverse=True)[:14]
        lines.append("Калории по дням (последние 2 недели): " + ", ".join(f"{d}: {c}" for d, c in recent))
        # Тренировки из workouts.csv (основной источник)
        wkts = load_workouts(user_id=user_id)
        if wkts:
            wkts_sorted = sorted(wkts, key=lambda x: x[0], reverse=True)[:14]
            wkts_lines = [
                f"{d.strftime('%d.%m')}: {desc[:60]}{'...' if len(desc) > 60 else ''}"
                for d, _, _, desc in wkts_sorted if desc
            ]
            if wkts_lines:
                lines.append("Тренировки (последние 14 записей): " + "; ".join(wkts_lines))
        else:
            with_ex = [(e.date, e.exercises) for e in entries if e.exercises and e.exercises.strip()]
            if with_ex:
                lines.append("Тренировки: " + "; ".join(
                    f"{d}: {e[:50]}..." if len(e) > 50 else f"{d}: {e}" for d, e in with_ex[-10:]
                ))
        # Звание, стрик и читмилы для контекста LLM
        streak = compute_streak(user_id, entries, profile.calorie_limit)
        lines.append(f"Звание: {profile.role.value}")
        lines.append(f"Стрик (дней без нарушений): {streak}")
        if profile.cheat_meals > 0:
            lines.append(f"Читмилы доступны: {profile.cheat_meals} шт.")
    return "\n".join(lines)


async def agent_button_hint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Подсказка при нажатии кнопки «💬 Агент» — сам агент запускается только по /agent."""
    if not is_allowed(update):
        return
    await update.message.reply_text(
        "Для чата с агентом отправь команду /agent",
        reply_markup=MAIN_MENU_KEYBOARD,
    )


async def agent_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    if not generate_answer:
        await update.message.reply_text("Агент недоступен (модуль giga не подключён).", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    await update.message.reply_text(
        "💬 Режим агента. Напиши вопрос или сообщение — ответит помощник с учётом твоих данных из бота.\nДля выхода из режима общения напиши: «Выйти».",
        reply_markup=ReplyKeyboardRemove(),
    )
    return AGENT_CHAT


async def agent_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    if not update.message or not update.message.text:
        return AGENT_CHAT
    text = update.message.text.strip()
    # Выход: по команде /cancel или по словам "выйти"/"выход"/"отмена"
    if text.startswith("/cancel") or "выйти" in text.lower() or "выход" in text.lower() or text.lower() == "отмена":
        await update.message.reply_text("Выход из режима агента.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    tg_user = update.effective_user
    if not tg_user:
        return AGENT_CHAT
    ctx = build_user_context(tg_user.id)
    full_query = f"{ctx}\n\nВопрос пользователя: {text}"
    try:
        answer = generate_answer(full_query)
        if answer is None:
            answer = "Не удалось получить ответ. Попробуй ещё раз."
        elif isinstance(answer, dict):
            answer = answer.get("content", str(answer))
        await update.message.reply_text(str(answer)[:4000])
    except Exception as e:
        logger.exception("Ошибка агента: %s", e)
        await update.message.reply_text("Ошибка при обращении к агенту. Попробуй позже.")
    return AGENT_CHAT


# --- Изменить ККЛ за день ---

async def edit_cal_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    months = get_available_months(user_id=update.effective_user.id) if update.effective_user else []
    today = date.today()
    if (today.year, today.month) not in months:
        months.insert(0, (today.year, today.month))
    if not months:
        await update.message.reply_text("Нет записей. Сначала добавь калории.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    keyboard = []
    for y, m in months[:12]:
        keyboard.append([InlineKeyboardButton(f"{calendar.month_name[m]} {y}", callback_data=f"editcal_{y}_{m:02d}")])
    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="editcal_cancel")])
    await update.message.reply_text("Выбери месяц:", reply_markup=InlineKeyboardMarkup(keyboard))
    return EDIT_CAL_MONTH


async def edit_cal_cancel_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Отмена.")
    await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
    return ConversationHandler.END


async def edit_cal_month_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "editcal_cancel":
        await edit_cal_cancel_cb(update, context)
        return ConversationHandler.END
    if not query.data.startswith("editcal_"):
        return EDIT_CAL_MONTH
    parts = query.data.replace("editcal_", "").split("_")
    if len(parts) < 2:
        return EDIT_CAL_MONTH
    try:
        y, m = int(parts[0]), int(parts[1])
    except ValueError:
        return EDIT_CAL_MONTH
    context.user_data["edit_cal_year"], context.user_data["edit_cal_month"] = y, m
    cal = calendar.Calendar(firstweekday=0)
    days = [d for d in cal.itermonthdates(y, m) if d.month == m]
    keyboard = []
    row = []
    for i, d in enumerate(days):
        row.append(InlineKeyboardButton(str(d.day), callback_data=f"editcalday_{d.day:02d}"))
        if len(row) == 7:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="editcal_cancel")])
    await query.edit_message_text(f"Выбери день ({calendar.month_name[m]} {y}):", reply_markup=InlineKeyboardMarkup(keyboard))
    return EDIT_CAL_DAY


async def edit_cal_day_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "editcal_cancel":
        await query.edit_message_text("Отмена.")
        await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    if not query.data.startswith("editcalday_"):
        return EDIT_CAL_DAY
    try:
        day = int(query.data.replace("editcalday_", ""))
    except ValueError:
        return EDIT_CAL_DAY
    y, m = context.user_data.get("edit_cal_year"), context.user_data.get("edit_cal_month")
    if not y or not m:
        return ConversationHandler.END
    try:
        d = date(y, m, day)
    except ValueError:
        await query.answer("Неверная дата")
        return EDIT_CAL_DAY
    context.user_data["edit_cal_date"] = d
    await query.edit_message_text(f"Введи новое значение калорий за {d.strftime('%d.%m.%Y')} (одним числом):")
    return EDIT_CAL_VALUE


async def edit_cal_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    try:
        cal_val = int(update.message.text.strip())
        if cal_val < 0:
            raise ValueError
    except (TypeError, ValueError):
        await update.message.reply_text("Введи неотрицательное целое число.")
        return EDIT_CAL_VALUE
    d = context.user_data.get("edit_cal_date")
    tg_user = update.effective_user
    if not d or not tg_user:
        await update.message.reply_text("Ошибка. Начни заново.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    set_entry_calories_for_day(tg_user.id, tg_user.username or "", d, cal_val)
    await update.message.reply_text(f"За {d.strftime('%d.%m.%Y')} установлено {cal_val} ккал.", reply_markup=MAIN_MENU_KEYBOARD)
    return ConversationHandler.END


# --- Добавить тренировку ---

async def sport_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    months = get_available_months(user_id=update.effective_user.id) if update.effective_user else []
    today = date.today()
    if (today.year, today.month) not in months:
        months.insert(0, (today.year, today.month))
    if not months:
        months = [(today.year, today.month)]
    keyboard = [[InlineKeyboardButton("📅 Сегодня", callback_data="sport_today")]]
    for y, m in months[:12]:
        keyboard.append([InlineKeyboardButton(f"{calendar.month_name[m]} {y}", callback_data=f"sport_{y}_{m:02d}")])
    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="sport_cancel")])
    await update.message.reply_text("Выбери дату тренировки:", reply_markup=InlineKeyboardMarkup(keyboard))
    return SPORT_MONTH


async def sport_cancel_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Отмена.")
    await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
    return ConversationHandler.END


async def sport_date_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "sport_cancel":
        await sport_cancel_cb(update, context)
        return ConversationHandler.END
    today = date.today()
    if query.data == "sport_today":
        context.user_data["sport_date"] = today
        await query.edit_message_text("Опиши, что и сколько делал (например: бег 30 мин, приседания 3×15):")
        return SPORT_DESC
    if not query.data.startswith("sport_"):
        return SPORT_MONTH
    parts = query.data.replace("sport_", "").split("_")
    if len(parts) == 2:
        try:
            y, m = int(parts[0]), int(parts[1])
            context.user_data["sport_year"], context.user_data["sport_month"] = y, m
            cal = calendar.Calendar(firstweekday=0)
            days = [d for d in cal.itermonthdates(y, m) if d.month == m]
            keyboard = []
            row = []
            for d in days:
                row.append(InlineKeyboardButton(str(d.day), callback_data=f"sportday_{d.day:02d}"))
                if len(row) == 7:
                    keyboard.append(row)
                    row = []
            if row:
                keyboard.append(row)
            keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="sport_cancel")])
            await query.edit_message_text("Выбери день:", reply_markup=InlineKeyboardMarkup(keyboard))
            return SPORT_DAY
        except (ValueError, TypeError):
            pass
    return SPORT_MONTH


async def sport_day_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "sport_cancel":
        await query.edit_message_text("Отмена.")
        await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    if not query.data.startswith("sportday_"):
        return SPORT_DAY
    try:
        day = int(query.data.replace("sportday_", ""))
    except ValueError:
        return SPORT_DAY
    y, m = context.user_data.get("sport_year"), context.user_data.get("sport_month")
    if y and m:
        try:
            context.user_data["sport_date"] = date(y, m, day)
        except ValueError:
            pass
    await query.edit_message_text("Опиши, что и сколько делал (например: бег 30 мин, приседания 3×15):")
    return SPORT_DESC


async def sport_desc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    text = update.message.text.strip()
    if not text:
        await update.message.reply_text("Введи текст тренировки.")
        return SPORT_DESC
    d = context.user_data.get("sport_date")
    tg_user = update.effective_user
    if not d or not tg_user:
        await update.message.reply_text("Ошибка. Начни заново.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    set_entry_exercises_for_day(tg_user.id, tg_user.username or "", d, text)
    await update.message.reply_text(f"Тренировка за {d.strftime('%d.%m.%Y')} записана.", reply_markup=MAIN_MENU_KEYBOARD)
    return ConversationHandler.END


# --- Календарь тренировок ---

async def sports_calendar_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    keyboard = [["👤 Мой календарь тренировок"], ["🌍 Общий календарь тренировок"], ["❌ Отмена"]]
    await update.message.reply_text("Что показать?", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
    return SPORTS_CAL_SCOPE


async def sports_calendar_scope(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not is_allowed(update):
        return ConversationHandler.END
    text = update.message.text.strip().lower()
    if "мой" in text or "👤" in text:
        context.user_data["sports_cal_scope"] = "personal"
        user_id = update.effective_user.id if update.effective_user else None
    elif "общ" in text or "🌍" in text:
        context.user_data["sports_cal_scope"] = "global"
        user_id = None
    else:
        await update.message.reply_text("Отмена.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    context.user_data["sports_cal_user_id"] = user_id
    months = get_available_workout_months(user_id=user_id)
    today = date.today()
    if (today.year, today.month) not in months:
        months.insert(0, (today.year, today.month))
    months = months[:12]
    keyboard = []
    for y, m in months:
        keyboard.append([InlineKeyboardButton(f"{calendar.month_name[m]} {y}", callback_data=f"sportscal_{y}_{m:02d}")])
    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="sportscal_cancel")])
    await update.message.reply_text("Выбери месяц:", reply_markup=InlineKeyboardMarkup(keyboard))
    return SPORTS_CAL_MONTH


async def sports_calendar_month_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "sportscal_cancel":
        await query.edit_message_text("Отмена.")
        await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    if not query.data.startswith("sportscal_"):
        return SPORTS_CAL_MONTH
    parts = query.data.replace("sportscal_", "").split("_")
    if len(parts) < 2:
        return SPORTS_CAL_MONTH
    try:
        y, m = int(parts[0]), int(parts[1])
    except ValueError:
        return SPORTS_CAL_MONTH
    try:
        user_id = context.user_data.get("sports_cal_user_id")
        workouts = load_workouts(user_id=user_id, year=y, month=m)
        img_path = build_sports_calendar_image(year=y, month=m, workouts=workouts, personal_user_id=user_id)
        photo_bytes = img_path.read_bytes()
        await query.message.reply_photo(
            photo=InputFile(io.BytesIO(photo_bytes)),
            caption=f"Календарь тренировок — {calendar.month_name[m]} {y}",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        await query.edit_message_text("Отправлено.", reply_markup=None)
    except Exception as e:
        logger.exception("Ошибка календаря тренировок: %s", e)
        try:
            await query.message.reply_text("❌ Ошибка при построении календаря.", reply_markup=MAIN_MENU_KEYBOARD)
        except Exception:
            pass
    return ConversationHandler.END


# --- Мои записи об упражнениях ---

def delete_workout_from_csv(user_id: int, workout_date: date) -> bool:
    """Удаляет тренировку из workouts.csv и очищает exercises в entries.csv."""
    deleted = False
    if WORKOUTS_CSV.exists():
        rows: List[Dict] = []
        with WORKOUTS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("user_id") == str(user_id) and row.get("date") == workout_date.isoformat():
                    deleted = True
                    continue
                rows.append(row)
        with WORKOUTS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "user_id", "username", "description"])
            writer.writeheader()
            writer.writerows(rows)
    rows_ent: List[Dict] = []
    with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row.setdefault("exercises", "")
            if row["user_id"] == str(user_id) and row["date"] == workout_date.isoformat():
                row["exercises"] = ""
            rows_ent.append(row)
    with ENTRIES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "user_id", "username", "calories", "weight", "exercises"])
        writer.writeheader()
        writer.writerows(rows_ent)
    return deleted


def _build_exercises_keyboard(workouts_list: list) -> tuple:
    """Строит список строк и inline-клавиатуру для просмотра/удаления тренировок."""
    lines = ["📋 *Мои тренировки* (нажми 🗑 для удаления):\n"]
    keyboard = []
    for d, _, _, desc in workouts_list[:20]:
        if not desc:
            continue
        short = desc[:80] + ("…" if len(desc) > 80 else "")
        lines.append(f"📅 {d.strftime('%d.%m.%Y')}: {short}")
        keyboard.append([InlineKeyboardButton(
            f"🗑 {d.strftime('%d.%m.%Y')}",
            callback_data=f"delworkout_{d.isoformat()}"
        )])
    keyboard.append([InlineKeyboardButton("✅ Закрыть", callback_data="delworkout_close")])
    return lines, keyboard


async def view_my_exercises(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    tg_user = update.effective_user
    if not tg_user:
        return
    workouts = load_workouts(user_id=tg_user.id)
    workouts.sort(key=lambda x: x[0], reverse=True)
    if not workouts:
        await update.message.reply_text(
            "Пока нет записей об упражнениях. Добавь тренировку через «🏃 Добавить тренировку».",
            reply_markup=MAIN_MENU_KEYBOARD
        )
        return
    lines, keyboard = _build_exercises_keyboard(workouts)
    await update.message.reply_text(
        "\n".join(lines)[:4000],
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )


async def delete_workout_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Callback для удаления тренировки по inline-кнопке."""
    query = update.callback_query
    await query.answer()
    if query.data == "delworkout_close":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
        return
    if not query.data.startswith("delworkout_"):
        return
    date_str = query.data.replace("delworkout_", "")
    try:
        workout_date = date.fromisoformat(date_str)
    except ValueError:
        return
    tg_user = query.from_user
    if not tg_user:
        return
    deleted = delete_workout_from_csv(tg_user.id, workout_date)
    if deleted:
        workouts = load_workouts(user_id=tg_user.id)
        workouts.sort(key=lambda x: x[0], reverse=True)
        if not workouts:
            await query.edit_message_text("Все тренировки удалены.", reply_markup=None)
            await query.message.reply_text("Главное меню:", reply_markup=MAIN_MENU_KEYBOARD)
            return
        lines, keyboard = _build_exercises_keyboard(workouts)
        await query.edit_message_text(
            "\n".join(lines)[:4000],
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    else:
        await query.answer("Запись не найдена или уже удалена.")


# --- Рейтинг ---

def compute_rankings(users: Dict[int, UserProfile], entries: List[DailyEntry]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """(a) дни в лимите, (b) макс стрик в лимите, (c) кол-во дней с тренировками (из workouts.csv)."""
    by_user_cals: Dict[int, Dict[date, int]] = defaultdict(lambda: defaultdict(int))
    for e in entries:
        by_user_cals[e.user_id][e.date] += e.calories
    by_user_ex: Dict[int, Set[date]] = defaultdict(set)
    for d, uid, _, desc in load_workouts():
        if desc:
            by_user_ex[uid].add(d)
    limit_ok_count: Dict[int, int] = {}
    max_streak: Dict[int, int] = {}
    for uid, profile in users.items():
        limit = profile.calorie_limit
        daily = by_user_cals.get(uid, {})
        limit_ok_count[uid] = sum(1 for d, c in daily.items() if c > 0 and c <= limit)
        streak, best = 0, 0
        for d in sorted(daily.keys(), reverse=True):
            if daily[d] > 0 and daily[d] <= limit:
                streak += 1
            else:
                best = max(best, streak)
                streak = 0
        max_streak[uid] = max(best, streak)
    a = [(users[uid].username or f"id{uid}", limit_ok_count.get(uid, 0)) for uid in users]
    b = [(users[uid].username or f"id{uid}", max_streak.get(uid, 0)) for uid in users]
    c = [(users[uid].username or f"id{uid}", len(by_user_ex.get(uid, set()))) for uid in users]
    a.sort(key=lambda x: x[1], reverse=True)
    b.sort(key=lambda x: x[1], reverse=True)
    c.sort(key=lambda x: x[1], reverse=True)
    return a, b, c


async def show_ranking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    users = load_users()
    if not users:
        await update.message.reply_text("Нет данных для рейтинга.", reply_markup=MAIN_MENU_KEYBOARD)
        return
    entries = []
    ensure_csv_files()
    with ENTRIES_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                d = date.fromisoformat(row["date"])
                entries.append(DailyEntry(d, int(row["user_id"]), row.get("username", ""), int(row.get("calories") or 0), None, row.get("exercises", "")))
            except (ValueError, KeyError):
                pass
    a, b, c = compute_rankings(users, entries)
    lines = [
        "🏆 Рейтинг",
        "",
        "📊 Дней в рамках лимита (не перебрал):",
    ]
    for i, (name, cnt) in enumerate(a, 1):
        lines.append(f"  {i}. @{name}: {cnt}")
    lines.extend(["", "🔥 Лучший стрик подряд (дней в лимите):"])
    for i, (name, cnt) in enumerate(b, 1):
        lines.append(f"  {i}. @{name}: {cnt}")
    lines.extend(["", "🏃 Больше всего тренировок (дней с записью):"])
    for i, (name, cnt) in enumerate(c, 1):
        lines.append(f"  {i}. @{name}: {cnt}")
    await update.message.reply_text("\n".join(lines), reply_markup=MAIN_MENU_KEYBOARD)


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
# --- Изменить диету ---

async def change_diet_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало процесса смены диеты с предупреждением."""
    if not is_allowed(update):
        return ConversationHandler.END
    keyboard = ReplyKeyboardMarkup(
        [["✅ Да, хочу изменить диету"], ["❌ Отмена"]],
        resize_keyboard=True,
    )
    await update.message.reply_text(
        "⚠️ *ВНИМАНИЕ!* ⚠️\n\n"
        "При изменении диеты будут *удалены все данные профиля*:\n"
        "• Текущий вес, целевой вес, лимит калорий\n"
        "• Рост, возраст, пол, активность\n\n"
        "История калорий и тренировок *сохранится*.\n\n"
        "Ты уверен, что хочешь начать заново?",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    return CHANGE_DIET_CONFIRM


async def change_diet_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Подтверждение — удаляем профиль и запускаем онбординг заново."""
    if not is_allowed(update):
        return ConversationHandler.END
    text = update.message.text.strip().lower()
    if "да" not in text and "изменить" not in text:
        await update.message.reply_text("Отмена. Возвращаемся в главное меню.", reply_markup=MAIN_MENU_KEYBOARD)
        return ConversationHandler.END
    tg_user = update.effective_user
    if not tg_user:
        return ConversationHandler.END
    users = load_users()
    if tg_user.id in users:
        del users[tg_user.id]
        save_users(users)
    context.user_data.clear()
    await update.message.reply_text(
        "✅ Профиль удалён. Начнём заново!\n\n"
        "Введи текущий вес в кг (например: 83.5):",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ONBOARD_WEIGHT


def build_application() -> "ApplicationBuilder":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")

    from telegram.request import HTTPXRequest
    proxy_url = os.getenv("TELEGRAM_PROXY", "socks5://QqQT5Y7N9w:KSMKLRCyFg@45.147.31.223:39785")
    request = HTTPXRequest(proxy=proxy_url)
    app = ApplicationBuilder().token(token).request(request).build()

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
        states={
            ADD_CALORIES: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_calories)],
            ADD_CALORIES_CHOICE: [CallbackQueryHandler(handle_add_calories_day_cb, pattern="^add_cal_day_")],
            ADD_CALORIES_BARCODE: [
                MessageHandler(filters.PHOTO, handle_add_calories_barcode),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_calories_barcode),
            ],
            ADD_CALORIES_GRAMMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_calories_gramms)],
        },
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
            # ✅ ДОБАВЬТЕ pattern="^stats_"
            STATS_MONTH_SELECT: [CallbackQueryHandler(stats_month_callback, pattern="^stats_")],
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
    # Не добавляем глобальный /cancel — тогда /cancel обрабатывается только активным диалогом
    # и состояние агента/других сценариев корректно сбрасывается.
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("energy", send_energy))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.Regex("^⚡ Получить заряд бодрости$"),
            send_energy,
        )
    )

    # Агент — только по команде /agent, чтобы не включался от других сообщений
    agent_conv = ConversationHandler(
        entry_points=[CommandHandler("agent", agent_start)],
        states={
            AGENT_CHAT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, agent_message),
                CommandHandler("cancel", cancel),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(agent_conv)
    # Кнопка «💬 Агент» не запускает диалог — только подсказка
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.Regex("^💬 Агент$"),
            agent_button_hint,
        )
    )

    # Изменить ККЛ за день
    edit_cal_conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^✏️ Изменить ККЛ за день$"), edit_cal_start),
        ],
        states={
            EDIT_CAL_MONTH: [CallbackQueryHandler(edit_cal_month_cb, pattern="^editcal_")],
            EDIT_CAL_DAY: [
                CallbackQueryHandler(edit_cal_day_cb, pattern="^editcalday_"),
                CallbackQueryHandler(edit_cal_cancel_cb, pattern="^editcal_cancel$"),
            ],
            EDIT_CAL_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, edit_cal_value)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(edit_cal_conv)

    # Добавить тренировку
    sport_conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^🏃 Добавить тренировку$"), sport_start),
        ],
        states={
            SPORT_MONTH: [CallbackQueryHandler(sport_date_cb, pattern="^sport_")],
            SPORT_DAY: [
                CallbackQueryHandler(sport_day_cb, pattern="^sportday_"),
                CallbackQueryHandler(sport_cancel_cb, pattern="^sport_cancel$"),
            ],
            SPORT_DESC: [MessageHandler(filters.TEXT & ~filters.COMMAND, sport_desc)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(sport_conv)

    # Календарь тренировок
    sports_cal_conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^📋 Календарь тренировок$"), sports_calendar_start),
        ],
        states={
            SPORTS_CAL_SCOPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, sports_calendar_scope)],
            SPORTS_CAL_MONTH: [CallbackQueryHandler(sports_calendar_month_cb, pattern="^sportscal_")],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(sports_cal_conv)

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^📋 Мои записи об упражнениях$"), view_my_exercises))
    app.add_handler(CallbackQueryHandler(delete_workout_cb, pattern="^delworkout_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^🏆 Рейтинг$"), show_ranking))

    # Изменить диету
    change_diet_conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex("^🔄 Изменить диету$"), change_diet_start),
        ],
        states={
            CHANGE_DIET_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, change_diet_confirm)],
            ONBOARD_WEIGHT:      [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_weight)],
            ONBOARD_TARGET:      [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_target)],
            ONBOARD_LIMIT:       [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_limit)],
            ONBOARD_HEIGHT:      [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_height)],
            ONBOARD_AGE:         [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_age)],
            ONBOARD_GENDER:      [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_gender)],
            ONBOARD_ACTIVITY:    [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_activity)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(change_diet_conv)

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