import base64
import io
import requests
from PIL import Image
from pyzbar.pyzbar import decode


def get_calories_from_image(image_source):
    """
    Принимает путь к файлу или base64 строку изображения.
    Возвращает словарь с информацией о продукте и калорийностью.
    """

    # 1. Загрузка и подготовка изображения
    try:
        if image_source.startswith(('http://', 'https://', 'data:image')):
            # Если это URL или Base64
            if image_source.startswith('data:image'):
                # Это Base64 (удаляем заголовок data:image/...;base64,)
                header, encoded = image_source.split(',', 1)
                image_data = base64.b64decode(encoded)
            else:
                # Это URL картинки (опционально, если захотите передавать ссылки)
                response = requests.get(image_source)
                image_data = response.content

            image = Image.open(io.BytesIO(image_data))
        else:
            # Если это путь к файлу
            image = Image.open(image_source)

    except Exception as e:
        return {"error": f"Не удалось открыть изображение: {str(e)}"}

    # 2. Распознавание штрихкода
    decoded_objects = decode(image)
    print(decoded_objects)
    if not decoded_objects:
        return {"error": "Штрихкод не найден на изображении"}

    # Берем первый найденный штрихкод
    barcode_data = decoded_objects[0].data.decode('utf-8')
    print(f"Найден штрихкод: {barcode_data}")

    # 3. Запрос к Open Food Facts API
    api_url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_data}.json"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Проверка на ошибки HTTP
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Ошибка при запросе к API: {str(e)}"}

    # 4. Парсинг ответа
    if data.get('status') == 0:
        return {"error": "Продукт с таким штрихкодом не найден в базе Open Food Facts"}

    product = data.get('product', {})
    product_name = product.get('product_name', 'Название неизвестно')

    # Получаем нутриенты
    nutriments = product.get('nutriments', {})

    # Калорийность обычно хранится в energy-kcal_100g (на 100г)
    # Иногда может быть просто energy-kcal (на порцию), но стандарт - на 100г
    calories = nutriments.get('energy-kcal_100g')

    # Если нет данных на 100г, пробуем найти просто energy-kcal
    if calories is None:
        calories = nutriments.get('energy-kcal')

    return {
        "barcode": barcode_data,
        "product_name": product_name,
        "calories_per_100g": calories,
        "full_data_url": f"https://world.openfoodfacts.org/product/{barcode_data}"
    }


# --- Примеры использования ---

if __name__ == "__main__":
    # Пример 1: Передача пути к файлу
    # Замените 'barcode.jpg' на реальный путь к вашей картинке
    # result = get_calories_from_image('barcode.jpg')

    # Пример 2: Передача Base64 строки (короткий пример для теста)
    # В реальности строка будет очень длинной
    #base64_string = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    # (Примечание: эта base64 строка - просто пустая картинка, она не сработает, это для демонстрации формата)

    # Чтобы протестировать, раскомментируйте строку ниже с реальным путем к файлу со штрихкодом:
    print(get_calories_from_image("photo_2026-02-28_15-32-02.jpg"))

    print("Функция готова к использованию. Передайте путь к изображению со штрихкодом.")