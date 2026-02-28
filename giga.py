import requests, json, dotenv, os
dotenv.load_dotenv()
import requests
TOKEN= ""
def get_oauth_token():
    global TOKEN
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    payload={
      'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Accept': 'application/json',
      'RqUID': '019ca047-89b6-7cb8-90a7-e75181414f5f',
      'Authorization': f'Basic {os.getenv("GIGACHAT_TOKEN")} '
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    TOKEN = json.loads(response.text).get('access_token')
    return TOKEN
def response_giga(context):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = {
        "model": "GigaChat",
        "messages": [
            {"role": "system", "content": "Ты экспертный помощник по здоровому образу жизни, и личный тренер. Будь максимально краток. Твои ответы должны быть максимально информативными"},
            {"role": "user", "content": context}
        ]
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
    return response
def validate_result(res):
    try:
        data = json.loads(res.text)

        if isinstance(data, dict) and 'choices' in data:
            choices = data['choices']
        else:
            choices = data

        if choices and len(choices) > 0:
            return choices[0]['message']['content']

        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Ошибка при извлечении content: {e}")
        return None
def generate_answer(context: str) -> dict:
    response = response_giga(context)
    if response.status_code == 200:
        return validate_result(response)
    else:
        get_oauth_token()
        response = response_giga(context)
        if response.status_code == 200:
            return validate_result(response)
        else:
            print(f"Ошибка: {response.status_code}, {response.text}")
            return {}

if __name__ == '__main__':
    print(generate_answer("Как мне накачаться?"))