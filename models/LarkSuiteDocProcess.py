import requests
import json

def get_access_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
    "app_id": "cli_a9b5c4a31379dc",
    "app_secret": "8JnwXAQLpQ5YqPamNYEZ6eWeoN5hLG6Kfa"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["tenant_access_token"]
    else:
        raise Exception(f"Failed to get access token: {response.status_code} {response.text}")
    
def get_table_name(url):
    access_token = get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json()
        name = results['data']['spreadsheet']['title']
        token = results['data']['spreadsheet']['token']
        return name, token
    else:
        raise Exception(f"Failed to get table name: {response.status_code} {response.text}")
    
def get_sheet(token):
    access_token = get_access_token()
    url = f"https://open.feishu.cn/open-apis/sheets/v3/spreadsheets/{token}/sheets/query"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json()
        return results
    else:
        raise Exception(f"Failed to get sheet: {response.status_code} {response.text}")


def preprocess_karksuite_doc(token, sheet_id):
    access = get_access_token()
    url = f'https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{token}/values/{sheet_id}!A2:Z100?valueRenderOption=ToString&dateTimeRenderOption=FormattedString'

    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        base = []
        results = response.json()
        for i, content_value in enumerate(results['data']['valueRange']['values']):
            none_lists = all(x is None for x in content_value)
            if none_lists:
                continue
            base.append(content_value)
        
        column_name = base[0]
        fliter_column_name = [col for col in column_name if col is not None]

        all_text = []
        for row_data in base[1:]:
            info = []
            # 确保行数据长度与列名数量一致
            for i, col_name in enumerate(fliter_column_name):
                if i < len(row_data):
                    info.append(f"{col_name}: {row_data[i]}")
                else:
                    info.append(f"{col_name}: ")
            
            # 将当前行的所有列信息连接成一个字符串
            row_text = ",".join(info)
            all_text.append(row_text)
        return all_text
    else:
        raise Exception(f"Failed to get sheet: {response.status_code} {response.text}")

