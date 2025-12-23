import requests
import json

# 定义请求地址
url = "http://localhost:8080/api/rag/chat"

# 定义问题 (基于我们之前 ingest 的 legal_docs.txt)
payload = {
    "query": "如果甲方逾期支付本金，需要承担什么违约责任？"
}

print(f"正在发送问题: {payload['query']} ...")

try:
    # 发送 POST 请求
    response = requests.post(url, json=payload)
    
    # 打印结果
    if response.status_code == 200:
        data = response.json()
        print("\n=== AI 回复 ===")
        print(data["response"])
        print("===============")
    else:
        print(f"Error: {response.status_code} - {response.text}")

except Exception as e:
    print(f"请求失败: {e}")