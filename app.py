import requests

print(requests.get("https://hackatime.hackclub.com/api/v1/my/heartbeats",  headers={'Authorization': 'Bearer f9c03b2f-9c57-4396-a94a-25abc1d9c1d1'}))