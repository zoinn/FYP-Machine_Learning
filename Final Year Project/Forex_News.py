import requests

key = 'W3i4Rs1x.86lRS3JQ1JTBnVz5xNEdVK2b2kX4w4ZL'

headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}'
    }
response = requests.get('https://www.jblanked.com/news/api/list/',headers=headers)

def print_news():
    news = response.text.split('}')
    #Prints the last 5 latest news articles
    print(news[1:4])