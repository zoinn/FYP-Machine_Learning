import requests

headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API-TOKEN}'
    }
response = requests.get('https://www.jblanked.com/news/api/list/',headers=headers)

def print_news():
    news = response.text.split('}')
    #Prints the last 5 latest news articles
    print(news[1:4])
