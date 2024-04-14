import requests
from bs4 import BeautifulSoup
import os
import time

def download_pdf(url, filename):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename} from {url}. Error: {e}")

def find_and_download_pdfs(base_url):
    response = requests.get(base_url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    pdf_urls = [base_url + link['href'] for link in links if "frmPDF.aspx?qry=" in link['href']]

    if not os.path.exists('pdf_downloads'):
        os.mkdir('pdf_downloads')
    doc_counter = 1
    for pdf_url in pdf_urls:
        filename = os.path.join('pdf_downloads', f'doc{doc_counter}.pdf')
        download_pdf(pdf_url, filename)
        doc_counter += 1
        time.sleep(5)  \

website_url = "https://shasanadesh.up.gov.in/"
find_and_download_pdfs(website_url)
