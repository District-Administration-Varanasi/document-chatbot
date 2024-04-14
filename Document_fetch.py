import requests
from bs4 import BeautifulSoup
import os
import time

def download_pdf(url, filename):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

def find_and_download_pdfs(base_url, directory_name, delay, filename_prefix, pdf_link_pattern=None):
    for _ in range(5):
        try:
            response = requests.get(base_url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False)
            if response.status_code == 200:
                break
        except requests.ConnectionError as e:
            print(f"Retrying due to error: {e}")
            time.sleep(5)
    else:
        print("Failed to retrieve the webpage after several attempts.")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)

    if pdf_link_pattern:
        pdf_urls = [base_url + link['href'] for link in links if pdf_link_pattern in link['href']]
    else:
        pdf_urls = [link['href'] for link in links if link['href'].endswith('.pdf')]
        pdf_urls = [link if link.startswith('http') else base_url + link for link in pdf_urls]

    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    doc_counter = 1
    for pdf_url in pdf_urls:
        filename = os.path.join(directory_name, f'{filename_prefix}{doc_counter}.pdf')
        download_pdf(pdf_url, filename)
        doc_counter += 1
        time.sleep(delay)

directory = 'Documents'
website_url_upvidhai = "https://upvidhai.gov.in/"
find_and_download_pdfs(website_url_upvidhai, directory, 1, 'upvidhaidocument')

website_url_shasanadesh = "https://shasanadesh.up.gov.in/"
find_and_download_pdfs(website_url_shasanadesh, directory, 5, 'shashandesdocument', "frmPDF.aspx?qry=")
