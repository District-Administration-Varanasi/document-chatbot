import requests
from bs4 import BeautifulSoup
import os

def download_pdfs(url, download_folder):
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            if href.endswith('.pdf'):
                pdf_url = requests.compat.urljoin(url, href)
                file_name = os.path.join(download_folder, os.path.basename(href))
            elif 'frmPDF.aspx' in href:
                pdf_url = requests.compat.urljoin(url, href)
                file_name = os.path.join(download_folder, href.split('=')[-1] + ".pdf")
            else:
                continue  
            
            try:
                with requests.get(pdf_url, stream=True, verify=False) as pdf_req:
                    if pdf_req.status_code == 200:
                        with open(file_name, 'wb') as pdf_file:
                            for chunk in pdf_req.iter_content(chunk_size=8192):
                                pdf_file.write(chunk)
                        print(f'Downloaded {file_name}')
                    else:
                        print(f"Failed to download PDF from {pdf_url} - Status code {pdf_req.status_code}")
            except Exception as e:
                print(f"Error downloading {pdf_url}: {str(e)}")

urls = [
    'https://upvidhai.gov.in',  
    'https://shasanadesh.up.gov.in/'     
]

download_folder = "Government Document Chatbot\\downloaded_folder"

for website_url in urls:
    download_pdfs(website_url, download_folder)
