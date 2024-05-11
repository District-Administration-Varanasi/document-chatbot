import requests
from bs4 import BeautifulSoup
import os

# Step 1: Fetch the website content
website_url = 'https://upvidhai.gov.in/Act.aspx'  # Ensure this is correct
response = requests.get(website_url)
print("HTTP Status Code:", response.status_code)

# Check for successful response
if response.status_code != 200:
    raise Exception("Failed to fetch the website.")

# Step 2: Parse the content to find PDF links
soup = BeautifulSoup(response.content, 'html.parser')

pdf_links = []
for link in soup.find_all('a', href=True):
    if link['href'].endswith('.pdf'):  # Check for .pdf files
        pdf_links.append(link['href'])

print("PDF Links Found:", pdf_links)

# Step 3: Download and save each PDF
download_directory = './pdf_downloads'
if not os.path.exists(download_directory):
    os.makedirs(download_directory)

for pdf_link in pdf_links:
    # Handle relative links by joining them with the base URL
    if not pdf_link.startswith('http'):
        pdf_link = requests.compat.urljoin(website_url, pdf_link)
    
    print("Downloading:", pdf_link)
    
    pdf_response = requests.get(pdf_link)
    pdf_filename = os.path.join(download_directory, pdf_link.split('/')[-1])

    with open(pdf_filename, 'wb') as pdf_file:
        pdf_file.write(pdf_response.content)

print(f'Downloaded {len(pdf_links)} PDF(s) to {download_directory}.')
