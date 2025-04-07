from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import zipfile
from dotenv import load_dotenv
import warnings
import os

# Skip warning.
warnings.filterwarnings("ignore")
# load the Environment Variables.
load_dotenv()


user_email = os.getenv('USEREMAIL')
user_passwd = os.getenv('PASSWORD')
download_path = os.getenv('DOWNLOAD_PATH')

path_to_chromedriver = 'https://cpistore.internal.ericsson.com/'
driver = webdriver.Firefox()
driver.get(path_to_chromedriver)
time.sleep(3)

username_field = driver.find_element(By.NAME, "loginfmt")
username_field.send_keys(user_email)
login_button = driver.find_element(By.ID, "idSIButton9")
login_button.click()
time.sleep(3)

password_field = driver.find_element(By.NAME, "passwd")
password_field.send_keys(user_passwd)

login_button = driver.find_element(By.ID, "idSIButton9")
login_button.click()
time.sleep(15)


url = path_to_chromedriver + "download/#?id=46888"
driver.get(url)
time.sleep(10)

login_button = driver.find_element(By.ID, "download-button")
login_button.click()

filename = "en_lzn7640060_r7b(1).elxl"

# check if file downloaded file path exists, if not then sleep for 3sec
while not os.path.exists(download_path + filename):
    time.sleep(3)

driver.quit()


with zipfile.ZipFile(download_path + filename, 'r') as zip_ref:
    zip_ref.extractall(download_path + "extract")
