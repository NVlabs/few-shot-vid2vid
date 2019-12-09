import os
from download_gdrive import *

file_id = '1NyxrzJbHgDOpf-nRJhfxsHqWNDW_D1TV'
chpt_path = './datasets/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
destination = os.path.join(chpt_path, 'datasets.zip')
download_file_from_google_drive(file_id, destination) 
unzip_file(destination, chpt_path)