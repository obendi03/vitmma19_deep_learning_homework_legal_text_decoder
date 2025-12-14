import os
import requests
import zipfile

url = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1&xsdata=MDV8MDJ8fDIyOTc1YmYyMWMzNzQyODFlZWZhMDhkZTM3YmNkMjdifDZhMzU0OGFiNzU3MDQyNzE5MWE4NThkYTAwNjk3MDI5fDB8MHw2MzkwMDk0ODEyNTgwNjE0Njd8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDNSbFlXMXpMekU1T2xSM1NIcHViVlpTVlVKUGFGUjFRVTFyWlc1blIyVlhTSEkzYjB0WVNWQkNTamxxTWtKbkxVdFdkMnN4UUhSb2NtVmhaQzUwWVdOMk1pOWphR0Z1Ym1Wc2N5OHhPVHBVZDBoNmJtMVdVbFZDVDJoVWRVRk5hMlZ1WjBkbFYwaHlOMjlMV0VsUVFrbzVhakpDWnkxTFZuZHJNVUIwYUhKbFlXUXVkR0ZqZGpJdmJXVnpjMkZuWlhNdk1UYzJOVE0xTVRNeU5ETTJPQT09fDBiYmVmZWIwYWJmOTRkZTFlZWZhMDhkZTM3YmNkMjdifGRlNDNhNjEyMWZmNzQxOTk4OGJiYzk4ZWMzZjU4MTdk&sdata=MEVHaDVlSkQrR09NUWRsbFV1SXBTMDNEMDV5OUlWV0hEbmlVcEI5YWNuTT0%3D&ovuser=6a3548ab-7570-4271-91a8-58da00697029%2Colah.bendeguzistvan%40edu.bme.hu"
data_dir = "/app/data"
os.makedirs(data_dir, exist_ok=True)

# ----------------------------
# Download ZIP
# ----------------------------
zip_path = os.path.join(data_dir, "downloaded.zip")

print(f"Downloading to {zip_path} ...")
response = requests.get(url)
with open(zip_path, "wb") as f:
    f.write(response.content)
print("File downloaded successfully!")

# ----------------------------
# Extract ZIP
# ----------------------------
# extract_path = os.path.join(data_dir, "extracted")
# os.makedirs(extract_path, exist_ok=True)
#
# with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     zip_ref.extractall(extract_path)
# print(f"ZIP extracted to {extract_path}")