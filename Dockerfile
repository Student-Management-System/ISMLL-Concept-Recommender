FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# Issue with Pytorch and invalid ELF header, solution based on: https://discuss.pytorch.org/t/importerror-home-name-anaconda3-envs-tf21-lib-python3-7-site-packages-torch-lib-libnvtoolsext-so-1-invalid-elf-header/91764/5
RUN pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

# Environment Variables
ENV FLASK_APP=Webservice.py
ENV FLASK_ENV=production

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--no-debugger"]
EXPOSE 5000