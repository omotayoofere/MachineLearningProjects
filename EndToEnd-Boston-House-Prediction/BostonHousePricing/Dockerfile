#from the docker hub, a base image which has linux is gotten and python3.7 is installed
FROM python:3.7 
#Copies whatever code is in the repo/current dir and place in another location
COPY . /app
#Makes the new location the current working directory
WORKDIR /app
#installs all dependencies
RUN pip install -r requirements.txt
#expose some ports within the docker container in other to access the apps in that container
EXPOSE $PORT
#helps to run the entire application inside the heroku cloud. workers allows request be divided into instances
#the exposed port will be getting binded to the local IP address
#the IP address will be the local address in the heroku cloud
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app