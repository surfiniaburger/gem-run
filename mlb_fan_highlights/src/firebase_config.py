import firebase_admin
from firebase_admin import credentials

#Load the service account key from json file
cred = credentials.Certificate("./gem-rush-007-firebase-adminsdk-pzp02-b7a2022e8b.json") #replace with your service account key json file path

#Initialize the firebase admin app
firebase_admin.initialize_app(cred)
