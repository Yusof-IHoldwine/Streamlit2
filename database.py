import os
import streamlit as st
from deta import Deta
from dotenv import load_dotenv

load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")
#command
#heroku config:set DETA_KEY = c0zmo5mv_twfsQAydJypNKSMY8CAVudZs2fJxL3f9

#Initialize with a project key
#deta = Deta(DETA_KEY)

# Connect to Deta Base with your Project Key
deta = Deta(st.secrets["deta_key"])

#How to create/connect a database
db = deta.Base("users_db")

def insert_user(username, name, password):
    """Returns the uer on a successful user creation, otherwise raises and error"""
    return db.put({"key": username, "name": name, "password": password})

def fetch_all_users():
    """Returns a dict of all users"""
    res = db.fetch()
    return res.items

def get_user(username):
    """If not found, the function will return None"""
    return db.get(username)

def update_user(username, updates):
    """If the item is updated, returns None. Otherwise, an exception is raised"""
    return db.update(updates, username)

def delete_user(username):
    """Always returns None, even if the key does not exist"""
    return db.delete(username)

